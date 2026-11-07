#!/usr/bin/env python3
"""Measure the right wrist view on the moving belt using the installed M4 guards."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

from ament_index_python.packages import get_package_prefix
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from sorting_interfaces.action import ExecuteArmTrajectory

sys.path.insert(0, str(Path(get_package_prefix('sorting_supervisor')) / 'lib/sorting_supervisor'))
from m4a_runtime import ns
from m4c_common import config
from m4c_core import bind_capture
from m4c_evaluator import MovingEvaluator
from m4c_timing import FullCollectionWindow
from m4d_common import checked_preflight
from m4d_runtime import IntegratedExecutor, IntegratedInspector
from classifier_core import digest, require
from m2b_runtime import normalize_plan, plan_request
from ownership_core import NAMES


FIXTURES = ('red_partial', 'green_partial')


class Evaluator(MovingEvaluator):
    def __init__(self, settings):
        super().__init__(settings)
        self.last_transport = None
        self.last_contact_message = None
        self.contact_guard_armed = False

    def receive_contact(self, message):
        self.last_contact_message = message
        if self.contact_guard_armed:
            super().receive_contact(message)

    def arm_contact_guard(self):
        self.contact_guard_armed = True
        self.receive_contact(self.last_contact_message)
        require(self.failure is None, self.failure)

    def receive_transport(self, message):
        self.last_transport = json.loads(message.data)
        super().receive_transport(message)


class Inspector(IntegratedInspector):
    def __init__(self, folder, report, settings, joint_targets):
        super().__init__(folder, report, settings)
        self.joint_targets = joint_targets

    def move(self, name):
        # Use one checked joint target per pose. MoveIt still checks the path.
        require(self.feedback_armed, 'Joint feedback readiness has not been confirmed')
        self.events.append(dict(state='PLAN', pose=name, enter_ns=self.now(), wall=time.monotonic()))
        seed = dict(self.q)
        request = plan_request(seed, self.c['poses'][name], self.c['hand_quaternion_xyzw'], self.settings,
                               joint_goal=self.joint_targets[name])
        response = self.future(self.planner.call_async(request), timeout=12.)
        result = response.motion_plan_response
        require(result.error_code.val == 1, f'Inspection plan failed: {result.error_code.val}')
        trajectory = normalize_plan(result.trajectory.joint_trajectory)
        require(len(trajectory.points) >= 2, 'Inspection plan is empty')
        duration = ns(trajectory.points[-1].time_from_start) / 1e9
        require(0 < duration < self.c['maximum_motion_sim_s'], 'Inspection plan exceeds duration bound')
        require(max(abs(self.q[n] - seed[n]) for n in NAMES) < .025, 'Robot moved away from planning seed')
        epoch, deadline = self.acquire('PLANNED', duration + 5.)
        goal = ExecuteArmTrajectory.Goal(epoch=epoch, deadline=deadline, trajectory=trajectory)
        handle = self.future(self.action.send_goal_async(goal))
        require(handle.accepted, 'Command gate rejected inspection trajectory')
        self.active_goal = handle
        self.events.append(dict(state='MOVE', pose=name, enter_ns=self.now(), planned_duration_s=duration))
        response = self.future(handle.get_result_async(), timeout=self.c['maximum_motion_wall_s'])
        self.active_goal = None
        require(response.status == GoalStatus.STATUS_SUCCEEDED and response.result.success, response.result.reason)
        self.acquire('HOLD')
        self.wait_sim(self.c['settle_sim_s'])
        self.refine(name)
        error, angle = self.pose_error(name)
        require(error < self.c['maximum_position_error_m'] and angle < self.c['maximum_orientation_error_rad'],
                f'Physical inspection endpoint error: {error:.6f} m, {angle:.6f} rad')
        self.events.append(dict(state='SETTLED', pose=name, enter_ns=self.now(), position_error_m=error,
                                orientation_error_rad=angle))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def costs(events, view):
    timeline = [(e['state'], e['pose'], e['enter_ns']) for e in events
                if e['state'] in ('PLAN', 'SETTLED', 'CAPTURE_COMPLETE')]
    require([(state, pose) for state, pose, _ in timeline] == [
        ('PLAN', 'assess'), ('SETTLED', 'assess'), ('CAPTURE_COMPLETE', 'assess'),
        ('PLAN', view), ('SETTLED', view), ('CAPTURE_COMPLETE', view),
        ('PLAN', 'assess'), ('SETTLED', 'assess')], 'Unexpected wrist motion or capture sequence')
    times = [stamp for _, _, stamp in timeline]
    require(all(end >= start for start, end in zip(times, times[1:])), 'Motion timestamps are out of order')
    return dict(travel_and_settle_sim_s=(times[4] - times[3]) / 1e9,
                capture_sim_s=(times[5] - times[4]) / 1e9,
                return_and_settle_sim_s=(times[7] - times[6]) / 1e9,
                total_sim_s=(times[7] - times[3]) / 1e9)


def observations(samples, view, label, threshold):
    result = {}
    for pose in ('assess', view):
        frames = [s for s in samples if s['pose'] == pose]
        require(len(frames) == 5, 'Each pose must supply five fresh wrist frames')
        result[pose] = dict(x_range_m=[min(s['handle_xyz'][0] for s in frames),
                                      max(s['handle_xyz'][0] for s in frames)],
                            confidence=[s['class_evidence']['confidence'] for s in frames],
                            true_class_probability=[s['class_evidence']['probabilities'][label] for s in frames],
                            accepted_frames=sum(s['class_evidence']['confidence'] >= threshold for s in frames))
    return result


def prerequisites(gate, m5a, fixture):
    summary, manifest = read(gate / 'summary.json'), read(gate / 'manifest.json')
    require(summary['status'] == 'PASS' and summary['full_M4_gate_passed']
            and summary['successful_episodes'] == 10
            and summary['manifest_sha256'] == digest(gate / 'manifest.json'), 'Complete passing M4 gate required')
    preflight, frozen = checked_preflight(manifest['preflight_directory'])
    require(manifest['preflight_sha256'] == digest(preflight / 'preflight.json')
            and manifest['installed_manifest'] == frozen['installed_manifest'], 'M4 gate and preflight differ')
    report, split = read(m5a / 'report.json'), read(m5a / 'split-manifest.json')
    require(report['checkpoint'] == 'M5A' and report['status'] == 'PASS'
            and report['calibration_parameters_sha256'] == digest(m5a / 'calibration-parameters.json')
            and read(m5a / 'calibration-parameters.json')['split_manifest_sha256'] == digest(m5a / 'split-manifest.json')
            and split['m4_gate_sha256'] == digest(gate / 'summary.json')
            and split['frozen_model_sha256'] == frozen['model_sha256'], 'M5A evidence differs from the M4 baseline')
    poses = preflight / 'inspection/m4b/m4a/native/views.yaml'
    require(split['poses_sha256'] == digest(poses), 'Native wrist pose file differs from M5A')
    native = yaml.safe_load(poses.read_text())
    require(native['status'] == 'PASS', 'Native wrist poses did not pass preflight')
    joint_targets = {pose: native['views'][pose]['joints'] for pose in ('assess', 'inspect_right')}
    require(all(len(q) == 7 and all(math.isfinite(v) for v in q) for q in joint_targets.values()),
            'Invalid native wrist joint targets')
    left, = [row for row in summary['cases'] if row['fixture'] == fixture and row['arrival'] == 'upstream']
    path = Path(left['report_path'])
    require(left['status'] == 'PASS' and left['policy_path'] == 'inspect'
            and left['report_sha256'] == digest(path), 'M4 left-view reference differs')
    data = read(path)
    events_path = path.with_name('inspection-events.json')
    label = int(fixture.startswith('green'))
    left_reference = dict(report=str(path), report_sha256=digest(path), events_sha256=digest(events_path),
                          costs=costs(read(events_path), 'inspect_left'),
                          observations=observations(data['inspection']['samples'], 'inspect_left', label,
                                                    report['confidence_threshold']))
    return preflight, frozen, left_reference, joint_targets


def capture(control, evaluator, pose, out, settings):
    control.move(pose)
    samples = control.capture(pose, out)
    for sample in samples:
        sample['class_evidence'] = sample.pop('classifier_diagnostic_only')
        sample['track_binding'] = bind_capture(sample, list(control.history), control.episode, control.bound_track,
                                               settings, control.filter_config)
        lo, hi = settings['classification_x_ranges']['assess' if pose == 'assess' else 'inspect_left']
        require(lo <= sample['handle_xyz'][0] <= hi, 'Wrist capture is outside the test position interval')
    control.until(lambda: evaluator.have_pairs(samples), timeout=2.)
    checks = evaluator.assess(samples, settings['maximum_localization_error_m'])
    return samples, checks


def run(args, preflight, frozen, left_reference, joint_targets):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    folder = preflight / 'inspection'
    inspection = read(folder / 'preflight.json')
    settings = config()
    rclpy.init()
    executor = IntegratedExecutor(settings['callback_budget_s'])
    control, evaluator = Inspector(folder, inspection, settings, joint_targets), Evaluator(settings)
    executor.inspector = control
    executor.startup_clock_pending = control.startup_clock_pending
    executor.diagnostic_context = control.timing_context
    executor.add_node(control)
    executor.add_node(evaluator)
    gc_window = FullCollectionWindow()
    report = dict(checkpoint='M5B', status='FAIL', fixture=args.fixture, arrival='upstream',
                  preflight_directory=str(preflight), preflight_sha256=digest(preflight / 'preflight.json'),
                  model_sha256=control.model.sha, probe_sha256=digest(__file__),
                  joint_targets=joint_targets,
                  full_M5_gate_passed=False, physical_pickup_permitted=False,
                  output_directory=str(out), final_hold_status='NOT_RUN', final_belt_stop_status='NOT_RUN')
    samples, checks = [], []
    try:
        control.startup()
        require(control.running_scene_path == inspection['scene_paths'][args.fixture + ':upstream'],
                'Terminal 1 must launch the matching upstream fixture')
        require(control.model.sha == frozen['model_sha256'], 'Live classifier differs from the frozen M4 model')
        control.until(lambda: evaluator.last_contact_message is not None, timeout=5.)
        evaluator.arm_contact_guard()
        for topic in ('/sorting/ground_truth/camera_sample', '/sorting/ground_truth/conveyor_state',
                      '/sorting/contact_state'):
            subscriptions = evaluator.get_subscriptions_info_by_topic(topic)
            require(subscriptions and all(s.node_name == evaluator.get_name() for s in subscriptions),
                    'Physical truth has a subscriber outside the evaluator')
        # Use the tested left allocation as a cap during the right-view probe.
        control.costs['inspect_right'] = control.costs['inspect_left']
        gc_window.begin()
        control.prime_timing()
        control.start_transport()
        evaluator.window_start_ns = control.now()
        print('Belt started. Wait for the tray at the assess position.', flush=True)
        control.wait_sim(2.)
        control.until(lambda: control.target()['predicted_x_m'] >= settings['assess_trigger_x_m'],
                      timeout=settings['entry_wait_wall_s'], sim_timeout=settings['entry_wait_sim_s'])
        control.begin_operation('initial_assess', capture=True)
        first, grades = capture(control, evaluator, 'assess', out, settings)
        samples += first
        checks += grades
        print('Assess capture complete. Move to inspect_right.', flush=True)
        control.begin_operation('inspect_right', capture=True, returning=True)
        right, grades = capture(control, evaluator, 'inspect_right', out, settings)
        samples += right
        checks += grades
        print('Right capture complete. Measure the return to assess.', flush=True)
        control.begin_operation('return_assess')
        control.move('assess')
        end = control.now()
        evaluator.window_end_ns = end
        control.until(lambda: evaluator.latest_transport_ns is not None and evaluator.latest_transport_ns >= end,
                      timeout=2.)
        control.capture_monitoring = False
        control.stop()
        control.until(lambda: evaluator.last_transport is not None
                      and evaluator.last_transport['command_fresh']
                      and evaluator.last_transport['requested_speed_m_s'] == 0.0
                      and abs(evaluator.last_transport['tray_vx_m_s']) < .0002, timeout=3., check=False)
        require(control.owner.get('owner') == 'HOLD' and not control.owner.get('busy')
                and control.owner['stats']['overlap_violations'] == 0, 'Final exclusive HOLD failed')
        report.update(final_hold_status='PASS', final_belt_stop_status='PASS')
        control.until(lambda: not evaluator.pending, timeout=2., check=False)
        require(control.timing.report()['status'] == 'PASS', 'Overhead camera timing failed')
        report['transport'] = evaluator.finish(control.events)
        report['images'] = control.flush_capture_images()
        report.update(status='PASS', left_reference=left_reference,
                      right_live=dict(costs=costs(control.events, 'inspect_right'),
                                      observations=observations(samples, 'inspect_right',
                                                                int(args.fixture.startswith('green')),
                                                                settings['acceptance_confidence'])),
                      wrist_samples=len(samples), maximum_wrist_error_m=max(x['localization_error_m'] for x in checks))
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='FAIL', error=str(error) or 'Interrupted')
        control.capture_monitoring = False
        try:
            control.stop()
            report['final_hold_status'] = 'PASS'
        except Exception as cleanup_error:
            report['cleanup_error'] = str(cleanup_error)
    finally:
        gc_window.restore('m5b_finished')
        report.update(final_owner=control.owner.get('owner'),
                      overlap_violations=control.owner.get('stats', {}).get('overlap_violations'),
                      overhead_timing=control.timing.report() if control.timing else None)
        write(out / 'events.json', control.events)
        write(out / 'evidence.json', dict(samples=samples, checks=checks))
        write(out / 'summary.json', report)
        executor.remove_node(control)
        executor.remove_node(evaluator)
        control.destroy_node()
        evaluator.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()
    print(json.dumps({k: report[k] for k in ('checkpoint', 'status', 'fixture', 'wrist_samples',
                                            'final_hold_status', 'final_belt_stop_status', 'final_owner',
                                            'overlap_violations', 'right_live', 'output_directory', 'error')
                      if k in report}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4-batch-dir', type=Path, required=True)
    parser.add_argument('--m5a-dir', type=Path, required=True)
    parser.add_argument('--fixture', choices=FIXTURES, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    return run(args, *prerequisites(args.m4_batch_dir.resolve(), args.m5a_dir.resolve(), args.fixture))


if __name__ == '__main__':
    raise SystemExit(main())
