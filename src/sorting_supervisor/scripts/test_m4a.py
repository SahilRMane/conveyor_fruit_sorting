#!/usr/bin/env python3
"""Live M4A inspection sweep. Evaluator truth stays in a separate node."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from sorting_interfaces.msg import CameraTruth
from m4a_common import checked_preflight, config, manifest
from m4a_runtime import InspectionExecutor, ns
from classifier_core import require
from camera_core import quaternion_rotation


class Evaluator(Node):
    """Test assertions only. No truth is passed to the inspection executor."""
    def __init__(self):
        super().__init__('m4a_evaluator', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.truth, self.contact = {}, None
        self.failure, self.reference = None, None
        self.create_subscription(CameraTruth, '/sorting/ground_truth/camera_sample', self.receive_truth, 200)
        self.create_subscription(String, '/sorting/contact_state', self.receive_contact, 10)

    def receive_truth(self, m):
        if m.view_id == 'wrist':
            self.truth[(m.episode_id, ns(m.header.stamp))] = m
            while len(self.truth) > 500:
                self.truth.pop(next(iter(self.truth)))

    def receive_contact(self, m):
        self.contact = json.loads(m.data)
        c = self.contact
        if self.reference is None:
            self.reference = c['tray_xyz']
        if c['forbidden_contacts'] or c['left_force_N']+c['right_force_N'] >= .1:
            self.failure = 'Inspection made physical contact'
        if min(c['finger_positions_m']) < .039 or c['support_force_N'] < .5:
            self.failure = 'Open-finger or supported-tray check failed'
        if np.linalg.norm(np.asarray(c['tray_xyz'])-self.reference) >= .001:
            self.failure = 'Tray moved during stationary inspection'
        if abs(c.get('external_force_x_N', 0.)) > 1e-9:
            self.failure = 'Conveyor force must be zero for this development fixture'

    def assess(self, samples, tolerance):
        result = []
        for s in samples:
            truth = next((self.truth[(s['episode_id'], s['capture_ns']+delta)]
                          for delta in (0, -1, 1, -2, 2)
                          if (s['episode_id'], s['capture_ns']+delta) in self.truth), None)
            require(truth is not None, 'Evaluator capture-time pair missing')
            p, q = truth.camera_pose.position, truth.camera_pose.orientation
            camera_error = float(np.linalg.norm(np.asarray(s['camera_xyz'])-[p.x, p.y, p.z]))
            angle = float(np.arccos(np.clip((np.trace(quaternion_rotation(s['camera_quaternion_xyzw']) @
                                                     quaternion_rotation([q.x, q.y, q.z, q.w]).T)-1)/2, -1, 1)))
            h = truth.handle_position
            local_error = float(np.linalg.norm(np.asarray(s['handle_xyz'][:2])-[h.x, h.y]))
            require(camera_error < .001 and angle < .005, 'Capture-time camera TF differs from physical camera')
            require(local_error < tolerance, 'Image localization exceeds 2 mm')
            result.append(dict(capture_ns=s['capture_ns'], camera_error_m=camera_error,
                               camera_error_rad=angle, localization_error_m=local_error))
        require(self.failure is None, self.failure)
        return result

    def have_pairs(self, samples):
        return all(any((s['episode_id'], s['capture_ns']+delta) in self.truth for delta in (0, -1, 1, -2, 2))
                   for s in samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    folder, _ = checked_preflight(args.preflight_dir)
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        parser.error('Use a new output directory to preserve the previous evidence')
    out.mkdir(parents=True)
    report = dict(checkpoint='M4A', status='FAIL', full_M4_gate_passed=False, moving_conveyor=False,
                  physical_grasp='NOT_RUN', classifier_new_views_validated=False, policy_decisions='NOT_EXECUTED',
                  preflight_directory=str(folder), output_directory=str(out), final_hold_status='NOT_RUN')
    rclpy.init()
    executor = SingleThreadedExecutor()
    control, evaluator = InspectionExecutor(folder), Evaluator()
    executor.add_node(control); executor.add_node(evaluator)
    samples, checks = [], []
    begin = time.monotonic()
    try:
        control.startup()
        control.until(lambda: evaluator.contact is not None, timeout=5.)
        require(evaluator.failure is None, evaluator.failure)
        # Test both candidates. The eventual policy may select only one extra view.
        for name in ('assess', 'inspect_left', 'assess', 'inspect_right', 'assess'):
            control.move(name)
            captured = control.capture(name, out)
            # Wait for the exact evaluator pairs; no fixed transport-delay assumption.
            control.until(lambda: evaluator.have_pairs(captured), timeout=2.)
            checks.extend(evaluator.assess(captured, config()['maximum_localization_error_m']))
            samples.extend(captured)
            print(f'{name}: PASS ({len(captured)} fresh frames)', flush=True)
        control.stop()
        require(control.owner.get('stats', {}).get('overlap_violations') == 0, 'Command overlap')
        require(control.episode == control.initial_episode, 'Episode changed during sweep')
        report.update(status='PASS', acceptance_passed=True, final_hold_status='PASS',
                      return_to_assess='PASS', checked_samples=len(samples),
                      maximum_capture_age_sim_s=max(s['age_sim_s'] for s in samples),
                      maximum_localization_error_m=max(c['localization_error_m'] for c in checks))
    except (Exception, KeyboardInterrupt) as error:
        report.update(error=str(error) or 'Interrupted', acceptance_passed=False)
        try:
            control.stop()
            report['final_hold_status'] = 'PASS'
        except Exception as cleanup:
            report['cleanup_error'] = str(cleanup)
    finally:
        report.update(elapsed_wall_s=time.monotonic()-begin, final_owner=control.owner.get('owner'),
                      overlap_violations=control.owner.get('stats', {}).get('overlap_violations'),
                      capture_rejections=dict(control.rejections), installed_manifest=manifest())
        report['last_state'] = control.events[-1] if control.events else None
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        (out/'samples.json').write_text(json.dumps(dict(samples=samples, checks=checks), indent=2)+'\n')
        (out/'events.json').write_text(json.dumps(control.events, indent=2)+'\n')
        executor.remove_node(control); executor.remove_node(evaluator)
        control.destroy_node(); evaluator.destroy_node(); executor.shutdown(); rclpy.try_shutdown()
    print(json.dumps({k: v for k, v in report.items() if k != 'installed_manifest'}, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
