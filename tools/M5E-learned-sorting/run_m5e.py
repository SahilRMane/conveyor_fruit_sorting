#!/usr/bin/env python3
"""Run one guarded physical sort from a fresh learned inspection decision."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.parameter import Parameter

M5D = Path(__file__).resolve().parent.parent / 'M5D-live-selector'
sys.path.insert(0, str(M5D))
from prepare_m5d import config as moving_config, load_case, read, require, sha
from probe_m5d import image_features, model_for
from utility_core import VIEWS, choose_view, predict
from carried_checks import planning_configuration
from filter_core import forecast
from m3d_components import runtime_configuration
from m4c_core import bind_capture, pick_reserve, remaining
from m4c_timing import FullCollectionWindow
from m4d_common import config as integration_config
from m4d_core import RouteLease
from m4d_evaluator import IntegratedEvaluator
from m4d_handoff import check_handoff
from m4d_pickup import InspectedSortController
from m4d_runtime import IntegratedExecutor, wrist_timing_report


def check_proposals(folder, suite):
    suite_hash = sha(folder / 'suite.json')
    for name in ('left', 'right', 'late'):
        report = read(folder / name / 'summary.json')
        require(report['checkpoint'] == 'M5D' and report['case'] == name
                and report['suite_sha256'] == suite_hash and report['selector_sha256'] == suite['selector_sha256']
                and report['probe_sha256'] == sha(M5D / 'probe_m5d.py')
                and all(report[k] == 'PASS' for k in ('status', 'final_hold_status', 'final_belt_stop_status'))
                and report['final_owner'] == 'HOLD' and report['overlap_violations'] == 0, 'M5D report differs: ' + name)
        case = suite['cases'][name]
        if name == 'late':
            passed = report['pass_exit']
            require(report['decision']['action'] == 'PASS' and passed['no_arm_commands']
                    and passed['tray_x_m'] >= passed['exit_x_m'], 'M5D late exit did not pass')
        else:
            require(report['decision']['selected_view'] == case['expected_view']
                    and report['proposal']['action'] == 'PICK_READY'
                    and report['proposal']['class_id'] == case['expected_class'], 'M5D proposal differs: ' + name)


def panda_model(scene):
    tree = ET.parse(scene)
    # The right scene includes the original Panda model from another directory.
    source, = [scene.parent / node.attrib['file'] for node in tree.findall('include')
               if Path(node.attrib['file']).name == 'panda.xml']
    return source


def reset_reference(scene):
    key = ET.parse(panda_model(scene)).find(".//key[@name='home']")
    return {name: [float(v) for v in key.get(name, ' '.join(['0'] * 15)).split()]
            for name in ('qpos', 'qvel', 'ctrl')}


def pose_targets(suite, scene):
    bias = read(Path(__file__).with_name('pose-bias.json'))
    require(sha(suite['poses']) == suite['poses_sha256'] == bias['poses_sha256']
            and sha(panda_model(scene)) == bias['panda_sha256'], 'Inspection pose or Panda model differs')
    poses = yaml.safe_load(Path(suite['poses']).read_text())['views']
    result = {}
    for name in ('assess', *VIEWS):
        offsets = np.asarray(bias['offsets'][name])
        require(offsets.shape == (7,) and np.isfinite(offsets).all() and np.max(np.abs(offsets)) <= .02,
                'Inspection pose offset exceeds its bound')
        result[name] = (np.asarray(poses[name]['joints']) + offsets).tolist()
    return result


def capture(control, pose, out):
    control.move(pose)
    sample, = control.capture(pose, out)
    sample['class_evidence'] = sample.pop('classifier_diagnostic_only')
    sample['track_binding'] = bind_capture(sample, list(control.history), control.episode, control.bound_track,
                                          control.mc, control.filter_config)
    now = control.now()
    require(0 <= now - sample['capture_ns'] < round(control.mc['maximum_capture_age_s'] * 1e9),
            'Wrist image expired before the decision')
    domain = 'assess' if pose == 'assess' else 'inspect_left'
    lo, hi = control.mc['classification_x_ranges'][domain]
    require(lo <= sample['handle_xyz'][0] <= hi, 'Wrist image is outside the test position interval')
    return sample


def proposal_from_capture(sample, view, control, now, horizon, service):
    probabilities = sample['class_evidence']['probabilities']
    index = int(np.argmax(probabilities))
    label = control.mc['class_ids'][index]
    return dict(action='PICK_READY', episode_id=control.episode, track_id=control.bound_track,
                pose=view, frame_id='world', model_sha256=control.model.sha,
                capture_ns=sample['capture_ns'], decision_ns=now,
                decision_time_ns=now, evidence_capture_ns=sample['capture_ns'],
                deadline_ns=now + round(horizon * 1e9), remaining_s=horizon,
                estimated_service_time_s=service, probabilities=probabilities,
                class_id=label, proposed_bin=control.mc['bins'][label], confidence=probabilities[index],
                reason='fresh_learned_view_capture')


def inspect(control, out, model, suite, limits, report):
    mc = control.mc
    control.prime_timing()
    control.start_transport()
    report['transport_start_ns'] = control.now()
    print('Belt started. Wait for the tray to reach assess.', flush=True)
    control.wait_sim(2.)
    control.until(lambda: control.target()['predicted_x_m'] >= mc['assess_trigger_x_m'],
                  timeout=mc['entry_wait_wall_s'], sim_timeout=mc['entry_wait_sim_s'])
    control.begin_operation('initial_assess', capture=True)
    initial = capture(control, 'assess', out)
    report['samples'] = [initial]
    require(initial['class_evidence']['confidence'] < mc['acceptance_confidence'],
            'Assess already resolved the class. This pilot requires a learned view.')
    scores = predict(model, image_features(control, initial))
    horizon = control.target()['remaining_s']
    costs = {view: suite['view_budgets'][view]['total'] for view in VIEWS}
    reserve = pick_reserve(mc)
    view = choose_view(scores, costs, horizon, reserve)
    report['selection'] = dict(view=view, remaining_s=horizon, assess_capture_ns=initial['capture_ns'],
                               predicted_payoff=dict(zip(VIEWS, map(float, scores), strict=True)))
    print(json.dumps(report['selection']), flush=True)
    require(view is not None, 'No useful view fits the current deadline')
    budget = suite['view_budgets'][view]
    control.costs[view], control.costs['return_assess'] = budget['travel'], budget['return']
    control.begin_operation(view, capture=True, returning=True)
    sample = capture(control, view, out)
    report['samples'].append(sample)
    now = control.now()
    horizon = control.target()['remaining_s']
    decision = proposal_from_capture(sample, view, control, now, horizon, budget['return'] + reserve)
    target = max((r for r in control.history if r['capture_ns'] <= sample['capture_ns']),
                 key=lambda r: r['capture_ns'])
    wrist = dict(episode_id=sample['episode_id'], capture_ns=sample['capture_ns'], frame_id='world',
                 view_id='wrist', valid=True, position=sample['handle_xyz'])
    # Admit only the selected, preflight-bound pose and the existing moving image interval.
    admission = dict(mc, inspection_view=view, classification_x_ranges={
        **mc['classification_x_ranges'], view: mc['classification_x_ranges']['inspect_left']})
    report['handoff_audit'] = check_handoff(decision, target, wrist, control.owner,
        dict(fresh=control.feedback_snapshot()['fresh'],
             fingers_open=all(control.q[n] >= .039 for n in ('panda_finger_joint1', 'panda_finger_joint2'))),
        now_ns=now, episode=control.episode, config=admission, filter_config=control.filter_config,
        model_sha=control.model.sha, forecast=forecast, remaining=remaining)
    control.route_lease = RouteLease(decision, control.model.sha, limits['maximum_route_age_sim_s'])
    report['proposal'] = decision
    control.begin_operation('return_assess')
    control.move('assess')
    control.operation = None
    control.route_lease.check(control.overhead, control.now(), mc)
    require(control.target()['remaining_s'] > reserve, 'View used the pickup time reserve')


def run(args, suite, case, model):
    helper = Path(suite['m5b_helper'])
    require(sha(helper) == suite['m5b_helper_sha256'] and sha(suite['poses']) == suite['poses_sha256'],
            'M5B helper or wrist pose file differs')
    sys.path.insert(0, str(helper.parent))
    from probe_m5b import Inspector

    scene = Path(case['scene'])
    expected_reset = reset_reference(scene)
    folder = Path(suite['preflight']) / 'inspection'
    preflight = read(folder / 'preflight.json')
    preflight['scene_paths']['m5e'] = str(scene)
    joints = pose_targets(suite, scene)
    mc, limits = moving_config(), integration_config()
    mc.update(assess_trigger_x_m=.532, entry_wait_sim_s=85.)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    executor = IntegratedExecutor(mc['callback_budget_s'])
    control = Inspector(folder, preflight, mc, joints)
    control.c['samples_per_pose'] = 1
    executor.inspector = control
    executor.startup_clock_pending = control.startup_clock_pending
    executor.diagnostic_context = control.timing_context
    evaluator_node = Node('m4d_evaluator', parameter_overrides=[Parameter('use_sim_time', value=True)],
        cli_args=['--ros-args', '-r', '/sorting/targets:=/sorting/m4c/targets',
                  '-r', '/sorting/perception/filter_status:=/sorting/m4c/filter_status'])
    evaluator = IntegratedEvaluator(evaluator_node, executor)
    executor.add_node(control)
    executor.add_node(evaluator_node)
    window = FullCollectionWindow()
    controller = node = None
    detached = False
    report = dict(checkpoint='M5E', case=args.case, status='FAIL', full_M5_gate_passed=False,
                  selector_sha256=suite['selector_sha256'], suite_sha256=sha(args.suite_dir / 'suite.json'),
                  source_sha256=sha(__file__), output_directory=str(out),
                  pose_bias_sha256=sha(Path(__file__).with_name('pose-bias.json')),
                  physical_grasp='NOT_RUN', placement_status='NOT_RUN', final_reset_status='NOT_RUN')
    try:
        control.startup()
        require(control.running_scene_path == str(scene) and control.model.sha == suite['model_sha256'],
                'Terminal 1 scene or classifier differs')
        node = Node('m4d_sort_controller', parameter_overrides=[Parameter('use_sim_time', value=True)],
                    cli_args=['--ros-args', '-r', '/sorting/targets:=/sorting/m4c/targets'])
        controller = InspectedSortController(node, executor, control, None, mc, control.filter_config, limits,
                                             time.monotonic() + limits['maximum_episode_wall_s'], None)
        executor.add_node(node)
        executor.physical = controller
        controller.until(lambda: controller.epoch is not None and controller.belief is not None
                         and len(controller.q) >= 9 and controller.owner.get('owner') == 'HOLD'
                         and controller.scene.get('valid')
                         and controller.buffer.can_transform('world', 'panda_hand', rclpy.time.Time()), 5.)
        control.until(lambda: bool(evaluator.contact), timeout=5.)
        evaluator.plan['initial_state'] = deepcopy(evaluator.contact)
        report['runtime_configuration'] = runtime_configuration(evaluator)
        report['planning_configuration'] = planning_configuration(evaluator)
        report['isolation'] = [evaluator.isolation(n.get_name()) for n in (control, node)]
        controller.wrist.calibrate()
        window.begin()
        inspect(control, out, model, suite, limits, report)
        report['wrist_checks'] = evaluator.wrist_checks(report['samples'])
        require(control.timing.failure is None and control.owner.get('owner') == 'HOLD'
                and not control.owner.get('busy') and not control.refinement_active
                and control.active_goal is None, 'Inspection is not ready for an exclusive handoff')
        handoff = time.monotonic()
        control.handed_off = True
        control.tracking_active = False
        executor.startup_clock_pending = None
        controller.enable_pickup(control.route_lease, controller.wrist.tare)
        require(time.monotonic() - handoff < limits['maximum_handoff_wall_s'], 'Handoff exceeded its bound')
        route = controller.current_route()
        print('Physical controller accepted the camera route to ' + route['bin_id'] + '.', flush=True)
        controller.pending = SimpleNamespace(episode_id=controller.epoch.episode_id,
                                             track_id=route['track_id'], trial_id=1)
        physical = out / 'physical'
        physical.mkdir()
        controller.run_pending(physical)
        report['controller'] = controller.result
        require(controller.result.get('status') == 'PASS', controller.result.get('error', 'Physical controller failed'))
        evaluator.until(lambda: evaluator.control_result is not None and not evaluator.control_state.get('active'), 5.)
        report['physical'] = evaluator.verify_sort(controller.result, case['expected_class'])
        events = controller.result['events']
        report['continuous_transport'] = evaluator.transport_checks(report['transport_start_ns'],
            next(e['time_ns'] for e in events if e['phase'] == 'hold'), mc['belt_speed_m_s'],
            next(e['time_ns'] for e in events if e['phase'] == 'close'))
        require(control.pickup_timing is not None and control.pickup_timing.report()['status'] == 'PASS',
                'Wrist tracking timing failed')
        require(report['selection']['view'] == case['expected_view'], 'Evaluator: unexpected learned view')
        report.update(physical_grasp='PASS', placement_status='PASS')
        control.capture_monitoring = False
        control.stop()
        controller.input_enabled = False
        controller.belt_speed = 0.
        controller.send_belt()
        executor.remove_node(control)
        executor.remove_node(node)
        executor.inspector = executor.physical = None
        detached = True
        evaluator.reset_manager.expected = expected_reset
        report['reset'] = evaluator.reset_episode()
        require(evaluator.owner.get('owner') == 'HOLD' and not evaluator.owner.get('busy')
                and evaluator.owner['stats']['overlap_violations'] == 0, 'Final exclusive HOLD failed')
        report.update(status='PASS', final_reset_status='PASS')
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = str(error) or 'Interrupted'
        if controller is not None:
            controller.input_enabled = controller.target_required = False
            controller.cleanup = True
            controller.belt_speed = 0.
            controller.send_belt()
        try:
            control.stop()
        except Exception as stop_error:
            report['stop_error'] = str(stop_error)
        report['failure_handling'] = evaluator.stop_after_failure()
    finally:
        control.capture_monitoring = False
        window.restore('m5e_finished')
        try:
            report['image_evidence'] = control.flush_capture_images()
        except Exception as error:
            report.update(status='FAIL', image_error=str(error))
        report.update(final_owner=evaluator.owner.get('owner'),
                      overlap_violations=evaluator.owner.get('stats', {}).get('overlap_violations'),
                      overhead_timing=control.timing.report() if control.timing else None,
                      wrist_timing=wrist_timing_report(control))
        for name, value in (('summary', report), ('inspection-events', control.events),
                            ('physical-trace', evaluator.history), ('transport', evaluator.transport)):
            (out / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        if not detached:
            executor.remove_node(control)
            if node is not None:
                executor.remove_node(node)
        if node is not None:
            node.destroy_node()
        executor.remove_node(evaluator_node)
        control.destroy_node()
        evaluator_node.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()
    keys = ('checkpoint', 'case', 'status', 'physical_grasp', 'placement_status', 'final_reset_status',
            'final_owner', 'overlap_violations', 'output_directory', 'error')
    print(json.dumps({k: report[k] for k in keys if k in report}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument('--case', choices=('left', 'right'), required=True)
    parser.add_argument('--output-dir', type=lambda x: Path(x).resolve())
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    suite, case = load_case(args.suite_dir, args.case)
    check_proposals(args.suite_dir, suite)
    model = model_for(suite)
    reset_reference(Path(case['scene']))
    pose_targets(suite, Path(case['scene']))
    if args.check:
        print(json.dumps(dict(checkpoint='M5E-precheck', status='PASS', case=args.case, robot_commanded=False)))
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required for a physical run')
    return run(args, suite, case, model)


if __name__ == '__main__':
    raise SystemExit(main())
