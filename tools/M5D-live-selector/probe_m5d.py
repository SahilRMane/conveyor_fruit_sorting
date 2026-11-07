#!/usr/bin/env python3
"""Check a frozen view decision and bounded motion in the visible simulator."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import rclpy
import yaml

from prepare_m5d import CASES, config, load_case, read, require, sha
from m4c_core import DeadlinePass, pick_reserve
from m4c_timing import FullCollectionWindow
from m4d_runtime import IntegratedExecutor
from utility_core import VIEWS, assess_features, choose_view, predict


def model_for(suite):
    path = Path(suite['selector'])
    require(sha(path) == suite['selector_sha256']
            and sha(Path(__file__).with_name('utility_core.py')) == suite['utility_sha256'], 'Frozen selector differs')
    shapes = dict(center=(145,), scale=(145,), bias=(2,), weights=(145, 2))
    with np.load(path, allow_pickle=False) as data:
        model = {key: data[key] for key in shapes}
    require(all(model[k].shape == shape and np.isfinite(model[k]).all() for k, shape in shapes.items())
            and (model['scale'] > 0).all(), 'Invalid selector parameters')
    return model


def image_features(control, sample):
    path, rgb = control.deferred_images[0]
    require(path.name == f"assess-{sample['capture_ns']}.png", 'Assess image stamp differs')
    x0, y0, x1, y1 = sample['crop_xyxy']
    return assess_features(rgb[y0:y1, x0:x1], sample['class_evidence']['confidence'])


def pass_exit(control, evaluator, executor, settings):
    target = control.target()
    lower_x = target['predicted_x_m'] - settings['uncertainty_sigma'] * target['position_sigma_m']
    seconds = max(0., settings['exit_x_m'] + .004 - lower_x) / settings['belt_speed_m_s']
    require(math.isfinite(seconds) and seconds <= 45., 'Late PASS exceeds its exit time bound')
    stats = dict(control.owner['stats'])
    control.operation = None
    control.tracking_active = False
    control.capture_monitoring = False
    control.policy_started = None
    executor.startup_clock_pending = None
    start = control.now()

    def finished():
        require(control.owner['owner'] == 'HOLD' and not control.owner.get('busy'), 'Late PASS lost HOLD')
        return control.now() - start >= round(seconds * 1e9)

    control.until(finished, timeout=65., sim_timeout=seconds + 2.)
    cutoff = control.now()
    control.until(lambda: evaluator.contact is not None
                  and round(evaluator.contact['sim_time'] * 1e9) >= cutoff, timeout=2.)
    require(evaluator.contact['tray_xyz'][0] >= settings['exit_x_m'], 'Tray did not reach the exit')
    require(all(control.owner['stats'][key] == stats[key] for key in
                ('planned_accepted', 'servo_input_accepted', 'servo_output_forwarded')), 'Arm command during late PASS')
    return dict(exit_x_m=settings['exit_x_m'], tray_x_m=evaluator.contact['tray_xyz'][0],
                runout_sim_s=seconds, no_arm_commands=True)


def stop_motion(control, evaluator):
    control.capture_monitoring = False
    control.stop()
    control.until(lambda: evaluator.last_transport is not None and evaluator.last_transport['command_fresh']
                  and evaluator.last_transport['requested_speed_m_s'] == 0.
                  and abs(evaluator.last_transport['tray_vx_m_s']) < .0002, timeout=3., check=False)
    require(control.owner['owner'] == 'HOLD' and not control.owner.get('busy')
            and control.owner['stats']['overlap_violations'] == 0, 'Final exclusive HOLD failed')


def probe(args, suite, case, model):
    helper = Path(suite['m5b_helper'])
    require(sha(helper) == suite['m5b_helper_sha256'] and sha(suite['poses']) == suite['poses_sha256'],
            'M5B motion code or wrist pose file differs')
    sys.path.insert(0, str(helper.parent))
    from probe_m5b import Evaluator, Inspector, capture

    out = args.suite_dir / args.case
    out.mkdir(exist_ok=False)
    folder = Path(suite['preflight']) / 'inspection'
    preflight = read(folder / 'preflight.json')
    preflight['scene_paths']['m5d'] = case['scene']
    joints = {name: item['joints'] for name, item in yaml.safe_load(Path(suite['poses']).read_text())['views'].items()}
    settings = config()
    # shortcut: fixed assess trigger for these covers; remeasure for a new belt speed.
    settings.update(assess_trigger_x_m=.532, entry_wait_sim_s=85.)
    rclpy.init()
    executor = IntegratedExecutor(settings['callback_budget_s'])
    control, evaluator = Inspector(folder, preflight, settings, joints), Evaluator(settings)
    control.c['samples_per_pose'] = 1
    executor.inspector = control
    executor.startup_clock_pending = control.startup_clock_pending
    executor.diagnostic_context = control.timing_context
    executor.add_node(control)
    executor.add_node(evaluator)
    window = FullCollectionWindow()
    report = dict(checkpoint='M5D', status='FAIL', case=args.case, proposal_only=True,
                  full_M5_gate_passed=False, physical_pickup_permitted=False, scene=case['scene'],
                  selector_sha256=suite['selector_sha256'], probe_sha256=sha(__file__),
                  suite_sha256=sha(args.suite_dir / 'suite.json'), final_hold_status='NOT_RUN',
                  final_belt_stop_status='NOT_RUN', output_directory=str(out))
    samples, checks = [], []
    try:
        control.startup()
        require(control.running_scene_path == case['scene'] and control.model.sha == suite['model_sha256'],
                'Terminal 1 scene or classifier differs from the selected case')
        control.until(lambda: evaluator.last_contact_message is not None, timeout=5.)
        evaluator.arm_contact_guard()
        for topic in ('/sorting/ground_truth/camera_sample', '/sorting/ground_truth/conveyor_state',
                      '/sorting/contact_state'):
            subscribers = evaluator.get_subscriptions_info_by_topic(topic)
            require(subscribers and all(x.node_name == evaluator.get_name() for x in subscribers),
                    'Physical truth has a subscriber outside the evaluator')
        window.begin()
        control.prime_timing()
        control.start_transport()
        evaluator.window_start_ns = control.now()
        print('Belt started. Wait for the tray.', flush=True)
        control.wait_sim(2.)
        control.until(lambda: control.target()['predicted_x_m'] >= settings['assess_trigger_x_m'],
                      timeout=settings['entry_wait_wall_s'], sim_timeout=settings['entry_wait_sim_s'])
        reserve = pick_reserve(settings)
        try:
            control.begin_operation('initial_assess', capture=True)
        except DeadlinePass as error:
            report['decision'] = dict(action='PASS', reason=str(error), selected_view=None,
                                      remaining_s=control.target()['remaining_s'])
            require(case['expected_view'] is None, 'Early case lacked time for assess')
            print('PASS. Keep HOLD while the tray reaches the exit.', flush=True)
            report['pass_exit'] = pass_exit(control, evaluator, executor, settings)
        else:
            first, grades = capture(control, evaluator, 'assess', out, settings)
            samples += first
            checks += grades
            initial = first[0]
            require(0 <= control.now() - initial['capture_ns'] < round(settings['maximum_capture_age_s'] * 1e9),
                    'Assess image expired before the decision')
            remaining = control.target()['remaining_s']
            view_costs = {view: suite['view_budgets'][view]['total'] for view in VIEWS}
            scores = predict(model, image_features(control, initial))
            confidence = initial['class_evidence']['confidence']
            if confidence >= settings['acceptance_confidence']:
                view = None
                action = 'PICK_READY' if remaining > reserve else 'PASS'
            else:
                view = choose_view(scores, view_costs, remaining, reserve)
                action = 'INSPECT' if view else 'PASS'
            report['decision'] = dict(action=action, selected_view=view, remaining_s=remaining,
                                      confidence=confidence, capture_ns=initial['capture_ns'],
                                      predicted_payoff={name: float(v) for name, v in zip(VIEWS, scores, strict=True)},
                                      slack_s={name: remaining - cost - reserve for name, cost in view_costs.items()})
            print(json.dumps(report['decision']), flush=True)
            require(view is not None and view == case['expected_view'], 'Live selector chose an unexpected view')
            budget = suite['view_budgets'][view]
            control.costs[view], control.costs['return_assess'] = budget['travel'], budget['return']
            control.begin_operation(view, capture=True, returning=True)
            extra, grades = capture(control, evaluator, view, out, settings)
            samples += extra
            checks += grades
            evidence = extra[0]
            require(0 <= control.now() - evidence['capture_ns'] < round(settings['maximum_capture_age_s'] * 1e9),
                    'Inspection image expired before the proposal')
            class_scores = evidence['class_evidence']['probabilities']
            class_id = settings['class_ids'][int(np.argmax(class_scores))]
            report['proposal'] = dict(action='PICK_READY' if max(class_scores) >= settings['acceptance_confidence'] else 'PASS',
                                      class_id=class_id, confidence=max(class_scores), capture_ns=evidence['capture_ns'],
                                      track_id=control.bound_track, episode_id=control.episode)
            require(report['proposal']['action'] == 'PICK_READY' and class_id == case['expected_class'],
                    'Selected view did not resolve the expected class')
            control.begin_operation('return_assess')
            control.move('assess')
            control.operation = None
            require(control.target()['remaining_s'] > reserve, 'View used the pickup time reserve')
        cutoff = control.now()
        evaluator.window_end_ns = cutoff
        control.until(lambda: evaluator.latest_transport_ns is not None and evaluator.latest_transport_ns >= cutoff,
                      timeout=2.)
        stop_motion(control, evaluator)
        report.update(final_hold_status='PASS', final_belt_stop_status='PASS')
        control.until(lambda: not evaluator.pending, timeout=2., check=False)
        require(control.timing.report()['status'] == 'PASS', 'Overhead timing failed')
        report['transport'] = evaluator.finish(control.events)
        report['images'] = control.flush_capture_images()
        report['status'] = 'PASS'
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = str(error) or 'Interrupted'
        try:
            stop_motion(control, evaluator)
            report.update(final_hold_status='PASS', final_belt_stop_status='PASS')
        except Exception as cleanup:
            report['cleanup_error'] = str(cleanup)
    finally:
        window.restore('m5d_finished')
        report.update(final_owner=control.owner.get('owner'),
                      overlap_violations=control.owner.get('stats', {}).get('overlap_violations'),
                      overhead_timing=control.timing.report() if control.timing else None)
        for name, value in (('summary', report), ('events', control.events),
                            ('evidence', dict(samples=samples, checks=checks))):
            (out / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        executor.remove_node(control)
        executor.remove_node(evaluator)
        control.destroy_node()
        evaluator.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()
    print(json.dumps({k: report[k] for k in ('checkpoint', 'status', 'case', 'decision', 'proposal', 'pass_exit',
                                            'final_hold_status', 'final_belt_stop_status', 'overlap_violations',
                                            'error', 'output_directory') if k in report}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument('--case', choices=list(CASES), required=True)
    args = parser.parse_args()
    suite, case = load_case(args.suite_dir, args.case)
    return probe(args, suite, case, model_for(suite))


if __name__ == '__main__':
    raise SystemExit(main())
