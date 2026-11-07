"""Batch inspection sequence based on the installed M4D pilot."""
from classifier_core import require
from filter_core import forecast
from m4c_core import DeadlinePass, MovingPolicy, bind_capture, remaining
from m4d_core import RouteLease
from m4d_handoff import check_handoff


def inspect(control, output, integration):
    c = control.mc
    control.prime_timing()
    control.start_transport()
    start_ns = control.now()
    samples = []
    policy = None
    handoff = None
    try:
        control.wait_sim(2.)
        control.until(lambda: control.target()['predicted_x_m'] >= c['assess_trigger_x_m'],
                      timeout=c['entry_wait_wall_s'], sim_timeout=c['entry_wait_sim_s'])
        policy = MovingPolicy(control.episode, control.bound_track, control.now(), c, control.costs)
        for pose in ('assess', 'inspect_left'):
            if pose == 'inspect_left':
                policy.start_inspection(control.now())
            name = 'initial_assess' if pose == 'assess' else 'inspect_left'
            control.begin_operation(name, capture=True, returning=pose == 'inspect_left')
            control.move(pose)
            captured = control.capture(pose, output)
            for sample in captured:
                sample['class_evidence'] = sample.pop('classifier_diagnostic_only')
                sample['track_binding'] = bind_capture(sample, list(control.history), control.episode,
                    control.bound_track, c, control.filter_config)
            samples += captured
            evidence = captured[-1]
            lo, hi = c['classification_x_ranges'][pose]
            require(lo <= evidence['handle_xyz'][0] <= hi, 'Class capture is outside the tested moving domain')
            control.operation = None
            decision = policy.observe(track_id=control.bound_track, horizon=control.target()['remaining_s'],
                episode=control.episode, pose=pose, capture_ns=evidence['capture_ns'], now_ns=control.now(),
                probabilities=evidence['class_evidence']['probabilities'])
            decision['decision_time_ns'] = control.now()
            if decision['action'] != 'PASS':
                control.publish_decision(decision)
            if decision['action'] == 'PICK_READY':
                target = max((b for b in control.history if b['capture_ns'] <= evidence['capture_ns']),
                             key=lambda b: b['capture_ns'])
                d = dict(decision, frame_id='world', model_sha256=control.model.sha,
                    capture_ns=decision['evidence_capture_ns'], decision_ns=decision['decision_time_ns'],
                    deadline_ns=control.now()+round(decision['remaining_s']*1e9))
                wrist = dict(episode_id=evidence['episode_id'], capture_ns=evidence['capture_ns'],
                             frame_id='world', view_id='wrist', valid=True, position=evidence['handle_xyz'])
                handoff = check_handoff(d, target, wrist, control.owner,
                    dict(fresh=control.feedback_snapshot()['fresh'], fingers_open=all(control.q[n]>=.039
                         for n in ('panda_finger_joint1','panda_finger_joint2'))),
                    now_ns=control.now(), episode=control.episode, config=c,
                    filter_config=control.filter_config, model_sha=control.model.sha,
                    forecast=forecast, remaining=remaining)
                control.route_lease = RouteLease(decision, control.model.sha, integration['maximum_route_age_sim_s'])
            if decision['action'] != 'INSPECT':
                break
        if pose == 'inspect_left' and decision['action'] == 'PICK_READY':
            control.begin_operation('return_assess')
            control.move('assess')
            control.operation = None
    except DeadlinePass as error:
        decision = dict(action='PASS', reason=str(error), episode_id=control.episode,
                        track_id=control.bound_track, decision_time_ns=control.now())
        control.route_lease = None
    if control.route_lease is not None:
        control.route_lease.check(control.overhead, control.now(), c)
        require(control.target()['remaining_s'] > c['approach_allocation_s']+c['close_s']+c['margin_s'],
                'Pickup slack disappeared during the return from inspection')
    return dict(decision=decision, samples=samples, policy=policy.history if policy else [],
                inspections=policy.inspections if policy else 0, handoff=handoff, transport_start_ns=start_ns)
