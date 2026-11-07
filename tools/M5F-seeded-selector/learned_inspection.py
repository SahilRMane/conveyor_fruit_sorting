"""Select one wrist view and admit a route only from fresh class evidence."""


from run_m5e import (
    VIEWS, capture, check_handoff, choose_view, forecast,
    image_features, pick_reserve, predict, proposal_from_capture, remaining, require, RouteLease,
)
from m4c_core import DeadlinePass


def assess(probabilities, threshold, horizon, reserve, costs, payoff):
    if max(probabilities) >= threshold:
        return ('PICK_READY', None, 'fresh_assess_evidence') if horizon > reserve else (
            'PASS', None, 'insufficient_pickup_slack')
    view = choose_view(payoff, costs, horizon, reserve)
    if view is not None:
        return 'INSPECT', view, 'positive_view_utility'
    if any(horizon > cost + reserve for cost in costs.values()):
        return 'PASS', None, 'no_useful_inspection_view'
    return 'PASS', None, 'insufficient_inspection_slack'


def inspect(control, output, limits, suite, model):
    mc = control.mc
    reserve = pick_reserve(mc)
    costs = {view: suite['view_budgets'][view]['total'] for view in VIEWS}
    control.prime_timing()
    control.start_transport()
    start_ns = control.now()
    samples = []
    selection = None
    handoff = None
    view = 'assess'

    try:
        control.wait_sim(2.)
        control.until(lambda: control.target()['predicted_x_m'] >= mc['assess_trigger_x_m'],
                      timeout=mc['entry_wait_wall_s'], sim_timeout=mc['entry_wait_sim_s'])
        control.begin_operation('initial_assess', capture=True)
        evidence = capture(control, view, output)
        samples.append(evidence)
        horizon = control.target()['remaining_s']
        probabilities = evidence['class_evidence']['probabilities']
        scores = (predict(model, image_features(control, evidence))
                  if max(probabilities) < mc['acceptance_confidence'] else None)
        action, selected_view, reason = assess(
            probabilities, mc['acceptance_confidence'], horizon, reserve, costs, scores)
        selection = dict(action=action, selected_view=selected_view, reason=reason,
                         capture_ns=evidence['capture_ns'], remaining_s=horizon,
                         predicted_payoff=dict(zip(VIEWS, map(float, scores), strict=True)) if scores is not None else None)
        control.operation = None

        if action == 'INSPECT':
            view = selected_view
            control.publish_decision(dict(selection, pose='assess', evidence_capture_ns=evidence['capture_ns'],
                                          estimated_service_time_s=costs[view] + reserve, probabilities=probabilities))
            budget = suite['view_budgets'][view]
            control.costs[view], control.costs['return_assess'] = budget['travel'], budget['return']
            control.begin_operation(view, capture=True, returning=True)
            evidence = capture(control, view, output)
            samples.append(evidence)
            horizon = control.target()['remaining_s']
            if max(evidence['class_evidence']['probabilities']) < mc['acceptance_confidence']:
                action, reason = 'PASS', 'uncertain_after_one_extra_view'
            elif horizon <= budget['return'] + reserve:
                raise DeadlinePass('insufficient_slack_before_return_assess')
            else:
                action, reason = 'PICK_READY', 'fresh_learned_view_capture'
            control.operation = None

        if action == 'PICK_READY':
            now = control.now()
            service = reserve if view == 'assess' else control.costs['return_assess'] + reserve
            decision = proposal_from_capture(evidence, view, control, now, horizon, service)
            decision['reason'] = reason
            capture_ns = evidence['capture_ns']
            target = next(b for b in control.history
                          if b['capture_ns'] == evidence['track_binding']['overhead_capture_ns'])
            wrist = dict(episode_id=evidence['episode_id'], capture_ns=capture_ns, frame_id='world',
                         view_id='wrist', valid=True, position=evidence['handle_xyz'])
            admission = dict(mc, inspection_view=view, classification_x_ranges={
                **mc['classification_x_ranges'], view: mc['classification_x_ranges'][
                    'assess' if view == 'assess' else 'inspect_left']})
            handoff = check_handoff(decision, target, wrist, control.owner,
                dict(fresh=control.feedback_snapshot()['fresh'],
                     fingers_open=all(control.q[n] >= .039 for n in ('panda_finger_joint1', 'panda_finger_joint2'))),
                now_ns=now, episode=control.episode, config=admission, filter_config=control.filter_config,
                model_sha=control.model.sha, forecast=forecast, remaining=remaining)
            control.route_lease = RouteLease(decision, control.model.sha, limits['maximum_route_age_sim_s'])
            if view != 'assess':
                control.begin_operation('return_assess')
                control.move('assess')
                control.operation = None
            control.route_lease.check(control.overhead, control.now(), mc)
            if control.target()['remaining_s'] <= reserve:
                raise DeadlinePass('insufficient_slack_at_handoff')
            control.publish_decision(decision)
            path = 'direct' if view == 'assess' else 'inspect'
        else:
            decision = dict(action='PASS', reason=reason, pose=view, evidence_capture_ns=evidence['capture_ns'])
            path = 'unresolved' if reason in ('no_useful_inspection_view', 'uncertain_after_one_extra_view') else 'deadline'
    except DeadlinePass as error:
        decision = dict(action='PASS', reason=str(error))
        path = 'deadline'
        control.route_lease = None
        control.operation = None

    return dict(decision=decision, path=path, samples=samples, selection=selection,
                handoff=handoff, transport_start_ns=start_ns)
