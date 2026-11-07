"""Validate a live pickup proposal. This audit grants no command authority."""
import math
import numpy as np


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def age(now, capture, limit, name):
    value = (now - capture) / 1e9
    require(0 <= value < limit, name + ' is stale or future')
    return value


def check_handoff(decision, target, wrist, owner, feedback, *, now_ns, episode,
                  config, filter_config, model_sha, forecast, remaining):
    """Check current feedback and exact-capture class/position binding before a handoff."""
    c = config
    require(decision['action'] == 'PICK_READY', 'A pass or inspection is not a pickup proposal')
    require(decision['episode_id'] == target['episode_id'] == wrist['episode_id'] == episode,
            'Episode identity differs')
    require(decision['track_id'] == target['track_id'], 'Track identity differs')
    require(decision['frame_id'] == target['frame_id'] == wrist['frame_id'] == 'world', 'Frame differs')
    require(decision['model_sha256'] == model_sha, 'Classifier model differs')
    require(decision['pose'] in ('assess', c['inspection_view']), 'Unvalidated classifier view')
    require(decision['capture_ns'] == wrist['capture_ns'] and wrist['view_id'] == 'wrist'
            and wrist['valid'], 'Exact wrist capture is missing or invalid')
    decision_age = age(now_ns, decision['decision_ns'], c['maximum_capture_age_s'], 'Decision')
    capture_age = age(now_ns, decision['capture_ns'], c['maximum_capture_age_s'], 'Class capture')
    require(decision['capture_ns'] <= decision['decision_ns'], 'Class capture follows its decision')
    probabilities = np.asarray(decision['probabilities'], dtype=float)
    require(probabilities.shape == (len(c['class_ids']),) and np.isfinite(probabilities).all()
            and (probabilities >= 0).all() and (probabilities <= 1).all()
            and abs(float(probabilities.sum()) - 1.) < 1e-6, 'Class probabilities are invalid')
    index = int(np.argmax(probabilities))
    label = c['class_ids'][index]
    confidence = float(probabilities[index])
    require(confidence >= c['acceptance_confidence'] and math.isfinite(decision['confidence'])
            and abs(decision['confidence'] - confidence) < 1e-9, 'Class confidence is invalid')
    require(decision['class_id'] == label and decision['proposed_bin'] == c['bins'][label], 'Class route differs')
    require(owner.get('owner') == 'HOLD' and owner.get('ready') is True and owner.get('busy') is False
            and owner.get('stats', {}).get('overlap_violations') == 0, 'Exclusive ready HOLD is unavailable')
    require(feedback['fresh'] is True and feedback['fingers_open'] is True, 'Fresh open-finger feedback is unavailable')
    require(target['valid'] and target['camera_only'] and target['view_id'] == 'overhead', 'Overhead target is invalid')
    age(now_ns, target['capture_ns'], c['maximum_capture_age_s'], 'Overhead capture')
    # Binding uses an earlier overhead state, never a backward forecast.
    dt = (wrist['capture_ns'] - target['capture_ns']) / 1e9
    require(0 <= dt < c['maximum_capture_age_s'], 'Overhead capture cannot bind the wrist capture')
    state = np.asarray(target['state'], dtype=float)
    covariance = np.asarray(target['covariance'], dtype=float).reshape(4, 4)
    require(state.shape == (4,) and np.isfinite(state).all() and np.isfinite(covariance).all()
            and np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.)
            and np.linalg.eigvalsh(covariance).min() >= -1e-12, 'Overhead state or covariance is invalid')
    position, _ = forecast(state, covariance, dt, filter_config)
    xyz = np.asarray(wrist['position'], dtype=float)
    require(xyz.shape == (3,) and np.isfinite(xyz).all(), 'Wrist position is invalid')
    binding_error = float(np.linalg.norm(position[:2] - xyz[:2]))
    require(binding_error < c['maximum_capture_binding_error_m'], 'Wrist capture belongs to another target')
    lo, hi = c['classification_x_ranges'][decision['pose']]
    require(lo <= xyz[0] <= hi, 'Class capture is outside the validated domain')
    horizon = remaining(target, now_ns, c, filter_config)['remaining_s']
    required = (c['approach_allocation_s'] + c['close_s'] + c['margin_s'])
    # The published estimate also includes return travel for an inspection view.
    estimate = decision['estimated_service_time_s']
    require(math.isfinite(estimate) and estimate >= required, 'Service allocation is invalid')
    require(horizon > estimate and now_ns < decision['deadline_ns'], 'Current pickup slack is insufficient')
    return dict(status='PASS', episode_id=episode, track_id=target['track_id'],
                class_id=label, proposed_bin=c['bins'][label], model_sha256=model_sha,
                capture_ns=decision['capture_ns'], checked_ns=now_ns,
                decision_age_sim_s=decision_age, class_capture_age_sim_s=capture_age,
                capture_binding_error_m=binding_error, remaining_s=horizon,
                allocated_service_s=estimate, slack_s=horizon-estimate,
                physical_pickup_permitted=False, command_authority='NONE',
                scope='Receipt-time handoff audit only; recheck all guards at physical execution')
