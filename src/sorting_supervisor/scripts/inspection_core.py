"""Conservative deadline checks. No ROS, simulator or class-label inputs."""
from dataclasses import dataclass
import math


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


@dataclass(frozen=True)
class Costs:
    travel: float
    settle: float
    delivery: float
    processing: float
    return_to_grasp: float
    approach: float
    close: float
    margin: float

    @property
    def total(self):
        values = list(vars(self).values())
        require(all(math.isfinite(v) and v >= 0 for v in values), 'invalid_service_cost')
        # Costs are sequential conservative bounds, with no hidden overlap credit.
        return sum(values)


def time_to_exit(x, vx, sigma_x, sigma_v, config):
    values = (x, vx, sigma_x, sigma_v)
    require(all(math.isfinite(v) for v in values), 'nonfinite_target')
    require(sigma_x >= 0 and sigma_v >= 0, 'negative_uncertainty')
    require(vx >= 0, 'reverse_belt_outside_checkpoint')
    k = config['uncertainty_sigma']
    remaining = config['exit_x_m'] - x - k * sigma_x
    if remaining <= 0:
        return 0.0
    fast_v = vx + k * sigma_v
    if fast_v <= config['near_zero_speed_m_s']:
        return float(config['maximum_horizon_s'])
    return min(config['maximum_horizon_s'], remaining / fast_v)


def select_action(*, now, capture, x, vx, sigma_x, sigma_v, confidence,
                  views, inspection_count, held, config):
    """Pure policy proposal only. A returned PICK never commands the robot."""
    require(all(math.isfinite(v) for v in (now, capture, confidence)), 'invalid_evidence')
    require(0 <= confidence <= 1 and inspection_count in (0, 1), 'invalid_policy_state')
    if held:
        return dict(action='CONTINUE_PLACE', reason='belt_exit_does_not_bound_held_payload')
    if not 0 <= now-capture <= .15:
        return dict(action='PASS', reason='stale_or_future_capture')
    horizon = time_to_exit(x, vx, sigma_x, sigma_v, config)
    pick_cost = config['approach_s'] + config['close_s'] + config['margin_s']
    if confidence >= config['acceptance_confidence'] and horizon > pick_cost:
        return dict(action='PICK', remaining_s=horizon, slack_s=horizon-pick_cost)
    if inspection_count:
        return dict(action='PASS', reason='one_extra_view_used', remaining_s=horizon)
    feasible = []
    for view in views:
        # Reachability and collision checks are supplied by the executor, never assumed.
        if not view['reachable'] or not view['collision_free'] or not view['fresh_crop_valid']:
            continue
        cost = view['costs'].total
        if horizon > cost:
            feasible.append((cost, view['id']))
    if feasible:
        cost, name = min(feasible)
        return dict(action='INSPECT', selected_view=name, remaining_s=horizon, slack_s=horizon-cost)
    return dict(action='PASS', reason='no_feasible_view_or_pick', remaining_s=horizon)


class InspectionBudget:
    """Bound one inspection and reject evidence from a previous state or reset."""
    def __init__(self, episode, now, wall_now, sim_bound, wall_bound):
        require(all(math.isfinite(v) and v > 0 for v in (sim_bound, wall_bound)), 'invalid_state_bound')
        self.episode, self.begin, self.wall_begin = episode, now, wall_now
        self.sim_bound, self.wall_bound = sim_bound, wall_bound
        self.used, self.cancelled = False, False

    def start(self, episode):
        require(not self.cancelled and not self.used and episode == self.episode, 'inspection_unavailable')
        self.used = True

    def check(self, episode, now, wall_now, capture=None):
        require(not self.cancelled and episode == self.episode, 'inspection_cancel_or_reset')
        require(0 <= now-self.begin < self.sim_bound, 'inspection_sim_timeout_or_clock_reset')
        require(0 <= wall_now-self.wall_begin < self.wall_bound, 'inspection_wall_timeout')
        if capture is not None:
            require(self.begin < capture <= now and now-capture <= .15, 'inspection_capture_invalid')

    def cancel(self):
        self.cancelled = True
