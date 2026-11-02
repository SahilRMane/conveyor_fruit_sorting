"""Deadline boundaries, cancellation, clock faults and evaluator isolation."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'src/sorting_supervisor/scripts'
sys.path.insert(0, str(SCRIPTS))
from inspection_core import Costs, InspectionBudget, select_action, time_to_exit


class InspectionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4a.yaml').read_text())['deadline']
        self.cost = Costs(5., .5, .15, .02, 5., 3., .8, .8)
        self.view = dict(id='inspect_left', reachable=True, collision_free=True, fresh_crop_valid=True, costs=self.cost)
        self.args = dict(now=10., capture=9.95, x=.4, vx=.01, sigma_x=0., sigma_v=0.,
                         confidence=.6, views=[self.view], inspection_count=0, held=False, config=self.cfg)

    def decide(self, **changes):
        return select_action(**dict(self.args, **changes))

    def test_uncertain_feasible_inspect(self):
        self.assertEqual(self.decide()['action'], 'INSPECT')

    def test_confident_feasible_pick(self):
        self.assertEqual(self.decide(confidence=.9)['action'], 'PICK')

    def test_stationary_horizon_is_finite(self):
        self.assertEqual(time_to_exit(.4, 0., 0., 0., self.cfg), 25.)

    def test_position_uncertainty_shortens_horizon(self):
        self.assertLess(time_to_exit(.6, .03, .01, 0., self.cfg), time_to_exit(.6, .03, 0., 0., self.cfg))

    def test_velocity_uncertainty_shortens_horizon(self):
        self.assertLess(time_to_exit(.6, .03, 0., .01, self.cfg), time_to_exit(.6, .03, 0., 0., self.cfg))

    def test_recheck_speed_change(self):
        self.assertEqual(self.decide()['action'], 'INSPECT')
        self.assertEqual(self.decide(vx=.09)['action'], 'PASS')

    def test_exit_pass(self):
        self.assertEqual(self.decide(x=.76, confidence=.99)['action'], 'PASS')

    def test_one_extra_view_only(self):
        self.assertEqual(self.decide(inspection_count=1)['action'], 'PASS')

    def test_pick_after_one_good_view(self):
        self.assertEqual(self.decide(inspection_count=1, confidence=.9)['action'], 'PICK')

    def test_stale_capture_pass(self):
        self.assertEqual(self.decide(capture=9.8)['action'], 'PASS')

    def test_future_capture_pass(self):
        self.assertEqual(self.decide(capture=10.001)['action'], 'PASS')

    def test_unreachable_pass(self):
        self.view['reachable'] = False
        self.assertEqual(self.decide()['action'], 'PASS')

    def test_collision_pass(self):
        self.view['collision_free'] = False
        self.assertEqual(self.decide()['action'], 'PASS')

    def test_crop_invalid_pass(self):
        self.view['fresh_crop_valid'] = False
        self.assertEqual(self.decide()['action'], 'PASS')

    def test_return_cost_counted(self):
        self.assertAlmostEqual(self.cost.total, 15.27)
        self.assertAlmostEqual(self.decide()['slack_s'], 25.-15.27)

    def test_strict_zero_slack_reject(self):
        x = self.cfg['exit_x_m'] - self.cost.total * .02
        self.assertEqual(self.decide(x=x, vx=.02)['action'], 'PASS')

    def test_held_payload_has_no_belt_deadline(self):
        self.assertEqual(self.decide(x=2., capture=1., held=True)['action'], 'CONTINUE_PLACE')

    def test_nonfinite_reject(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                self.decide(x=value)

    def test_reverse_belt_reject(self):
        with self.assertRaises(ValueError):
            self.decide(vx=-.01)

    def test_negative_uncertainty_reject(self):
        with self.assertRaises(ValueError):
            self.decide(sigma_x=-1.)

    def test_invalid_cost_reject(self):
        with self.assertRaises(ValueError):
            _ = Costs(-1., 0., 0., 0., 0., 0., 0., 0.).total

    def test_second_inspection_reject(self):
        b = InspectionBudget(2, 10., 1., 20., 30.)
        b.start(2)
        with self.assertRaises(ValueError):
            b.start(2)

    def test_cancel_reject(self):
        b = InspectionBudget(2, 10., 1., 20., 30.); b.cancel()
        with self.assertRaises(ValueError):
            b.check(2, 11., 2.)

    def test_reset_reject(self):
        b = InspectionBudget(2, 10., 1., 20., 30.)
        with self.assertRaises(ValueError):
            b.check(3, 11., 2.)

    def test_frozen_clock_wall_timeout(self):
        b = InspectionBudget(2, 10., 1., 20., 30.)
        with self.assertRaises(ValueError):
            b.check(2, 10., 31.)

    def test_forward_clock_timeout(self):
        b = InspectionBudget(2, 10., 1., 20., 30.)
        with self.assertRaises(ValueError):
            b.check(2, 30., 2.)

    def test_backwards_clock_reject(self):
        b = InspectionBudget(2, 10., 1., 20., 30.)
        with self.assertRaises(ValueError):
            b.check(2, 9., 2.)

    def test_previous_state_capture_reject(self):
        b = InspectionBudget(2, 10., 1., 20., 30.)
        with self.assertRaises(ValueError):
            b.check(2, 10.1, 2., capture=10.)

    def test_fresh_post_move_capture(self):
        b = InspectionBudget(2, 10., 1., 20., 30.)
        b.check(2, 10.1, 2., capture=10.05)

    def test_executor_subscription_isolation(self):
        tree = ast.parse((SCRIPTS/'m4a_runtime.py').read_text())
        subscriptions = [n.args[1].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == 'create_subscription']
        self.assertEqual(set(subscriptions), {'/sorting/episode', '/camera/wrist/image_raw',
                         '/camera/wrist/camera_info', '/joint_states', '/sorting/arm/owner_state'})

    def test_no_pick_executor_in_launch(self):
        tree = ast.parse((ROOT/'src/sorting_supervisor/launch/panda_m4a.launch.py').read_text())
        executables = [k.value.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                       for k in n.keywords if k.arg == 'executable' and isinstance(k.value, ast.Constant)]
        self.assertNotIn('camera_sort.py', executables)
        self.assertNotIn('camera_pickup.py', executables)


if __name__ == '__main__':
    unittest.main()
