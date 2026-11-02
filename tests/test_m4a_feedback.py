"""Replay clock/joint arrival order through the real wait loop. No ROS nodes."""
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
from rcl_interfaces.msg import ParameterType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/sorting_supervisor/scripts'))
from m4a_runtime import InspectionExecutor, stamp
from ownership_core import NAMES


class ReplayInspectionExecutor(InspectionExecutor):
    @property
    def executor(self): return self.replay_executor


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.wall = 100.
        self.tick = 0
        self.hook = self.normal
        self.node = object.__new__(ReplayInspectionExecutor)
        n = self.node
        n.q, n.v, n.owner = {}, {}, dict(owner='HOLD', busy=False)
        n.q_received, n.q_stamp, n.q_sequence, n.q_ignored = 0., 0, 0, 0
        n.clock_last, n.sim = 0, 10_000_000_000
        n.feedback_armed, n.feedback_diagnostic = False, {}
        n.failure, n.events = None, []
        n.c = {'maximum_joint_speed_rad_s': .25}
        n.now = lambda: n.sim
        n.replay_executor = S(spin_once=self.step)
        self.monotonic = patch('m4a_runtime.time.monotonic', side_effect=lambda: self.wall)
        self.monotonic.start(); self.addCleanup(self.monotonic.stop)

    def joints(self, at=None, *, missing_velocity=False, nonfinite=False):
        names = list(NAMES)+['panda_finger_joint1', 'panda_finger_joint2']
        positions = [0.]*7+[.04, .04]
        velocity = [0.]*9
        if missing_velocity: velocity = []
        if nonfinite: velocity[0] = float('nan')
        self.node.receive_joints(S(name=names, position=positions, velocity=velocity,
                                   header=S(stamp=stamp(self.node.sim-2_000_000 if at is None else at))))

    def normal(self):
        self.node.sim += 20_000_000
        self.joints()

    def step(self, **kwargs):
        self.wall += .02
        self.tick += 1
        self.hook()

    def prime(self):
        self.node.prime_feedback()
        self.assertTrue(self.node.feedback_armed)
        self.assertTrue(self.node.feedback_diagnostic['fresh'])
        self.assertEqual([e['state'] for e in self.node.events], ['STARTUP_FEEDBACK', 'FEEDBACK_READY'])
        self.assertGreaterEqual(self.wall, 100.25)

    def test_normal_startup_waits_for_advancing_samples(self): self.prime()

    def setup_startup(self):
        n = self.node
        n.episode = 7
        n.owner.update(ready=True, stats={'planned_accepted': 0})
        n.owner_client = S(service_is_ready=lambda: True)
        n.planner = S(service_is_ready=lambda: True)
        n.action = S(server_is_ready=lambda: True)
        n.model_parameters = S(service_is_ready=lambda: True, call_async=lambda request: None)
        n.preflight_directory = Path('/simulated')
        model = '<robot><ros2_control><hardware><param name="mujoco_model">/simulated/scene-red.xml</param></hardware></ros2_control></robot>'
        n.future = lambda value: S(values=[S(type=ParameterType.PARAMETER_STRING, string_value=model),
                                          S(type=ParameterType.PARAMETER_BOOL, bool_value=True)])

    def test_real_startup_arms_feedback_before_post_start_settle(self):
        self.setup_startup()
        self.node.startup()
        self.assertTrue(self.node.feedback_armed)
        self.assertEqual(self.node.initial_episode, 7)
        self.assertGreaterEqual(self.node.sim, 10_750_000_000)

    def test_real_startup_still_rejects_previously_used_launch(self):
        self.setup_startup()
        self.node.owner['stats']['planned_accepted'] = 2
        with self.assertRaisesRegex(ValueError, 'Restart Terminal 1'): self.node.startup()
        self.assertFalse(self.node.feedback_armed)

    def test_joints_arrive_before_first_clock(self):
        self.node.sim = 0
        def arrival():
            if self.tick == 1: self.joints(10_000_000_000)
            elif self.tick < 5: pass
            else:
                self.node.sim = 10_000_000_000+(self.tick-4)*20_000_000
                self.joints()
        self.hook = arrival
        self.prime()

    def test_clock_arrives_before_joints(self):
        self.hook = lambda: self.normal() if self.tick > 8 else None
        self.prime()

    def test_initial_joint_stamp_behind_clock_can_recover(self):
        self.hook = lambda: self.joints(9_000_000_000) if self.tick < 5 else self.normal()
        self.prime()

    def test_initial_joint_stamp_ahead_of_clock_can_recover(self):
        self.hook = lambda: self.joints(10_080_000_000) if self.tick < 5 else self.normal()
        self.prime()

    def test_incoherent_sample_restarts_the_stable_window(self):
        def arrival():
            self.normal()
            if self.tick == 10: self.node.sim += 200_000_000
        self.hook = arrival
        self.prime()
        self.assertGreaterEqual(self.tick, 24)

    def test_missing_feedback_has_bounded_diagnostic_failure(self):
        self.hook = lambda: None
        with self.assertRaisesRegex(TimeoutError, 'missing_joint_fields'):
            self.node.prime_feedback()
        self.assertFalse(self.node.feedback_armed)
        self.assertLess(self.wall, 110.1)

    def test_missing_velocity_prevents_readiness(self):
        def arrival():
            self.node.sim += 20_000_000
            self.joints(missing_velocity=True)
        self.hook = arrival
        with self.assertRaisesRegex(TimeoutError, 'missing_joint_fields'):
            self.node.prime_feedback()
        self.assertFalse(self.node.feedback_armed)

    def test_duplicate_samples_cannot_refresh_watchdog(self):
        self.joints(); received = self.node.q_received
        self.wall += .2; self.joints()
        self.assertEqual(self.node.q_received, received)
        self.assertEqual(self.node.q_sequence, 1)
        self.assertEqual(self.node.q_ignored, 1)

    def test_old_sample_cannot_replace_current_state(self):
        self.joints(); previous = self.node.q_stamp
        self.joints(previous-1)
        self.assertEqual(self.node.q_stamp, previous)
        self.assertEqual(self.node.q_sequence, 1)

    def test_stopped_clock_and_duplicate_feedback_cannot_arm(self):
        self.hook = self.joints
        with self.assertRaisesRegex(TimeoutError, 'wall_feedback_timeout'):
            self.node.prime_feedback()
        self.assertFalse(self.node.feedback_armed)

    def test_wrong_owner_cannot_arm(self):
        self.node.owner['owner'] = 'SERVO'
        with self.assertRaisesRegex(ValueError, 'exclusive HOLD'): self.node.prime_feedback()
        self.assertFalse(self.node.feedback_armed)

    def test_nonfinite_feedback_fails_even_before_arming(self):
        self.hook = lambda: self.joints(nonfinite=True)
        with self.assertRaisesRegex(ValueError, 'Non-finite'): self.node.prime_feedback()

    def test_clock_rollback_fails_even_before_arming(self):
        self.node.clock_last = self.node.sim+1
        self.hook = lambda: None
        with self.assertRaisesRegex(ValueError, 'clock moved backwards'): self.node.prime_feedback()

    def test_move_cannot_start_before_readiness(self):
        with self.assertRaisesRegex(ValueError, 'readiness has not been confirmed'):
            self.node.move('assess')
        self.assertEqual(self.node.events, [])

    def armed(self):
        self.prime()
        self.hook = lambda: None

    def test_wall_staleness_still_fails_after_readiness(self):
        self.armed(); self.wall += .5
        with self.assertRaisesRegex(ValueError, 'wall_feedback_timeout'):
            self.node.until(lambda: True)

    def test_old_simulation_stamp_still_fails_after_readiness(self):
        self.armed(); self.node.sim = self.node.q_stamp+150_000_000
        with self.assertRaisesRegex(ValueError, 'joint_stamp_behind_clock'):
            self.node.until(lambda: True)

    def test_future_stamp_still_fails_after_readiness(self):
        self.armed(); self.node.q_stamp = self.node.sim+20_000_001
        with self.assertRaisesRegex(ValueError, 'joint_stamp_ahead_of_clock'):
            self.node.until(lambda: True)

    def test_cleanup_can_confirm_hold_after_feedback_failure(self):
        self.armed(); self.wall += 1.
        self.node.until(lambda: True, check=False)

    def test_active_wait_accepts_fresh_feedback(self):
        self.armed(); self.hook = self.normal
        self.node.until(lambda: True)


if __name__ == '__main__': unittest.main()
