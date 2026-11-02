"""Replay the planning-boundary clock/feedback race. No ROS nodes or DDS."""
from collections import deque
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/sorting_supervisor/scripts'))
from rclpy.executors import TimeoutException
from rclpy.task import Task, Future
from m4c_runtime import FeedbackExecutor
import test_m4c_delivery as delivery


class ReplayFeedbackExecutor(FeedbackExecutor):
    # No ROS resources exist in this node-free replay.
    def __del__(self): pass


class M4CCallbackTests(unittest.TestCase):
    def setUp(self):
        self.replay = delivery.M4CDeliveryTests()
        self.replay.setUp()
        self.addCleanup(self.replay.doCleanups)
        self.n = self.replay.n
        self.queue = deque()
        self.executor = object.__new__(ReplayFeedbackExecutor)
        self.executor.callback_budget_s = .015
        self.executor.callback_diagnostic = {}
        self.executor.wait_for_ready_callbacks = Mock(side_effect=self.ready)
        self.n.replay_executor = self.executor

    def ready(self, timeout_sec=None):
        if not self.queue: raise TimeoutException()
        return Task(self.queue.popleft()), None, None

    def planning_boundary(self, *, fresh=True, track=1, future=False):
        h = self.replay
        h.sim = 75_658_000_000
        self.n.q_stamp = h.sim
        self.n.q_received = h.wall
        self.n.overhead = None
        h.deliver(75_578_000_000)
        self.n.bound_track = 1
        self.n.tracking_active = True
        result = Future()
        h.wall += .076

        def clock():
            h.sim = 75_734_000_000

        def joints():
            self.n.q_received = h.wall
            self.n.q_stamp = h.sim-2_000_000

        self.queue.extend([clock, joints])
        if fresh:
            self.queue.append(lambda: h.deliver(75_748_000_000 if future else 75_648_000_000, track=track))
        self.queue.append(lambda: result.set_result('planning response'))
        return result

    def test_single_callback_reproduces_reported_156_ms_failure(self):
        result = self.planning_boundary()
        self.n.replay_executor = Mock(spin_once=lambda **kw: self.queue.popleft()())
        with self.assertRaisesRegex(ValueError, 'age_s=0.156000'):
            self.n.future(result)
        self.assertFalse(result.done())

    def test_ready_feedback_runs_before_planning_future_guard(self):
        result = self.planning_boundary()
        self.assertEqual(self.n.future(result), 'planning response')
        self.assertAlmostEqual(self.n.overhead_diagnostic['capture_age_sim_s'], .086)
        self.assertAlmostEqual(self.n.feedback_diagnostic['feedback_age_sim_s'], .002)
        self.assertEqual(self.executor.callback_diagnostic['callbacks'], 4)
        self.assertEqual(self.executor.callback_diagnostic['stop_reason'], 'queue_empty')

    def test_no_new_capture_still_fails_at_same_limit(self):
        with self.assertRaisesRegex(ValueError, 'age_s=0.156000'):
            self.n.future(self.planning_boundary(fresh=False))

    def test_track_change_still_fails_before_future_is_used(self):
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.n.future(self.planning_boundary(track=2))

    def test_future_capture_still_fails(self):
        with self.assertRaisesRegex(ValueError, 'stale or future'):
            self.n.future(self.planning_boundary(future=True))

    def test_joint_source_failure_is_not_hidden_by_fresh_capture(self):
        result = self.planning_boundary()
        self.queue.remove(self.queue[1])
        self.n.q_received = self.replay.wall-1.
        with self.assertRaisesRegex(ValueError, 'Joint feedback became stale'):
            self.n.future(result)

    def test_queue_is_never_waited_on_after_first_ready_callback(self):
        self.queue.extend([lambda: None, lambda: None])
        self.executor.spin_once(timeout_sec=.01)
        self.assertEqual([c.kwargs['timeout_sec'] for c in self.executor.wait_for_ready_callbacks.call_args_list],
                         [.01, 0., 0.])

    def test_continuously_ready_queue_has_callback_count_bound(self):
        self.executor.wait_for_ready_callbacks.side_effect = lambda **kw: (Task(lambda: None), None, None)
        self.executor.spin_once(timeout_sec=.01)
        self.assertEqual(self.executor.callback_diagnostic['callbacks'], 64)
        self.assertEqual(self.executor.callback_diagnostic['stop_reason'], 'callback_count')

    def test_slow_callback_does_not_create_unbounded_drain(self):
        self.queue.extend([lambda: setattr(self.replay, 'wall', self.replay.wall+.020), lambda: None])
        self.executor.spin_once(timeout_sec=.01)
        self.assertEqual(len(self.queue), 1)
        self.assertEqual(self.executor.callback_diagnostic['stop_reason'], 'wall_budget')
        self.assertGreater(self.executor.callback_diagnostic['maximum_callback_wall_s'], .015)

    def test_task_error_propagates(self):
        def fail(): raise ValueError('callback failed')
        self.queue.append(fail)
        with self.assertRaisesRegex(ValueError, 'callback failed'):
            self.executor.spin_once(timeout_sec=.01)


if __name__ == '__main__': unittest.main()
