"""Replay startup clock lag with the real guard and bounded executor. No ROS node."""
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from rclpy.executors import TimeoutException
from rclpy.task import Task
import test_m4c_callbacks as callbacks


class M4CStartupClockTests(unittest.TestCase):
    def setUp(self):
        self.h=callbacks.M4CCallbackTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.n=self.h.n;self.r=self.h.replay;self.e=self.h.executor
        self.e.startup_clock_pending=self.n.startup_clock_pending
        self.r.sim=9_169_999_999;self.n.q_stamp=9_193_999_999
        self.n.clock_last=8_681_999_999;self.n.q_received=self.r.wall
        self.n.events=[dict(state='FEEDBACK_READY',enter_ns=self.n.clock_last)]
        self.n.tracking_active=False
        self.predicate=Mock(return_value=True);self.clock_after=None;self.clock_sent=False
        self.e.wait_for_ready_callbacks.side_effect=self.ready

    def ready(self,timeout_sec=None):
        if self.h.queue:return self.h.ready(timeout_sec=timeout_sec)
        if self.clock_after is not None and self.r.wall>=self.clock_after and not self.clock_sent:
            self.clock_sent=True
            return Task(lambda:setattr(self.r,'sim',self.n.q_stamp)),S(topic_name='/clock'),S(get_name=lambda:'m4a_inspection_executor')
        self.r.wall+=float(timeout_sec or 0.)
        raise TimeoutException()

    def test_exact_reported_future_sample_waits_for_clock(self):
        self.clock_after=self.r.wall+.02
        self.n.until(self.predicate)
        self.predicate.assert_called_once()
        self.assertTrue(self.clock_sent);self.assertTrue(self.n.feedback_diagnostic['fresh'])
        self.assertEqual(self.n.q_stamp,9_193_999_999)
        self.assertEqual(self.r.commands,[])
        self.assertEqual(self.e.callback_diagnostic['startup_clock_waits'],1)
        self.assertLessEqual(self.e.callback_diagnostic['startup_clock_wait_wall_s'],.05)

    def test_frozen_clock_still_fails_with_original_twenty_ms_limit(self):
        with self.assertRaisesRegex(ValueError,'joint_stamp_ahead_of_clock'):
            self.n.until(self.predicate)
        self.predicate.assert_not_called()
        self.assertEqual(self.r.sim,9_169_999_999);self.assertEqual(self.n.q_stamp,9_193_999_999)
        self.assertLess(self.r.wall,100.061)
        self.assertEqual(self.r.commands,[])

    def test_real_post_readiness_settle_wait_recovers(self):
        self.r.sim=8_681_999_999;self.n.q_stamp=self.r.sim-2_000_000
        def receive_later_joints():
            self.r.sim=9_169_999_999;self.n.q_stamp=9_193_999_999;self.n.q_received=self.r.wall
        self.h.queue.append(receive_later_joints);self.clock_after=self.r.wall+.02
        self.n.wait_sim(.5)
        self.assertTrue(self.n.feedback_diagnostic['fresh'])
        self.assertEqual(self.r.commands,[])

    def test_slow_startup_callback_is_named_in_diagnostic(self):
        def slow():self.r.wall+=.244325375;self.n.q_received=self.r.wall
        self.e.wait_for_ready_callbacks.side_effect=[
            (Task(slow),S(topic_name='/tf_static'),S(get_name=lambda:'m4a_inspection_executor'))]
        original=self.e.wait_for_ready_callbacks.side_effect
        def ready(timeout_sec=None):
            try:return next(original)
            except StopIteration:return self.ready(timeout_sec=timeout_sec)
        self.e.wait_for_ready_callbacks.side_effect=ready
        self.clock_after=self.r.wall+.25
        self.n.until(self.predicate)
        d=self.e.callback_diagnostic
        self.assertEqual(d['maximum_callback']['name'],'/tf_static')
        self.assertGreater(d['maximum_callback_wall_s'],.244)
        self.assertTrue(self.n.feedback_diagnostic['fresh'])

    def test_active_moving_guard_does_not_wait_for_future_feedback(self):
        self.n.tracking_active=True
        with self.assertRaisesRegex(ValueError,'joint_stamp_ahead_of_clock'):
            self.n.until(self.predicate)
        self.assertEqual(self.e.callback_diagnostic['startup_clock_waits'],0)
        self.predicate.assert_not_called()

    def test_startup_alignment_requires_stationary_exclusive_hold(self):
        changes=[('owner',dict(owner='SERVO',busy=False)),('owner',dict(owner='HOLD',busy=True)),
                 ('active_goal',object()),('refinement_active',True),('belt_speed',.001),
                 ('policy_started',(self.r.sim,self.r.wall)),('feedback_armed',False)]
        for field,value in changes:
            with self.subTest(field=field,value=value):
                old=getattr(self.n,field,None);setattr(self.n,field,value)
                self.assertFalse(self.n.startup_clock_pending());setattr(self.n,field,old)

    def test_stale_wall_source_never_enters_alignment(self):
        self.n.q_received=self.r.wall-.5
        with self.assertRaisesRegex(ValueError,'wall_feedback_timeout'):
            self.n.until(self.predicate)
        self.assertEqual(self.e.callback_diagnostic['startup_clock_waits'],0)

    def test_old_joint_stamp_never_enters_alignment(self):
        self.n.q_stamp=self.r.sim-150_000_000
        with self.assertRaisesRegex(ValueError,'joint_stamp_behind_clock'):
            self.n.until(self.predicate)
        self.assertEqual(self.e.callback_diagnostic['startup_clock_waits'],0)

    def test_continuously_ready_future_source_has_callback_count_bound(self):
        self.e.wait_for_ready_callbacks.side_effect=lambda **kw:(Task(lambda:None),None,None)
        with self.assertRaisesRegex(ValueError,'joint_stamp_ahead_of_clock'):
            self.n.until(self.predicate)
        self.assertEqual(self.e.callback_diagnostic['startup_clock_callbacks'],64)
        self.assertEqual(self.e.callback_diagnostic['callbacks'],128)
        self.predicate.assert_not_called()

    def test_nonfinite_joint_field_never_enters_alignment(self):
        self.n.v[next(iter(self.n.v))]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite_joint_fields'):
            self.n.until(self.predicate)
        self.assertEqual(self.e.callback_diagnostic['startup_clock_waits'],0)

    def test_missing_velocity_never_enters_alignment(self):
        self.n.v={}
        with self.assertRaisesRegex(ValueError,'missing_joint_fields'):
            self.n.until(self.predicate)
        self.assertEqual(self.e.callback_diagnostic['startup_clock_waits'],0)

    def test_clock_rollback_is_not_recovered(self):
        self.n.clock_last=self.r.sim+1
        with self.assertRaisesRegex(ValueError,'clock moved backwards'):
            self.n.until(self.predicate)
        self.assertEqual(self.e.callback_diagnostic['startup_clock_waits'],0)

    def test_fault_during_alignment_prevents_predicate(self):
        self.h.queue.append(lambda:setattr(self.n,'failure','Ownership fault'))
        with self.assertRaisesRegex(ValueError,'Ownership fault'):
            self.n.until(self.predicate)
        self.predicate.assert_not_called()

    def test_callback_error_during_alignment_propagates(self):
        count=0
        def ready(timeout_sec=None):
            nonlocal count
            count+=1
            if count==1:raise TimeoutException()
            def fail():raise ValueError('alignment callback failed')
            return Task(fail),None,None
        self.e.wait_for_ready_callbacks.side_effect=ready
        with self.assertRaisesRegex(ValueError,'alignment callback failed'):
            self.n.until(self.predicate)
        self.assertEqual(self.e.callback_diagnostic['stop_reason'],'callback_error')


if __name__=='__main__':unittest.main()
