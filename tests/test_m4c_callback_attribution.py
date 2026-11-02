"""Separate startup and moving callbacks without creating a ROS node."""
import unittest
from types import SimpleNamespace as S
from rclpy.task import Task
import test_m4c_callbacks as callbacks


class M4CCallbackAttributionTests(unittest.TestCase):
    def setUp(self):
        self.h=callbacks.M4CCallbackTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.e=self.h.executor;self.phase='startup'
        self.e.diagnostic_context=lambda:dict(phase=self.phase,state='test',belt_speed_m_s=.001 if self.phase=='moving' else 0.)

    def batch(self,seconds):
        self.h.queue.append(lambda:setattr(self.h.replay,'wall',self.h.replay.wall+seconds))
        self.e.spin_once(timeout_sec=0.)
        return self.e.callback_diagnostic

    def test_startup_peak_does_not_become_moving_peak(self):
        self.batch(.262);self.phase='moving';r=self.batch(.002)
        self.assertEqual(r['maximum_batch']['phase'],'startup')
        moving=r['batches_by_phase']['moving']
        self.assertAlmostEqual(moving['maximum_batch_wall_s'],.002)
        self.assertIsNone(moving['first_over_budget_batch'])
        self.assertGreater(r['batches_by_phase']['startup']['maximum_batch_wall_s'],.26)

    def test_moving_overrun_is_retained_separately(self):
        self.batch(.262);self.phase='moving';self.batch(.02);r=self.batch(.03)
        moving=r['batches_by_phase']['moving']
        self.assertEqual(moving['batches'],2)
        self.assertAlmostEqual(moving['first_over_budget_batch']['batch_wall_s'],.02)
        self.assertAlmostEqual(moving['maximum_batch_wall_s'],.03)
        self.assertEqual(r['first_over_budget_batch']['phase'],'startup')

    def test_timer_callback_name_and_period_are_visible(self):
        def belt_heartbeat():self.h.replay.wall+=.02
        timer=S(callback=belt_heartbeat,timer_period_ns=100_000_000)
        values=iter([(Task(belt_heartbeat),timer,S(get_name=lambda:'inspection'))])
        self.e.wait_for_ready_callbacks.side_effect=lambda **kw:next(values)
        r=self.batch(0.)
        trace=r['maximum_batch']['slowest_callback']
        self.assertTrue(trace['callback'].endswith('belt_heartbeat'))
        self.assertEqual(trace['timer_period_ns'],100_000_000)
        self.assertEqual(trace['node'],'inspection')

    def test_phase_transition_is_not_assigned_to_stationary_or_moving(self):
        def transition():self.phase='moving';self.h.replay.wall+=.02
        self.h.queue.append(transition);self.e.spin_once(timeout_sec=0.)
        r=self.e.callback_diagnostic
        self.assertEqual(r['current_batch']['phase'],'phase_transition')
        self.assertEqual(r['current_batch']['start_context']['phase'],'startup')
        self.assertEqual(r['current_batch']['end_context']['phase'],'moving')
        self.assertNotIn('moving',r['batches_by_phase'])

    def test_empty_wait_does_not_create_phase_observation(self):
        self.e.spin_once(timeout_sec=0.)
        self.assertEqual(self.e.callback_diagnostic['batches_by_phase'],{})

    def test_cleanup_is_separate_from_moving(self):
        self.phase='moving';self.batch(.002);self.phase='cleanup';r=self.batch(.07)
        self.assertAlmostEqual(r['batches_by_phase']['moving']['maximum_batch_wall_s'],.002)
        self.assertAlmostEqual(r['batches_by_phase']['cleanup']['maximum_batch_wall_s'],.07)

    def test_first_phase_peak_stays_unchanged_after_later_batches(self):
        self.batch(.262);snapshot=self.e.callback_diagnostic['batches_by_phase']['startup']
        self.phase='moving';self.batch(.02)
        self.assertEqual(snapshot['maximum_batch']['start_context']['phase'],'startup')
        self.assertEqual(snapshot['maximum_batch']['end_context']['phase'],'startup')


if __name__=='__main__':unittest.main()
