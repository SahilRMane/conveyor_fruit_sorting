"""Replay a zero command at an older physics timestamp. No ROS nodes."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import Mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4c_evaluator import MovingEvaluator
from test_m4c import finish_transport


class M4CShutdownTests(unittest.TestCase):
    def setUp(self):
        self.e=object.__new__(MovingEvaluator)
        self.e.c={'belt_speed_m_s':.001};self.e.window_start_ns=10_000_000_000
        self.e.window_end_ns=None;self.e.latest_transport_ns=None
        self.e.transport=[];self.e.transport_failure=None;self.e.failure=None
        self.e.get_clock=lambda:S(now=lambda:S(nanoseconds=20_000_000_000))

    def sample(self,ns,speed=.001,fresh=True,force=.25,vx=.001):
        self.e.receive_transport(S(data=json.dumps(dict(sim_time=ns/1e9,requested_speed_m_s=speed,
            command_fresh=fresh,force_x_N=force,tray_vx_m_s=vx))))

    def control(self,cutoff,step):
        c=S(now=lambda:cutoff,stop=Mock())
        def until(predicate,timeout):
            self.assertFalse(c.stop.called)
            if not predicate():step()
            if not predicate():raise TimeoutError('Transport cutoff acknowledgement timed out')
        c.until=until;return c

    def test_old_order_reproduces_reported_error_at_shutdown(self):
        self.e.window_end_ns=20_000_000_000
        self.sample(19_998_000_000,speed=0.)
        self.assertEqual(self.e.failure,'Conveyor command stopped during the moving inspection')
        self.assertEqual(self.e.transport_failure['sample']['requested_speed_m_s'],0.)

    def test_waits_for_physics_cutoff_before_stop(self):
        self.sample(19_998_000_000)
        c=self.control(20_000_000_000,lambda:self.sample(20_002_000_000))
        finish_transport(c,self.e);c.stop.assert_called_once()
        self.sample(20_002_000_000,speed=0.)
        self.assertIsNone(self.e.failure)

    def test_exact_cutoff_acknowledges_half_open_window(self):
        c=self.control(20_000_000_000,lambda:self.sample(20_000_000_000))
        finish_transport(c,self.e);c.stop.assert_called_once()
        self.sample(20_000_000_000,speed=0.)
        self.assertIsNone(self.e.failure)

    def test_missing_cutoff_sample_fails_and_does_not_stop_in_helper(self):
        c=self.control(20_000_000_000,lambda:None)
        with self.assertRaises(TimeoutError):finish_transport(c,self.e)
        c.stop.assert_not_called() # The main failure handler still performs cleanup.

    def test_in_window_zero_command_is_still_a_failure(self):
        self.sample(19_000_000_000,speed=0.)
        c=self.control(20_000_000_000,lambda:self.sample(20_000_000_000))
        with self.assertRaisesRegex(ValueError,'command stopped'):finish_transport(c,self.e)
        c.stop.assert_not_called();self.assertEqual(self.e.transport_failure['sample_ns'],19_000_000_000)

    def test_delayed_in_window_fault_is_not_hidden_by_close(self):
        c=self.control(20_000_000_000,lambda:(self.sample(19_998_000_000,fresh=False),self.sample(20_002_000_000)))
        with self.assertRaisesRegex(ValueError,'command stopped'):finish_transport(c,self.e)
        self.assertEqual(self.e.transport_failure['window_end_ns'],20_000_000_000)

    def test_force_and_speed_limits_remain_active(self):
        for changes in ({'force':9.},{'vx':.01}):
            with self.subTest(changes=changes):
                self.e.failure=None;self.e.transport_failure=None
                self.sample(19_000_000_000,**changes)
                self.assertIn('force or speed',self.e.failure)

    def test_first_fault_sample_is_retained(self):
        self.sample(19_000_000_000,fresh=False);self.sample(19_020_000_000,speed=0.)
        self.assertEqual(self.e.transport_failure['sample_ns'],19_000_000_000)
        self.assertFalse(self.e.transport_failure['sample']['command_fresh'])

    def test_out_of_window_zero_does_not_replace_existing_fault(self):
        self.sample(19_000_000_000,fresh=False);self.e.window_end_ns=20_000_000_000
        self.sample(20_100_000_000,speed=0.)
        self.assertIsNotNone(self.e.failure)
        self.assertEqual(self.e.transport_failure['sample_ns'],19_000_000_000)

    def test_out_of_order_sample_does_not_move_acknowledgement_backwards(self):
        self.sample(20_000_000_000);self.sample(19_900_000_000)
        self.assertEqual(self.e.latest_transport_ns,20_000_000_000)


if __name__=='__main__':unittest.main()
