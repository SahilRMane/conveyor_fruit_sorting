"""Repeated trial/reset feedback checks with real ROS messages; no ROS node."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys,tempfile,unittest
import yaml
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/sorting_control/scripts'),str(R/'src/sorting_perception/scripts')]
from camera_pickup import Controller
from camera_sort import SortController
from ownership_core import NAMES
from m2b_common import timestamp
from sensor_msgs.msg import JointState
from rclpy.serialization import serialize_message,deserialize_message

class ClassificationComplete(RuntimeError):pass

class TrialJointGuardTests(unittest.TestCase):
    def controller(self):
        c=SortController.__new__(SortController)
        c.c=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text())
        c.q={};c.v={};c.q_stamp_ns=0;c.previous_q=[1.]*7;c.previous_q_stamp_ns=1
        c.maximum_joint_step=.01;c.maximum_joint_speed=.2
        c.active=False;c.cleanup=False;c.abort=False;c.epoch=NS(episode_id=0)
        c.pending=NS(trial_id=1,episode_id=0);c.set_phase=lambda phase:setattr(c,'phase',phase)
        c.publish_decision=lambda route:None
        return c
    def sample(self,c,position=0.,speed=0.,stamp=None):
        m=JointState();m.header.stamp=timestamp(stamp if stamp is not None else c.q_stamp_ns+2_000_000)
        m.name=list(NAMES);m.position=[float(position)]*7;m.velocity=[float(speed)]*7
        # Exercise the generated message schema as well as the controller callback.
        c.receive_joints(deserialize_message(serialize_message(m),JointState))
    def classify_trial(self,c,position=0.):
        def classify():
            self.assertEqual(c.phase,'classify_initial')
            self.sample(c,position);self.sample(c,position+.001)
            raise ClassificationComplete('classification boundary reached')
        c.select_class=classify
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ClassificationComplete):c.pickup(Path(folder))
    def test_twenty_reset_postures_do_not_compare_with_previous_trial(self):
        c=self.controller()
        for trial in range(20):
            c.active=False;c.pending.trial_id=trial+1
            c.receive_epoch(NS(episode_id=trial+1))
            self.sample(c,position=trial*.1) # Reset feedback is inactive.
            self.classify_trial(c,position=trial*.1)
            self.assertAlmostEqual(c.maximum_joint_step,.001)
            self.assertEqual(c.maximum_joint_speed,0.)
    def test_trial_start_clears_old_state_without_needing_an_epoch_callback(self):
        c=self.controller();self.classify_trial(c)
        self.assertAlmostEqual(c.previous_q[0],.001)
        self.assertIsNotNone(c.previous_q_stamp_ns)
    def test_previous_startup_order_reproduces_reported_failure(self):
        c=self.controller();c.active=True
        with self.assertRaisesRegex(RuntimeError,'joint feedback step'):self.sample(c,0.)
        self.assertEqual(c.joint_feedback_diagnostic['maximum_step_rad'],1.)
    def test_active_step_still_fails_with_previous_and_current_sample_evidence(self):
        c=self.controller();c.reset_joint_guard();c.active=True;self.sample(c)
        with self.assertRaisesRegex(RuntimeError,'joint feedback step'):self.sample(c,.021)
        d=c.joint_feedback_diagnostic
        self.assertEqual(d['previous_positions_rad'],[0.]*7)
        self.assertEqual(d['positions_rad'],[.021]*7)
        self.assertEqual(d['step_limit_rad'],.02)
        self.assertLess(d['previous_stamp_ns'],d['stamp_ns'])
    def test_active_speed_limit_also_applies_to_first_sample(self):
        c=self.controller();c.reset_joint_guard();c.active=True
        with self.assertRaisesRegex(RuntimeError,'joint speed bound'):self.sample(c,speed=c.c['maximum_joint_speed_rad_s'])
    def test_nonfinite_first_feedback_still_fails(self):
        for value in (float('nan'),float('inf')):
            c=self.controller();c.reset_joint_guard();c.active=True
            with self.assertRaisesRegex(RuntimeError,'Invalid joint feedback'):self.sample(c,position=value)
    def test_missing_velocity_has_a_clear_failure(self):
        c=self.controller();c.reset_joint_guard();c.active=True
        m=JointState();m.header.stamp=timestamp(2_000_000);m.name=list(NAMES);m.position=[0.]*7
        with self.assertRaisesRegex(RuntimeError,'Incomplete joint feedback'):c.receive_joints(m)
    def test_active_epoch_change_aborts_without_clearing_comparison_state(self):
        c=self.controller();c.active=True;c.previous_q=[.1]*7
        c.receive_epoch(NS(episode_id=1))
        self.assertTrue(c.abort);self.assertEqual(c.previous_q,[.1]*7)
    def test_guard_cannot_be_cleared_during_active_motion(self):
        c=self.controller();c.active=True
        with self.assertRaisesRegex(RuntimeError,'active trial'):c.reset_joint_guard()
        self.assertEqual(c.previous_q,[1.]*7)
    def test_duplicate_epoch_does_not_clear_active_guard(self):
        c=self.controller();c.active=True;c.receive_epoch(NS(episode_id=0))
        self.assertFalse(c.abort);self.assertEqual(c.previous_q,[1.]*7)
    def test_duplicate_and_old_feedback_do_not_advance_active_baseline(self):
        c=self.controller();c.reset_joint_guard();c.active=True;self.sample(c)
        stamp=c.q_stamp_ns
        self.sample(c,position=1.,stamp=stamp);self.sample(c,position=1.,stamp=stamp-1)
        self.assertEqual(c.previous_q,[0.]*7);self.assertEqual(c.q_stamp_ns,stamp)
    def test_base_pickup_keeps_classification_baseline_for_first_plan(self):
        c=self.controller();c.owner={}
        def classify():self.sample(c);return {'action':'PICK_PLACE'}
        c.select_class=classify
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'Arm gate readiness'):c.pickup(Path(folder))
        self.assertEqual(c.previous_q,[0.]*7)
        with self.assertRaisesRegex(RuntimeError,'joint feedback step'):self.sample(c,.021)
    def test_base_checkpoint_also_initializes_guard_before_activation(self):
        c=self.controller();c.owner={}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'Arm gate readiness'):Controller.pickup(c,Path(folder))
        self.assertIsNone(c.previous_q);self.sample(c)
    def test_covered_pass_path_keeps_joint_guard_active(self):
        c=self.controller();c.owner={'owner':'HOLD','busy':False};c.input_enabled=False
        c.s={'pass_transport_s':.1};c.send_belt=lambda:None
        def classify():
            self.sample(c);c.q.update(panda_finger_joint1=.04,panda_finger_joint2=.04)
            return {'action':'PASS'}
        c.select_class=classify
        c.wait_sim=lambda seconds:self.sample(c,.021)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'joint feedback step'):c.pickup(Path(folder))

if __name__=='__main__':unittest.main()
