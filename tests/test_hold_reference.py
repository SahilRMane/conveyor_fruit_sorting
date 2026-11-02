"""Loaded-hold reference regression, using installed Jazzy controller messages."""
import copy,sys,threading,time,unittest
from pathlib import Path
from unittest.mock import Mock
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.serialization import serialize_message,deserialize_message
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src/sorting_control/scripts'))
from hold_reference import select_hold
from ownership_core import NAMES
from arm_command_gate import CommandGate
Q=[0.,0.,0.,-1.57,0.,1.57,-.785]
class HoldReferenceTests(unittest.TestCase):
    def sample(self):return dict(names=NAMES[:],positions=[a+.01 for a in Q],stamp_ns=1_000_000_000,received_wall=10.)
    def test_default_preserves_existing_measured_hold(self):
        values,d=select_hold(Q,None,1_020_000_000,10.02);self.assertEqual(values,Q);self.assertEqual(d['policy'],'measured')
    def test_enabled_hold_retains_existing_position_reference(self):
        s=self.sample();values,d=select_hold(Q,s,1_020_000_000,10.02,True);self.assertEqual(values,s['positions']);self.assertAlmostEqual(d['maximum_delta_rad'],.01)
    def test_joint_order_is_reordered_by_name(self):
        s=self.sample();s['names'].reverse();s['positions'].reverse();values,_=select_hold(Q,s,1_020_000_000,10.02,True);self.assertEqual(values,[a+.01 for a in Q])
    def test_missing_stale_future_nonfinite_and_excessive_bias_fail(self):
        bad=[None]
        for key,value in [('stamp_ns',0),('stamp_ns',2_000_000_000),('received_wall',0),('positions',[float('nan')]*7),('positions',[a+.06 for a in Q]),('names',NAMES[:-1])]:
            s=self.sample();s[key]=value;bad.append(s)
        for s in bad:
            with self.assertRaises(RuntimeError):select_hold(Q,s,1_020_000_000,10.02,True)
    def test_actual_jazzy_message_uses_reference_not_feedback_or_output(self):
        m=JointTrajectoryControllerState();m.joint_names=NAMES;m.reference.positions=self.sample()['positions'];m.feedback.positions=Q
        decoded=deserialize_message(serialize_message(m),JointTrajectoryControllerState);self.assertEqual(list(decoded.reference.positions),self.sample()['positions'])
    def gate(self,preserve):
        g=CommandGate.__new__(CommandGate);g.core=Mock(lock=threading.RLock());g.q=dict(zip(NAMES,Q));g.controller_reference=self.sample();g.controller_reference['received_wall']=time.monotonic();g.now_ns=lambda:1_020_000_000;g.feedback_fresh=lambda:True;g.raw_pub=Mock();g.event=Mock()
        parameters={'preserve_hold_reference':preserve,'maximum_hold_reference_delta_rad':.05,'maximum_controller_reference_age_s':.15,'feedback_timeout_wall_s':.75};g.p=lambda key:parameters[key]
        return g
    def test_real_gate_publishes_reference_and_keeps_measured_feedback_separate(self):
        g=self.gate(True);g.publish_hold();msg=g.raw_pub.publish.call_args.args[0];self.assertEqual(list(msg.points[0].positions),self.sample()['positions']);self.assertEqual(g.q,dict(zip(NAMES,Q)));self.assertEqual(list(msg.points[0].velocities),[0.]*7)
    def test_real_gate_default_remains_measured(self):
        g=self.gate(False);g.publish_hold();self.assertEqual(list(g.raw_pub.publish.call_args.args[0].points[0].positions),Q)
    def test_stale_reference_produces_no_raw_command(self):
        g=self.gate(True);g.controller_reference['stamp_ns']=0
        with self.assertRaises(RuntimeError):g.publish_hold()
        g.raw_pub.publish.assert_not_called()
    def test_invalid_measured_feedback_still_blocks_hold(self):
        g=self.gate(True);g.feedback_fresh=lambda:False
        with self.assertRaises(RuntimeError):g.publish_hold()
        g.raw_pub.publish.assert_not_called()
    def test_reference_callback_does_not_refresh_duplicate_stamp(self):
        g=self.gate(True);before=copy.deepcopy(g.controller_reference);m=JointTrajectoryControllerState();m.header.stamp.sec=1;m.joint_names=NAMES;m.reference.positions=[a+.02 for a in Q];g.receive_controller_reference(m);self.assertEqual(g.controller_reference,before)
    def test_reference_policy_is_enabled_only_in_m3d_launch(self):
        self.assertIn('"preserve_hold_reference": True',(R/'src/sorting_control/launch/panda_m3d.launch.py').read_text())
        for name in ('panda_m3b.launch.py','panda_m3c.launch.py','panda_m2b.launch.py'):
            self.assertNotIn('"preserve_hold_reference": True',(R/'src/sorting_control/launch'/name).read_text())
class PlanningEntryTests(unittest.TestCase):
    def test_bad_loaded_orientation_cannot_reach_planner_or_action(self):
        import numpy as np,yaml
        from sort_planning import execute_carried_plan
        from m2b_common import timestamp
        c=Mock();c.s=yaml.safe_load((R/'src/sorting_control/config/m3d.yaml').read_text());c.last_carried_goal=None
        c.hand_pose.return_value=(np.array([.68,0,.575]),(np.array([.72,.69,0.,0.])/np.linalg.norm([.72,.69,0.,0.])).tolist());c.acquire_mode.return_value=(3,timestamp(55_000_000_000))
        with self.assertRaisesRegex(RuntimeError,'orientation changed during the planning handoff'):
            execute_carried_plan(c,[.55,0,.575],[.7071067811865476,.7071067811865476,0.,0.],Path('/tmp'),'bad-start')
        c.servo_to.assert_called_once();c.call.assert_not_called();c.action.send_goal_async.assert_not_called()
if __name__=='__main__':unittest.main()
