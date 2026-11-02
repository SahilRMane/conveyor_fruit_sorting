"""Mixed MoveIt/Servo endpoint regression; transport is mocked, no ROS node."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
from unittest.mock import Mock
from types import SimpleNamespace as NS
import numpy as np,yaml
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/sorting_control/scripts'),str(R/'src/sorting_perception/scripts')]
from camera_sort import SortController
from sort_planning import execute_carried_plan
from sort_core import placement_scene
from publish_static_fixtures import scene_message
from m2b_common import timestamp
from ownership_core import NAMES
from rclpy.serialization import serialize_message

class PlanRecorded(RuntimeError):pass

class CarriedEndpointTests(unittest.TestCase):
    def controller(self):
        c=SortController.__new__(SortController)
        c.s=yaml.safe_load((R/'src/sorting_control/config/m3d.yaml').read_text())
        c.settings=yaml.safe_load((R/'src/sorting_control/config/m2b.yaml').read_text())
        c.c=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text())
        c.state=NS(time_ns=1_000_000_000,hand=np.array([.55,0.,.5758]))
        c.now_ns=lambda:c.state.time_ns
        c.hand_pose=lambda:(c.state.hand.copy(),c.settings['grasp_quaternion_xyzw'])
        c.scene=dict(estimated_attachment=True)
        c.result={};c.phase='transfer_raise';c.input_enabled=False;c.owner={};c.last_carried_goal=c.state.hand.copy()
        c.q={n:0. for n in NAMES};c.q.update(panda_finger_joint1=.01,panda_finger_joint2=.01)
        c.scene_clients=['servo_scene','planner_scene'];c.planner='planner';c.ensure_oracle_current=lambda:None
        c.command_function=None;c.issued=[]
        def acquire(mode,*args):
            c.owner=dict(owner=mode,busy=False)
            c.input_deadline=c.now_ns()+30_000_000_000
            return 1,timestamp(c.input_deadline)
        c.acquire_mode=acquire
        def until(predicate,timeout):
            c.state.time_ns+=4_000_000_000
            c.state.hand=np.asarray(c.command_function(c.now_ns())[0]).copy()
            self.assertTrue(predicate())
        c.until=until;c.wait_sim=lambda seconds:setattr(c.state,'time_ns',c.now_ns()+round(seconds*1e9))
        attached=placement_scene(scene_message(),[0,0,0,0],c.c,True,np.eye(4),False).robot_state.attached_collision_objects
        def call(client,request,**kwargs):
            if client==c.planner:
                c.request=request;serialize_message(request);raise PlanRecorded('request captured before live service')
            return NS(scene=NS(robot_state=NS(attached_collision_objects=copy.deepcopy(attached))))
        c.call=call
        actual=c.servo_to
        def servo(goal,*args,**kwargs):
            c.issued.append(np.asarray(goal).copy());return actual(goal,*args,**kwargs)
        c.servo_to=servo
        return c
    def next_plan(self,c):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(PlanRecorded):execute_carried_plan(c,[.545,.02,.6458],c.settings['grasp_quaternion_xyzw'],Path(folder),'test')
    def test_raise_replaces_retract_endpoint_before_next_plan(self):
        c=self.controller();raised=np.array([.55,0.,.6458]);c.servo_to(raised,3.)
        np.testing.assert_array_equal(c.last_carried_goal,raised)
        c.phase='transfer';self.next_plan(c)
        self.assertEqual(len(c.issued),1);self.assertAlmostEqual(c.hand_pose()[0][2],.6458)
        self.assertEqual(len(c.request.motion_plan_request.start_state.attached_collision_objects),8)
    def test_small_drift_corrects_to_raised_goal_and_not_retract_height(self):
        c=self.controller();raised=np.array([.55,0.,.6458]);c.servo_to(raised,3.)
        c.state.hand[2]-=.003;c.phase='transfer';self.next_plan(c)
        self.assertEqual(len(c.issued),2);np.testing.assert_array_equal(c.issued[-1],raised)
    def test_previous_state_replay_returns_to_low_endpoint(self):
        c=self.controller();old=c.last_carried_goal.copy();c.servo_to(np.array([.55,0.,.6458]),3.)
        c.last_carried_goal=old;c.phase='transfer';self.next_plan(c)
        np.testing.assert_array_equal(c.issued[-1],old);self.assertAlmostEqual(c.hand_pose()[0][2],.5758)
    def test_new_endpoint_is_copied_and_logged(self):
        c=self.controller();goal=np.array([.55,0.,.6458]);c.servo_to(goal,3.);goal[2]=0.
        self.assertAlmostEqual(c.last_carried_goal[2],.6458)
        record=c.result['carried_goal_history'][-1]
        self.assertEqual(record['phase'],'transfer_raise');self.assertAlmostEqual(record['hand_goal_xyz'][2],.6458)
    def test_failed_servo_does_not_commit_an_unreached_endpoint(self):
        c=self.controller();old=c.last_carried_goal.copy();c.until=Mock(side_effect=TimeoutError('endpoint not reached'))
        with self.assertRaises(TimeoutError):c.servo_to(np.array([.55,0.,.6458]),3.)
        np.testing.assert_array_equal(c.last_carried_goal,old);self.assertNotIn('carried_goal_history',c.result)
    def test_failed_hold_does_not_commit_endpoint(self):
        c=self.controller();old=c.last_carried_goal.copy();original=c.acquire_mode
        def acquire(mode,*args):
            if mode=='HOLD':raise RuntimeError('handoff failed')
            return original(mode,*args)
        c.acquire_mode=acquire
        with self.assertRaisesRegex(RuntimeError,'handoff failed'):c.servo_to(np.array([.55,0.,.6458]),3.)
        np.testing.assert_array_equal(c.last_carried_goal,old)
    def test_detached_retreat_does_not_replace_a_carried_endpoint(self):
        c=self.controller();c.scene['estimated_attachment']=False;old=c.last_carried_goal.copy()
        c.servo_to(np.array([.55,0.,.6458]),3.)
        np.testing.assert_array_equal(c.last_carried_goal,old)
    def test_live_trace_contains_commanded_descent_without_support_contact(self):
        f=json.loads((R/'tests/fixtures/m3d-stale-carried-endpoint.json').read_text())
        self.assertGreater(f['raised_state']['tray_xyz'][2],.42)
        self.assertLess(f['last_servo_sample']['target_xyz'][2],f['raised_state']['hand_xyz'][2]-.04)
        self.assertLess(f['clearance_failure_state']['tray_xyz'][2]-f['grasp_baseline_z_m'],.085)
        self.assertEqual(f['clearance_failure_state']['support_force_N']+sum(f['clearance_failure_state']['bin_floor_force_N']),0.)

if __name__=='__main__':unittest.main()
