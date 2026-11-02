"""Release action semantics and the known resting-plane model guard."""
import copy,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import numpy as np,yaml
from rclpy.serialization import serialize_message
from action_msgs.msg import GoalStatus
from sorting_interfaces.srv import SetCameraPlacement
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/sorting_control/scripts'),str(R/'src/sorting_perception/scripts')]
from camera_sort import SortController
from camera_sort_scene import SortScene
from sort_core import resting_pose_from_release
C=yaml.safe_load((R/'src/sorting_control/config/m3d.yaml').read_text())

class ReleaseTests(unittest.TestCase):
    def controller(self,reached=False,stalled=True,status=GoalStatus.STATUS_SUCCEEDED):
        c=SortController.__new__(SortController);c.s=C;c.phase='release';c.input_enabled=True
        c.owner=dict(owner='SERVO',busy=False);c.now_ns=lambda:1_000_000_000;c.input_deadline=30_000_000_000
        c.result={};c.wait_sim=Mock();c.future=lambda value,*args:value;c.hand=Mock()
        handle=Mock(accepted=True);handle.get_result_async.return_value=NS(status=status,result=NS(reached_goal=reached,stalled=stalled))
        c.hand.send_goal_async.return_value=handle;return c
    def test_unload_accepts_successful_stall_without_claiming_full_open(self):
        c=self.controller();c.release_hand_step(.0049,unload=True)
        goal=c.hand.send_goal_async.call_args.args[0];serialize_message(goal)
        self.assertAlmostEqual(goal.command.position,.0049);self.assertEqual(goal.command.max_effort,20.)
        self.assertEqual(c.result['release_commands'][-1]['stage'],'unload');c.wait_sim.assert_called_once_with(1.)
    def test_unload_also_accepts_reached_goal_when_handle_is_already_clear(self):
        c=self.controller(reached=True,stalled=False);c.release_hand_step(.008,unload=True)
    def test_full_open_still_requires_reached_goal(self):
        c=self.controller()
        with self.assertRaisesRegex(RuntimeError,'complete release'):c.release_hand_step(.04)
        c.wait_sim.assert_not_called()
    def test_full_open_success_uses_original_settle_interval(self):
        c=self.controller(reached=True,stalled=False);c.release_hand_step(.04);c.wait_sim.assert_called_once_with(.15)
    def test_aborted_and_canceled_unload_goals_are_not_accepted(self):
        for status in (GoalStatus.STATUS_ABORTED,GoalStatus.STATUS_CANCELED):
            c=self.controller(status=status)
            with self.assertRaises(RuntimeError):c.release_hand_step(.0049,unload=True)
    def test_incomplete_unload_result_is_not_accepted(self):
        c=self.controller(reached=False,stalled=False)
        with self.assertRaises(RuntimeError):c.release_hand_step(.0049,unload=True)
    def test_wrong_owner_expired_lease_and_disabled_support_block_release(self):
        for change in ('owner','busy','lease','input','phase'):
            c=self.controller()
            if change=='owner':c.owner['owner']='HOLD'
            elif change=='busy':c.owner['busy']=True
            elif change=='lease':c.input_deadline=c.now_ns()
            elif change=='input':c.input_enabled=False
            else:c.phase='lower'
            with self.assertRaises(RuntimeError):c.release_hand_step(.0049,unload=True)
            c.hand.send_goal_async.assert_not_called()

class RestingModelTests(unittest.TestCase):
    def world(self):
        world=np.eye(4);world[:3,3]=C['bins']['bin_a']['center'];world[2,3]=C['release_root_z_m'];return world
    def test_known_resting_prior_updates_model_without_changing_input(self):
        world=self.world();old=world.copy();out=resting_pose_from_release(world,'bin_a',C)
        np.testing.assert_array_equal(world,old);self.assertAlmostEqual(out[2,3],.3059)
        np.testing.assert_array_equal(out[:2,3],world[:2,3])
    def test_prior_preserves_yaw_and_aligns_with_known_floor(self):
        world=self.world();yaw=.05;pitch=.005
        from scipy.spatial.transform import Rotation
        world[:3,:3]=Rotation.from_euler('zy',[yaw,pitch]).as_matrix()
        out=resting_pose_from_release(world,'bin_a',C);np.testing.assert_array_equal(out[:,2],[0,0,1,0])
        self.assertAlmostEqual(np.arctan2(out[1,0],out[0,0]),np.arctan2(world[1,0],world[0,0]))
    def test_below_floor_excessive_height_and_wrong_target_are_rejected(self):
        for z in (.305,.32,.307):
            world=self.world();world[2,3]=z
            with self.assertRaises(RuntimeError):resting_pose_from_release(world,'bin_a',C)
    def test_wrong_bin_footprint_is_rejected(self):
        with self.assertRaises(RuntimeError):resting_pose_from_release(self.world(),'bin_b',C)
    def test_nonfinite_and_invalid_matrix_are_rejected(self):
        for world in (np.eye(3),np.full((4,4),np.nan)):
            with self.assertRaises(RuntimeError):resting_pose_from_release(world,'bin_a',C)
    def scene(self):
        node=SortScene.__new__(SortScene);node.episode=2;node.owner=dict(epoch=7,owner='SERVO',busy=False)
        node.sort_config=C;node.relative=np.eye(4);node.support_cleared=True;node.floor_id='bin_a_floor'
        node.q=dict(panda_finger_joint1=.04,panda_finger_joint2=.04);node.publisher=Mock();node.tick=Mock()
        node.buffer=Mock();node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=1_000_000_000));return node
    def test_detach_requires_open_fingers_and_authorized_floor(self):
        for mutation in ('fingers','floor'):
            obj=self.scene()
            if mutation=='fingers':obj.q['panda_finger_joint1']=.01
            else:obj.floor_id=''
            request=SetCameraPlacement.Request(episode_id=2,owner_epoch=7,bin_id='bin_a',operation='detach')
            with patch('camera_sort_scene.hand_transform',return_value=(self.world(),1_000_000_000)):
                response=obj.placement(request,SetCameraPlacement.Response())
            self.assertFalse(response.success);obj.publisher.publish.assert_not_called()
    def test_detach_changes_only_moveit_model_to_resting_prior(self):
        obj=self.scene();request=SetCameraPlacement.Request(episode_id=2,owner_epoch=7,bin_id='bin_a',operation='detach')
        with patch('camera_sort_scene.hand_transform',return_value=(self.world(),1_000_000_000)):
            response=obj.placement(request,SetCameraPlacement.Response())
        self.assertTrue(response.success,response.reason);self.assertAlmostEqual(obj.placed[2,3],.3059)
        self.assertIsNone(obj.relative);message=obj.publisher.publish.call_args.args[0]
        self.assertEqual(len(message.robot_state.attached_collision_objects),8);serialize_message(message)

if __name__=='__main__':unittest.main()
