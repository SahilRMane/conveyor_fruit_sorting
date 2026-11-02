"""M3D timing/geometry regressions; no live ROS nodes."""
from pathlib import Path
import ast,copy,json,sys,time,unittest
import numpy as np,yaml
from rclpy.serialization import serialize_message,deserialize_message
from moveit_msgs.msg import PlanningScene,CollisionObject
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/sorting_control/scripts'),str(R/'src/sorting_perception/scripts')]
from camera_scene import lightweight_camera_update,scene_for_camera,TRAY_IDS
from publish_static_fixtures import scene_message
from camera_grasp_core import predict_belief
from sorting_interfaces.msg import TargetBelief
from m2b_common import timestamp

class TimingTests(unittest.TestCase):
    def setUp(self):
        self.c=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text());self.template=scene_message()
    def old_update(self,target,allow,permissions=False):
        s=scene_for_camera(self.template,target,self.c,allow)
        s.world.collision_objects=[o for o in s.world.collision_objects if o.id in TRAY_IDS]
        for o in s.world.collision_objects:o.operation=CollisionObject.MOVE;o.primitives=[]
        if not permissions:
            from moveit_msgs.msg import AllowedCollisionMatrix
            s.allowed_collision_matrix=AllowedCollisionMatrix()
        return s
    def test_fast_move_preserves_the_complete_previous_message(self):
        for target in ([.55,0,0,0],[.73,.09,.03,0],[.48,-.1,-.03,0]):
            for allow in (False,True):
                with self.subTest(target=target,allow=allow):
                    old=self.old_update(target,allow);new=lightweight_camera_update(self.template,target,self.c,allow)
                    self.assertEqual(deserialize_message(serialize_message(new),PlanningScene),old)
    def test_permission_changes_still_include_the_exact_collision_matrix(self):
        for allow in (False,True):
            old=self.old_update([.6,0,0,0],allow,True)
            new=lightweight_camera_update(self.template,[.6,0,0,0],self.c,allow,True)
            self.assertEqual(deserialize_message(serialize_message(new),PlanningScene),old)
            self.assertTrue(new.allowed_collision_matrix.entry_names)
    def test_fast_path_never_copies_static_objects_or_rebuilds_a_matrix(self):
        from unittest.mock import patch
        with patch('camera_scene.scene_for_camera',side_effect=AssertionError('full rebuild')):
            m=lightweight_camera_update(self.template,[.6,0,0,0],self.c,True)
        self.assertEqual({o.id for o in m.world.collision_objects},TRAY_IDS)
        self.assertEqual(m.allowed_collision_matrix.entry_names,[])
        self.assertTrue(all(o.operation==CollisionObject.MOVE and not o.primitives for o in m.world.collision_objects))
    def test_template_is_not_modified_and_messages_serialize(self):
        before=copy.deepcopy(self.template)
        m=lightweight_camera_update(self.template,[.6,0,0,0],self.c,True)
        deserialize_message(serialize_message(m),PlanningScene)
        self.assertEqual(before,self.template)
    def test_new_m3d_profile_changes_only_rate(self):
        old=yaml.safe_load((R/'src/sorting_perception/config/cameras_m3c.yaml').read_text())
        new=yaml.safe_load((R/'src/sorting_perception/config/cameras_m3d.yaml').read_text())
        self.assertEqual(new.pop('publish_rate_hz'),20.);self.assertEqual(old.pop('publish_rate_hz'),15.)
        self.assertEqual(old,new)
    def test_new_plugin_changes_only_m3d_publish_rate(self):
        old=yaml.safe_load((R/'src/sorting_simulation/config/camera_plugins_m3b.yaml').read_text())
        new=yaml.safe_load((R/'src/sorting_simulation/config/camera_plugins_m3d.yaml').read_text())
        for d in (old,new):d['/**']['ros__parameters']['mujoco_plugins']['camera_plugin'].pop('camera_publish_rate')
        self.assertEqual(old,new)
    def test_launch_uses_new_camera_and_plugin_profiles(self):
        text=(R/'src/sorting_control/launch/panda_m3d.launch.py').read_text()
        self.assertIn('camera_plugins_m3d.yaml',text);self.assertIn('cameras_m3d.yaml',text)
        self.assertNotIn('camera_plugins_m3b.yaml',text)
    def test_manifest_tracks_the_camera_producer_and_processing_sources(self):
        text=(R/'src/sorting_perception/scripts/m3d_manifest.py').read_text()
        for name in ('camera_plugins_m3d.yaml','cameras_m3d.yaml','handle_localizer.py','motion_filter.py','camera_qos.py'):
            self.assertIn(name,text)
    def test_stale_moving_belief_still_fails_at_150ms(self):
        b=TargetBelief(valid=True,camera_only=True,state=[.6,0,.03,0],covariance=(np.eye(4)*1e-7).reshape(-1).tolist())
        b.header.frame_id='world';b.header.stamp=timestamp(1_000_000_000)
        predict_belief(b,1_149_999_999,self.c)
        with self.assertRaisesRegex(ValueError,'invalid, stale or future'):predict_belief(b,1_150_000_000,self.c)
    def test_scene_reports_capture_age_and_capture_identity(self):
        text=(R/'src/sorting_control/scripts/camera_sort_scene.py').read_text()
        self.assertIn('belief_capture_ns=',text);self.assertIn('capture_age_s=',text)
    def test_live_failure_fixture_contains_stale_controller_and_scene_samples(self):
        rows=json.loads((R/'tests/fixtures/m3d-camera-age-failure.json').read_text())
        self.assertEqual(rows[0]['trial'],3);self.assertEqual(rows[0]['belief_age_s'],.15)
        self.assertAlmostEqual(rows[1]['scene_model_age_s'],.156)
        self.assertAlmostEqual(rows[1]['belief_age_s'],.072)
        self.assertFalse(rows[1]['scene_valid'])

if __name__=='__main__':unittest.main()
