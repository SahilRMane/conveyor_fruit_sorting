from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import time
import numpy as np
import yaml
from rclpy.serialization import serialize_message,deserialize_message
from sorting_interfaces.msg import TargetBelief,TargetBeliefArray
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/sorting_perception/scripts'),str(R/'src/sorting_control/scripts')]
from filter_core import MotionTrack,forecast
from camera_grasp_core import CameraClosureWindow,FingerWindow,predict_belief
from camera_scene import scene_for_camera,TRAY_IDS
from publish_static_fixtures import scene_message
from camera_pickup import Controller
from m3b_evaluator import Evaluator

class FilterTests(unittest.TestCase):
    def setUp(self):self.c=yaml.safe_load((R/'src/sorting_perception/config/m3b_filter.yaml').read_text());self.t=MotionTrack(self.c);self.t.reset(3,100)
    def feed(self,count=30,speed=.03):
        for i in range(count):
            stamp=1_000_000_000+i*100_000_000
            self.assertTrue(self.t.update(3,stamp,[.55+speed*i*.1,0.],np.eye(2).reshape(-1)*1e-7,'wrist',1.,stamp),self.t.reason)
    def test_capture_time_velocity_and_covariance(self):
        self.feed();self.assertAlmostEqual(self.t.x[2],.03,places=4);self.assertTrue(self.t.ready(self.t.stamp_ns));self.assertGreaterEqual(min(np.linalg.eigvalsh(self.t.p)),0)
        x,p=forecast(self.t.x,self.t.p,.1,self.c);self.assertAlmostEqual(x[0],self.t.x[0]+.003,places=4);self.assertGreater(p[0,0],self.t.p[0,0])
    def test_duplicate_does_not_reduce_covariance(self):
        self.feed(6);p=self.t.p.copy();self.assertFalse(self.t.update(3,self.t.stamp_ns,[.55,0.],np.eye(2)*1e-7,'overhead',1.,self.t.stamp_ns));np.testing.assert_array_equal(p,self.t.p)
    def test_initialization_requires_distinct_time_span(self):
        self.feed(5,0.);self.assertFalse(self.t.ready(self.t.stamp_ns));self.assertEqual(self.t.samples,5)
    def test_innovation_rejects_far_jump(self):
        self.feed(10);old=self.t.x.copy();stamp=self.t.stamp_ns+100_000_000
        self.assertFalse(self.t.update(3,stamp,[.74,.14],np.eye(2)*1e-7,'wrist',1.,stamp));np.testing.assert_array_equal(old,self.t.x)
    def test_speed_step_is_admitted_without_command_velocity_input(self):
        self.feed(15,0.)
        for i in range(1,20):
            stamp=2_400_000_000+i*100_000_000
            self.assertTrue(self.t.update(3,stamp,[.55+.003*i,0.],np.eye(2)*1e-7,'wrist',1.,stamp),self.t.reason)
        self.assertAlmostEqual(self.t.x[2],.03,places=3)
    def test_timeout_reacquires_with_new_identity(self):
        self.feed(6);old=self.t.track_id;self.t.expire(self.t.stamp_ns+400_000_000);self.assertIsNone(self.t.x)
        stamp=3_000_000_000;self.assertTrue(self.t.update(3,stamp,[.6,0.],np.eye(2)*1e-7,'wrist',1.,stamp));self.assertGreater(self.t.track_id,old);self.assertFalse(self.t.ready(stamp))
    def test_reset_rejects_old_episode_and_prebarrier_capture(self):
        self.feed(6);self.assertTrue(self.t.reset(4,4_000_000_000));self.assertFalse(self.t.reset(3,0))
        for ep,stamp in [(3,4_100_000_000),(4,3_999_999_999)]:self.assertFalse(self.t.update(ep,stamp,[.55,0.],np.eye(2)*1e-7,'wrist',1.,4_100_000_000))
    def test_stale_and_nonfinite_observations(self):
        for value in [(.55,float('nan')),(.8,0.)]:self.assertFalse(self.t.update(3,1_000_000_000,value,np.eye(2)*1e-7,'wrist',1.,1_000_000_000))
        self.assertFalse(self.t.update(3,1_000_000_000,[.55,0.],np.eye(2)*1e-7,'wrist',1.,1_200_000_000))
    def test_real_message_roundtrip_preserves_capture_epoch_and_covariance(self):
        self.feed(8);b=TargetBelief(valid=True,camera_only=True,episode_id=3,track_id=self.t.track_id,state=self.t.x.tolist(),covariance=self.t.p.reshape(-1).tolist())
        b.header.frame_id='world';b.header.stamp.sec=1;b.header.stamp.nanosec=700_000_000
        a=TargetBeliefArray(targets=[b]);copy=deserialize_message(serialize_message(a),TargetBeliefArray).targets[0]
        x,p,stamp=predict_belief(copy,1_800_000_000,self.c);self.assertEqual(stamp,1_700_000_000);self.assertEqual(copy.episode_id,3);self.assertAlmostEqual(x[0],self.t.x[0]+.1*self.t.x[2])
        with self.assertRaises(ValueError):predict_belief(copy,2_000_000_000,self.c)

class ClosureTests(unittest.TestCase):
    def setUp(self):self.c=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text());self.w=CameraClosureWindow(self.c)
    def sample(self,t,capture=None):return dict(position_error_m=[.0001,0.,.0001],relative_velocity_m_s=[.001,0,0],predicted_covariance=np.diag([1e-7,1e-7,3.6e-5,3.6e-5]),orientation_error_rad=.001,capture_age_s=.01,tf_age_s=.01,remaining_s=5.,descent_complete=True,tf_stamp_ns=t,capture_ns=capture if capture is not None else t//100_000_000*100_000_000)
    def test_good_proof_requires_span_and_multiple_captures(self):
        for t in range(1_000_000_000,1_200_000_000,20_000_000):self.assertFalse(self.w.observe(self.sample(t)))
        self.assertTrue(self.w.observe(self.sample(1_200_000_000)))
    def test_one_capture_cannot_authorize_close(self):
        for t in range(1_000_000_000,1_500_000_000,20_000_000):self.assertFalse(self.w.observe(self.sample(t,1_000_000_000)))
    def test_excess_uncertainty_speed_staleness_deadline_blocks(self):
        cases=[('predicted_covariance',np.diag([4e-6,4e-6,3.6e-5,3.6e-5])),('predicted_covariance',-np.eye(4)),('relative_velocity_m_s',[.006,0,0]),('capture_age_s',.2),('remaining_s',2.),('descent_complete',False),('orientation_error_rad',.02),('orientation_error_rad',float('nan'))]
        for key,value in cases:
            s=self.sample(1_000_000_000);s[key]=value;self.assertFalse(CameraClosureWindow(self.c).observe(s))
    def test_duplicate_tf_does_not_count(self):
        for _ in range(30):self.assertFalse(self.w.observe(self.sample(1_000_000_000)))
    def test_encoder_candidate_rejects_empty_or_open_gripper(self):
        for gap in [0.,.04]:
            f=FingerWindow(self.c)
            for t in range(1_000_000_000,1_500_000_000,20_000_000):self.assertFalse(f.observe(t,[gap,gap],[0,0]))
    def test_candidate_requires_stopped_gap_and_time(self):
        f=FingerWindow(self.c)
        for t in range(1_000_000_000,1_200_000_000,20_000_000):self.assertFalse(f.observe(t,[.01,.01],[0,0]))
        self.assertTrue(f.observe(1_200_000_000,[.01,.01],[0,0]))

class IsolationTests(unittest.TestCase):
    def test_constructor_creates_no_truth_subscriptions(self):
        node=Mock()
        with patch('camera_pickup.Buffer'),patch('camera_pickup.TransformListener'),patch('camera_pickup.ActionClient'):
            controller=Controller(node)
        topics=[call.args[1] for call in node.create_subscription.call_args_list]
        self.assertTrue('/sorting/targets' in topics);self.assertFalse(any('ground_truth' in t or t in {'/sorting/contact_state','/sorting/reset_state'} for t in topics))
        pubs=[call.args[1] for call in node.create_publisher.call_args_list];self.assertEqual(pubs.count('/sorting/arm/servo_command'),1)
        controller.epoch=NS(episode_id=3);controller.now_ns=lambda:1_010_000_000;controller.received=time.monotonic()
        b=TargetBelief(episode_id=3,track_id=7,valid=True,camera_only=True,fixed_handle_z=.4229,state=[.55,0.,0.,0.],covariance=(np.eye(4)*1e-7).reshape(-1).tolist())
        b.header.frame_id='world';b.header.stamp.sec=1;controller.belief=b;controller.bound_track=6
        with self.assertRaises(RuntimeError):controller.target()
        controller.bound_track=None;self.assertAlmostEqual(controller.target()[0][0],.55)
    def test_evaluator_creates_no_command_publishers(self):
        node=Mock()
        with patch('m3b_evaluator.ResetManager'):Evaluator(node)
        node.create_publisher.assert_not_called()
    def test_scene_moves_to_camera_position_and_estimated_attachment_is_only_moveit(self):
        c=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text());template=scene_message()
        scene=scene_for_camera(template,[.62,.01,0,0],c,False);tray=next(o for o in scene.world.collision_objects if o.id=='tray_base')
        self.assertAlmostEqual(tray.primitive_poses[0].position.x,.62)
        relative=np.eye(4);relative[:3,3]=[0.,0.,.22];scene=scene_for_camera(template,[.62,.01,0,0],c,True,relative)
        self.assertEqual({a.object.id for a in scene.robot_state.attached_collision_objects},TRAY_IDS)
        self.assertFalse(TRAY_IDS.intersection(o.id for o in scene.world.collision_objects))
        self.assertTrue(all(a.object.header.frame_id=='panda_hand' for a in scene.robot_state.attached_collision_objects))
        for a in scene.robot_state.attached_collision_objects:self.assertEqual(set(a.touch_links),{'panda_leftfinger','panda_rightfinger'} if a.object.id=='handle_bar' else set())
        serialize_message(scene)
    def test_launch_has_no_oracle_and_defaults_visible(self):
        s=(R/'src/sorting_control/launch/panda_m3b.launch.py').read_text();self.assertNotIn('oracle_predictor',s);self.assertNotIn('oracle_target_source',s)
        self.assertIn('"headless", default_value="false"',s);self.assertIn('"allow_trajectory_execution": False',s)



class ViewSelectionTests(unittest.TestCase):
    def fixture(self):
        from motion_filter import MotionFilter
        from builtin_interfaces.msg import Time
        o=MotionFilter.__new__(MotionFilter);o.c=yaml.safe_load((R/'src/sorting_perception/config/m3b_filter.yaml').read_text());o.track=MotionTrack(o.c);o.track.reset(2,0)
        o.pending={};o.held=False;o.last_wall=0.;o.last_now=0;o.rewind_episode=None;o.publisher=Mock();o.status=Mock()
        o.get_clock=Mock();o.get_clock.return_value.now.return_value=NS(nanoseconds=1_010_000_000,to_msg=lambda:Time(sec=1,nanosec=10_000_000))
        return o
    def observation(self,view):
        from sorting_interfaces.msg import HandleObservation
        m=HandleObservation(episode_id=2,view_id=view,valid=True,quality=1.,planar_covariance=[1e-7,0.,0.,1e-7]);m.header.stamp.sec=1;m.position.x=.55;return m
    def test_correlated_views_at_one_stamp_update_only_once(self):
        from motion_filter import MotionFilter
        o=self.fixture();MotionFilter.receive(o,self.observation('overhead'));MotionFilter.receive(o,self.observation('wrist'));MotionFilter.tick(o)
        self.assertEqual(o.track.samples,1);self.assertEqual(o.track.accepted,1);MotionFilter.tick(o);self.assertEqual(o.track.accepted,1)
    def test_held_candidate_stops_supported_plane_updates(self):
        from motion_filter import MotionFilter
        from std_msgs.msg import String
        o=self.fixture();MotionFilter.control_state(o,String(data='{"episode_id":2,"grip_candidate":true}'));MotionFilter.receive(o,self.observation('wrist'))
        self.assertTrue(o.held);self.assertEqual(o.pending,{})
    def test_clock_rewind_requires_new_episode(self):
        from motion_filter import MotionFilter
        o=self.fixture();o.last_now=2_000_000_000;MotionFilter.tick(o);MotionFilter.receive(o,self.observation('wrist'))
        self.assertEqual(o.rewind_episode,2);self.assertEqual(o.pending,{})

class AcceptanceTests(unittest.TestCase):
    def assess(self,trials,stopped=False):
        from m3b_acceptance import accepted_batch
        return accepted_batch(trials,20,16,'PASS','HOLD',0,stopped)
    def batch(self,successes):return [dict(status='PASS' if i<successes else 'FAIL',reset=dict(status='PASS')) for i in range(20)]
    def test_sixteen_grasps_and_all_twenty_resets_pass(self):self.assertTrue(self.assess(self.batch(16)))
    def test_fifteen_grasps_fail(self):self.assertFalse(self.assess(self.batch(15)))
    def test_failed_last_trial_cannot_reuse_previous_reset_pass(self):
        t=self.batch(19);t[-1].pop('reset');self.assertFalse(self.assess(t))
    def test_constraint_stop_and_aborted_trials_fail(self):
        self.assertFalse(self.assess(self.batch(20),True));t=self.batch(20);t[-1]=dict(status='ABORTED');self.assertFalse(self.assess(t))

class TimingResetTests(unittest.TestCase):
    def test_one_camera_is_processed_without_the_old_40ms_wait(self):
        from motion_filter import MotionFilter
        o=ViewSelectionTests().fixture();m=ViewSelectionTests().observation('wrist')
        with patch('motion_filter.time.monotonic',return_value=1.):
            MotionFilter.receive(o,m);MotionFilter.tick(o)
        self.assertEqual(o.track.accepted,1)
    def test_no_republication_can_refresh_a_capture(self):
        from motion_filter import MotionFilter
        o=ViewSelectionTests().fixture();MotionFilter.receive(o,ViewSelectionTests().observation('wrist'));MotionFilter.tick(o)
        MotionFilter.tick(o);self.assertEqual(o.publisher.publish.call_count,1)
    def test_moveit_implicit_disallowed_pairs_pass_reset(self):
        from camera_scene_checks import reset_scene_diagnostic
        from moveit_msgs.msg import AllowedCollisionMatrix
        scene=scene_message();scene.allowed_collision_matrix=AllowedCollisionMatrix()
        self.assertTrue(reset_scene_diagnostic(scene,TRAY_IDS)['valid'])
    def test_allowed_defaults_or_attached_payload_fail_reset(self):
        from camera_scene_checks import reset_scene_diagnostic
        from moveit_msgs.msg import AllowedCollisionMatrix,AttachedCollisionObject
        scene=scene_message();scene.allowed_collision_matrix=AllowedCollisionMatrix(default_entry_names=['handle_bar'],default_entry_values=[True])
        self.assertFalse(reset_scene_diagnostic(scene,TRAY_IDS)['valid'])
        scene.allowed_collision_matrix=AllowedCollisionMatrix();a=AttachedCollisionObject();a.object.id='tray_base';scene.robot_state.attached_collision_objects=[a]
        self.assertFalse(reset_scene_diagnostic(scene,TRAY_IDS)['valid'])
    def test_explicit_allowed_pair_fails_reset(self):
        from camera_scene_checks import reset_scene_diagnostic
        scene=scene_for_camera(scene_message(),[.55,0,0,0],dict(fixed_handle_z_m=.4229),True)
        self.assertFalse(reset_scene_diagnostic(scene,TRAY_IDS)['valid'])
    def test_wait_for_valid_scene_before_planning_does_not_send_a_command(self):
        c=Controller.__new__(Controller);c.input_enabled=False;c.target_required=True;c.scene=dict(valid=False,camera_only=True,time_ns=1_000_000_000)
        c.now_ns=lambda:1_000_000_000;c.target=Mock();c.publisher=Mock()
        def until(predicate,timeout):
            self.assertFalse(predicate());c.scene['valid']=True;self.assertTrue(predicate())
        c.until=until;c.ensure_oracle_current();c.publisher.publish.assert_not_called()

class PregraspHandoffTests(unittest.TestCase):
    def test_reported_154ms_capture_waits_without_sending(self):
        c=Controller.__new__(Controller);c.input_enabled=False;c.belt_speed=0.;c.bound_track=None
        c.c=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text());c.fc=c.c;c.epoch=NS(episode_id=0);c.received=time.monotonic()
        b=TargetBelief(episode_id=0,track_id=1,valid=True,camera_only=True,fixed_handle_z=.4229,state=[.55,0.,0.,0.],covariance=(np.eye(4)*1e-7).reshape(-1).tolist())
        b.header.frame_id='world';b.header.stamp.sec=14;b.header.stamp.nanosec=114_000_000;c.belief=b;c.now_ns=lambda:14_268_000_000
        c.scene=dict(valid=True,camera_only=True,time_ns=14_260_000_000,model_capture_ns=14_114_000_000);c.publisher=Mock()
        def until(predicate,timeout):
            self.assertFalse(predicate());self.assertFalse(c.input_enabled)
            b.header.stamp.nanosec=240_000_000;c.scene['model_capture_ns']=14_240_000_000
            self.assertTrue(predicate())
        c.until=until;c.wait_refinement_ready();c.publisher.publish.assert_not_called()
    def test_refinement_wait_rejects_a_moving_belt(self):
        c=Controller.__new__(Controller);c.input_enabled=False;c.belt_speed=.03
        with self.assertRaises(RuntimeError):c.wait_refinement_ready()
    def test_overhead_does_not_replace_the_selected_wrist_stream(self):
        from motion_filter import MotionFilter
        o=ViewSelectionTests().fixture();MotionFilter.receive(o,ViewSelectionTests().observation('overhead'));MotionFilter.tick(o)
        self.assertEqual(o.track.accepted,0)
        MotionFilter.receive(o,ViewSelectionTests().observation('wrist'));MotionFilter.tick(o);self.assertEqual(o.track.accepted,1)
    def test_lightweight_updates_reuse_shapes_and_preserve_fixture_allowances(self):
        from camera_scene import lightweight_camera_update
        from moveit_msgs.msg import CollisionObject
        c=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text());m=lightweight_camera_update(scene_message(),[.62,0,0,0],c,False)
        self.assertEqual({o.id for o in m.world.collision_objects},TRAY_IDS)
        self.assertTrue(all(o.operation==CollisionObject.MOVE and not o.primitives and o.primitive_poses for o in m.world.collision_objects))
        self.assertFalse(m.allowed_collision_matrix.entry_names);serialize_message(m)
    def test_vectorized_contour_has_the_same_world_centroid_as_scalar_projection(self):
        from camera_core import localize,ray_plane,quaternion_rotation
        import cv2
        folder=R/'tests/fixtures/m3b-contour';data=yaml.safe_load((folder/'native-render.yaml').read_text())
        c=yaml.safe_load((R/'src/sorting_perception/config/m3b_localization.yaml').read_text())
        f=next(f for f in data['frames'] if f['label']=='wrist-inspection');rgb=cv2.cvtColor(cv2.imread(str(folder/f['image'])),cv2.COLOR_BGR2RGB)
        focal=f['height']/2/np.tan(np.deg2rad(f['fovy_deg'])/2);k=np.array([[focal,0,f['width']/2],[0,focal,f['height']/2],[0,0,1.]])
        rotation=quaternion_rotation(f['camera_quat_xyzw']);result=localize(rgb,k,f['camera_position'],rotation,c)
        mask=cv2.inRange(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV),np.array(c['hsv_lower'],np.uint8),np.array(c['hsv_upper'],np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);contour=max(contours,key=cv2.contourArea)
        projected=np.array([ray_plane(p[0]+c['pixel_center_offset'],k,f['camera_position'],rotation,c)[:2] for p in contour],np.float32)
        moments=cv2.moments(projected.reshape(-1,1,2));expected=[moments['m10']/moments['m00'],moments['m01']/moments['m00']]
        np.testing.assert_allclose(result['position'][:2],expected,atol=1e-10)

class CoherentClosureTests(unittest.TestCase):
    def config(self):return yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text())
    def test_recorded_tracking_snapshots_reach_closure_before_the_exit(self):
        import json
        from camera_grasp_core import ClosureProof
        samples=json.loads((R/'tests/fixtures/m3b-closure-live.json').read_text());proof=ClosureProof(self.config());authorized=[]
        for sample in samples:
            if proof.observe(sample) and proof.authorized(sample['time_ns']):authorized.append(sample)
        self.assertTrue(authorized);self.assertAlmostEqual(authorized[0]['time_ns']/1e9,15.948,places=3)
        self.assertGreater(authorized[0]['remaining_s'],2.5)
    def test_duplicate_snapshot_is_not_rescored_at_a_later_clock_time(self):
        import json
        from camera_grasp_core import ClosureProof
        proof=ClosureProof(self.config());samples=json.loads((R/'tests/fixtures/m3b-closure-live.json').read_text())
        for sample in samples:
            proof.observe(sample)
            if proof.ready:break
        count=proof.window.count;proof.observe(sample);self.assertEqual(count,proof.window.count)
        self.assertTrue(proof.authorized(sample['time_ns']+2_000_000))
        self.assertFalse(proof.authorized(sample['time_ns']+25_000_000))
    def test_stale_capture_and_short_deadline_cannot_reuse_authorization(self):
        import json
        from camera_grasp_core import ClosureProof
        proof=ClosureProof(self.config());samples=json.loads((R/'tests/fixtures/m3b-closure-live.json').read_text())
        for sample in samples:
            proof.observe(sample)
            if proof.ready:break
        proof.sample['capture_ns']=proof.last_time-150_000_000;self.assertFalse(proof.authorized(proof.last_time))
        proof.sample['capture_ns']=proof.last_time;proof.sample['remaining_s']=2.5;self.assertFalse(proof.authorized(proof.last_time))

class ClearLiftSupportTests(unittest.TestCase):
    def setUp(self):
        self.c=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text())
        self.owner=dict(owner='SERVO',busy=False)
    def state(self,z=.3059,now=1_100_000_000,cleared=False,owner=None):
        from camera_scene import support_contact_state
        world=np.eye(4);world[:3,3]=[.65,0.,z]
        return support_contact_state(world,now,1_000_000_000,cleared,self.owner if owner is None else owner,self.c)
    def test_only_lower_payload_support_pairs_change(self):
        from camera_scene_checks import collision_allowed
        template=scene_message()
        blocked=scene_for_camera(template,[.65,0,0,0],self.c,True,np.eye(4))
        allowed=scene_for_camera(template,[.65,0,0,0],self.c,True,np.eye(4),True)
        changed=set()
        for a in allowed.allowed_collision_matrix.entry_names:
            for b in allowed.allowed_collision_matrix.entry_names:
                if collision_allowed(blocked.allowed_collision_matrix,a,b)!=collision_allowed(allowed.allowed_collision_matrix,a,b):changed.add((a,b))
        from camera_scene import SUPPORT_CONTACT_IDS
        self.assertEqual(changed,{(part,'support_surface') for part in SUPPORT_CONTACT_IDS}|{('support_surface',part) for part in SUPPORT_CONTACT_IDS})
        self.assertFalse(collision_allowed(allowed.allowed_collision_matrix,'panda_hand','support_surface'))
        serialize_message(allowed)
    def test_world_payload_cannot_get_clear_lift_permission(self):
        with self.assertRaises(RuntimeError):scene_for_camera(scene_message(),[.65,0,0,0],self.c,True,allow_support_contact=True)
    def test_expected_contact_allowed_then_revoked_and_latched(self):
        gap,cleared,allowed=self.state();self.assertAlmostEqual(gap,-.0001);self.assertFalse(cleared);self.assertTrue(allowed)
        gap,cleared,allowed=self.state(z=.3311);self.assertTrue(cleared);self.assertFalse(allowed)
        self.assertFalse(self.state(z=.3059,cleared=True)[2])
    def test_deadline_owner_switch_and_clock_rewind_revoke_permission(self):
        self.assertFalse(self.state(now=5_000_000_000)[2])
        self.assertFalse(self.state(now=999_999_999)[2])
        self.assertFalse(self.state(owner=dict(owner='HOLD',busy=False))[2])
        self.assertFalse(self.state(owner=dict(owner='SERVO',busy=True))[2])
    def test_excess_penetration_or_nonfinite_transform_rejected(self):
        with self.assertRaises(RuntimeError):self.state(z=.3029)
        with self.assertRaises(RuntimeError):self.state(z=float('nan'))
    def test_reset_checks_support_permission(self):
        from camera_scene_checks import reset_scene_diagnostic
        scene=scene_for_camera(scene_message(),[.65,0,0,0],self.c,False,np.eye(4),True)
        # Make the remaining scene otherwise match reset, while retaining the leaked pair.
        scene.robot_state.attached_collision_objects=[];scene.world=scene_message().world
        report=reset_scene_diagnostic(scene,TRAY_IDS)
        self.assertTrue(report['tray_support_permission']);self.assertFalse(report['valid'])
    def test_stale_hand_tf_revokes_published_permission(self):
        from camera_scene import CameraScene
        o=CameraScene.__new__(CameraScene)
        o.get_clock=lambda:NS(now=lambda:NS(nanoseconds=1_100_000_000))
        o.relative=np.eye(4);o.relative[2,3]=.3059;o.template=scene_message();o.c=self.c
        o.allow=True;o.last_support_sent=True;o.support_contact=True;o.support_cleared=False;o.support_gap=-.0001
        o.buffer=Mock();o.publisher=Mock();o.state=Mock();o.episode=2;o.belief=TargetBelief(track_id=3)
        o.belief.header.stamp.sec=1;o.model_capture_ns=1_000_000_000;o.model_time_ns=1_000_000_000
        with patch('camera_scene.hand_transform',side_effect=RuntimeError('stale TF')):o.tick()
        update=o.publisher.publish.call_args.args[0]
        from camera_scene_checks import collision_allowed
        self.assertFalse(collision_allowed(update.allowed_collision_matrix,'tray_base','support_surface'))
        self.assertEqual(update.world.collision_objects,[]);self.assertEqual(update.robot_state.attached_collision_objects,[])
        self.assertFalse(o.support_contact);self.assertFalse(o.last_support_sent)

class StationaryCameraWaitTests(unittest.TestCase):
    def fixture(self,phase='pregrasp_refinement'):
        node=Mock()
        with patch('camera_pickup.Buffer'),patch('camera_pickup.TransformListener'),patch('camera_pickup.ActionClient'):c=Controller(node)
        c.c=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text());c.fc=c.c
        c.now={'ns':1_154_000_000};c.now_ns=lambda:c.now['ns']
        from m2b_common import timestamp
        node.get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:timestamp(c.now_ns())))
        c.epoch=NS(episode_id=2);c.belief=TargetBelief(episode_id=2,track_id=7,valid=True,camera_only=True,
            fixed_handle_z=.4229,state=[.55,0.,0.,0.],covariance=(np.eye(4)*1e-7).reshape(-1).tolist())
        c.belief.header.frame_id='world';c.belief.header.stamp.sec=1;c.received=10.
        c.active=True;c.phase=phase;c.input_enabled=True;c.input_epoch=6;c.input_deadline=20_000_000_000
        c.owner=dict(owner='SERVO',busy=False,epoch=6);c.q=dict(panda_finger_joint1=.04,panda_finger_joint2=.04)
        c.scene=dict(valid=False,camera_only=True,reason='Camera belief is invalid, stale or future',capture_age_s=.154,
            time_ns=c.now_ns(),model_capture_ns=1_000_000_000,estimated_attachment=False)
        c.sample_hand=Mock(side_effect=lambda now:(np.array([.55,0.,.5858]),[0.,0.,0.,1.],now,0.,np.zeros(3)))
        c.command_function=Mock(return_value=([.551,0.,.5858],[0.,0.,0.],[0.,0.,0.,1.]))
        c.publisher=Mock()
        return c
    def fresh(self,c):
        c.now['ns']=1_174_000_000;c.belief.header.stamp.nanosec=100_000_000;c.received=10.02
        c.scene.update(valid=True,capture_age_s=.074,time_ns=c.now_ns(),model_capture_ns=1_100_000_000,reason='camera_geometry')
    def test_recorded_154ms_age_brakes_without_using_stale_reference(self):
        c=self.fixture()
        with patch('camera_pickup.time.monotonic',return_value=10.):c.send()
        c.command_function.assert_not_called();m=c.publisher.publish.call_args.args[0]
        self.assertEqual([m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z,
            m.twist.twist.angular.x,m.twist.twist.angular.y,m.twist.twist.angular.z],[0.]*6)
        self.assertEqual(m.epoch,6);self.assertEqual(c.camera_wait_trace[0]['capture_ns'],1_000_000_000)
    def test_fresh_capture_and_scene_resume_normal_reference(self):
        c=self.fixture()
        with patch('camera_pickup.time.monotonic',return_value=10.):c.send()
        self.fresh(c)
        with patch('camera_pickup.time.monotonic',return_value=10.02):c.send()
        self.assertIsNone(c.camera_wait);c.command_function.assert_called_once();self.assertGreater(c.publisher.publish.call_args.args[0].twist.twist.linear.x,0.)
        self.assertEqual(c.events[-1]['phase'],'stationary_camera_resume')
    def test_new_belief_alone_cannot_resume_without_matching_scene(self):
        c=self.fixture()
        with patch('camera_pickup.time.monotonic',return_value=10.):c.send()
        self.fresh(c);c.scene['model_capture_ns']=1_000_000_000
        with patch('camera_pickup.time.monotonic',return_value=10.02):c.send()
        c.command_function.assert_not_called();self.assertIsNotNone(c.camera_wait)
    def test_moving_tracking_and_closure_still_fail_on_stale_data(self):
        from camera_pickup import CameraCaptureWait
        for phase in ('track','close','lift'):
            c=self.fixture(phase)
            with patch('camera_pickup.time.monotonic',return_value=10.):
                with self.assertRaises(CameraCaptureWait):c.send()
            c.publisher.publish.assert_not_called()
    def test_belt_motion_closed_fingers_or_payload_block_recovery(self):
        for mutation in ('belt','fingers','payload'):
            c=self.fixture()
            if mutation=='belt':c.belt_speed=.03
            elif mutation=='fingers':c.q['panda_finger_joint1']=.01
            else:c.scene['estimated_attachment']=True
            with patch('camera_pickup.time.monotonic',return_value=10.):
                with self.assertRaises(RuntimeError):c.send()
            c.publisher.publish.assert_not_called()
    def test_late_fresh_data_cannot_bypass_wait_timeout(self):
        c=self.fixture()
        with patch('camera_pickup.time.monotonic',return_value=10.):c.send()
        self.fresh(c)
        with patch('camera_pickup.time.monotonic',return_value=10.13):
            with self.assertRaisesRegex(RuntimeError,'exceeded its bound'):c.send()
        c.command_function.assert_not_called()
    def test_sim_timeout_and_identity_change_stop_recovery(self):
        for mutation in ('sim','identity'):
            c=self.fixture()
            with patch('camera_pickup.time.monotonic',return_value=10.):c.send()
            self.fresh(c)
            if mutation=='sim':c.now['ns']=1_255_000_000;c.belief.header.stamp.nanosec=200_000_000;c.scene.update(time_ns=c.now_ns(),model_capture_ns=1_200_000_000)
            else:c.belief.track_id=8
            with patch('camera_pickup.time.monotonic',return_value=10.02):
                with self.assertRaises(RuntimeError):c.send()
            c.command_function.assert_not_called()
    def test_ownership_and_lease_changes_block_braking(self):
        for mutation in ('owner','lease'):
            c=self.fixture()
            if mutation=='owner':c.owner['epoch']=7
            else:c.input_deadline=c.now_ns()
            with patch('camera_pickup.time.monotonic',return_value=10.):
                with self.assertRaises(RuntimeError):c.send()
            c.publisher.publish.assert_not_called()
    def test_fresh_resume_rechecks_stationary_conditions(self):
        c=self.fixture()
        with patch('camera_pickup.time.monotonic',return_value=10.):c.send()
        self.fresh(c);c.belt_speed=.03
        with patch('camera_pickup.time.monotonic',return_value=10.02):
            with self.assertRaises(RuntimeError):c.send()
        c.command_function.assert_not_called()
    def test_wait_count_and_total_time_are_bounded(self):
        c=self.fixture();c.camera_wait_count=3
        with patch('camera_pickup.time.monotonic',return_value=10.):
            with self.assertRaisesRegex(RuntimeError,'Too many'):c.send()
        c.publisher.publish.assert_not_called()
        c=self.fixture();c.camera_wait_total_wall_s=.19
        with patch('camera_pickup.time.monotonic',return_value=10.):c.send()
        self.fresh(c)
        with patch('camera_pickup.time.monotonic',return_value=10.02):
            with self.assertRaisesRegex(RuntimeError,'exceeded its bound'):c.send()
        c.command_function.assert_not_called()

class WristRuntimeProfileTests(unittest.TestCase):
    def test_m3b_streams_its_selected_view_and_keeps_both_calibrated_models(self):
        cameras=yaml.safe_load((R/'src/sorting_perception/config/cameras_m3b.yaml').read_text())
        filt=yaml.safe_load((R/'src/sorting_perception/config/m3b_filter.yaml').read_text())
        plugins=yaml.safe_load((R/'src/sorting_simulation/config/camera_plugins_m3b.yaml').read_text())['/**']['ros__parameters']['mujoco_plugins']['camera_plugin']
        active=[view for view,cam in cameras['cameras'].items() if cam.get('publish_policy','streaming')=='streaming']
        self.assertEqual(active,[filt['tracking_view']]);self.assertEqual(active,['wrist'])
        self.assertEqual(set(cameras['cameras']),{'overhead','wrist'})
        for view,cam in cameras['cameras'].items():self.assertEqual(plugins[view]['policy'],cam['publish_policy'])
        self.assertEqual(cameras['render_quality']['offscreen_samples'],4)
        self.assertEqual(filt['maximum_capture_age_s'],.15)

class CompleteCarriedGeometryTests(unittest.TestCase):
    def setUp(self):self.c=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text())
    def test_internal_rigid_part_permissions_exist_only_when_carried(self):
        from camera_scene_checks import collision_allowed
        template=scene_message()
        world=scene_for_camera(template,[.65,0,0,0],self.c,True)
        carried=scene_for_camera(template,[.65,0,0,0],self.c,True,np.eye(4))
        for a in TRAY_IDS:
            for b in TRAY_IDS:
                if a!=b:
                    self.assertFalse(collision_allowed(world.allowed_collision_matrix,a,b))
                    self.assertTrue(collision_allowed(carried.allowed_collision_matrix,a,b))
            for link in ('panda_hand','panda_link5','panda_link7'):
                self.assertFalse(collision_allowed(carried.allowed_collision_matrix,a,link))
        self.assertEqual(len(carried.robot_state.attached_collision_objects),8)
        serialize_message(carried)
    def test_clearance_revokes_all_lower_support_pairs_and_preserves_rigid_assembly(self):
        from camera_scene import SUPPORT_CONTACT_IDS
        from camera_scene_checks import collision_allowed
        template=scene_message()
        contact=scene_for_camera(template,[.65,0,0,0],self.c,True,np.eye(4),True)
        clear=scene_for_camera(template,[.65,0,0,0],self.c,True,np.eye(4),False)
        for part in TRAY_IDS:
            self.assertEqual(collision_allowed(contact.allowed_collision_matrix,part,'support_surface'),part in SUPPORT_CONTACT_IDS)
            self.assertFalse(collision_allowed(clear.allowed_collision_matrix,part,'support_surface'))
        self.assertTrue(collision_allowed(clear.allowed_collision_matrix,'tray_base','tray_wall_x_plus'))
        self.assertFalse(collision_allowed(contact.allowed_collision_matrix,'handle_bar','support_surface'))
    def test_reset_detects_leaked_internal_permissions(self):
        from camera_scene_checks import reset_scene_diagnostic
        carried=scene_for_camera(scene_message(),[.65,0,0,0],self.c,False,np.eye(4))
        carried.robot_state.attached_collision_objects=[];carried.world=scene_message().world
        result=reset_scene_diagnostic(carried,TRAY_IDS)
        self.assertFalse(result['valid']);self.assertEqual(len(result['remaining_internal_tray_permissions']),28)
        self.assertFalse(result['tray_support_permission'])
    def test_reset_detects_a_wall_permission_without_base_permission(self):
        from camera_scene_checks import reset_scene_diagnostic
        world=scene_for_camera(scene_message(),[.65,0,0,0],self.c,False)
        names=world.allowed_collision_matrix.entry_names;a=names.index('tray_wall_x_plus');b=names.index('support_surface')
        world.allowed_collision_matrix.entry_values[a].enabled[b]=True
        world.allowed_collision_matrix.entry_values[b].enabled[a]=True
        result=reset_scene_diagnostic(world,TRAY_IDS)
        self.assertFalse(result['valid']);self.assertTrue(result['tray_support_permissions']['tray_wall_x_plus'])
    def test_support_permission_covers_only_parts_within_the_servo_proximity_band(self):
        from camera_scene import SUPPORT_CONTACT_IDS
        near=set()
        for obj in scene_message().world.collision_objects:
            if obj.id in TRAY_IDS:
                bottom=obj.primitive_poses[0].position.z-obj.primitives[0].dimensions[2]/2
                if bottom-.30<.02:near.add(obj.id)
        self.assertEqual(SUPPORT_CONTACT_IDS,near)

if __name__=='__main__':unittest.main()
