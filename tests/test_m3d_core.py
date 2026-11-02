"""Routing, evidence identity, payload scene and complete gate regressions."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np,yaml
from rclpy.serialization import serialize_message,deserialize_message
from moveit_msgs.msg import PlanningScene
from sorting_interfaces.msg import Decision,TargetBeliefArray,TargetBelief,ClassObservation
from sorting_interfaces.srv import SetCameraPlacement
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src/sorting_control/scripts'),str(R/'src/sorting_perception/scripts')]
from sort_core import choose_route,ClassWindow,placement_scene,footprint_inside_estimate
from camera_scene import TRAY_IDS,SUPPORT_CONTACT_IDS
from camera_scene_checks import collision_allowed
from publish_static_fixtures import scene_message
from m3d_acceptance import assess
C=yaml.safe_load((R/'src/sorting_control/config/m3d.yaml').read_text());BASE=yaml.safe_load((R/'src/sorting_control/config/m3b.yaml').read_text())
class RouteTests(unittest.TestCase):
    def test_clear_classes_route_to_distinct_bins(self):
        self.assertEqual(choose_route(C['class_ids'],[.9,.1],C)['bin_id'],'bin_a');self.assertEqual(choose_route(C['class_ids'],[.1,.9],C)['bin_id'],'bin_b')
    def test_covered_probability_does_not_invent_a_sort(self):
        value=choose_route(C['class_ids'],[.52,.48],C);self.assertEqual((value['action'],value['class_id'],value['bin_id']),('PASS','',''))
    def test_wrong_order_nan_and_unnormalized_probabilities_fail(self):
        for ids,p in [(list(reversed(C['class_ids'])),[.9,.1]),(C['class_ids'],[np.nan,.1]),(C['class_ids'],[.9,.9]),(C['class_ids'],[-.1,1.1])]:
            with self.assertRaises(RuntimeError):choose_route(ids,p,C)
    def test_repeated_publications_are_one_capture(self):
        w=ClassWindow(C);r=choose_route(C['class_ids'],[.9,.1],C)
        for _ in range(20):self.assertFalse(w.observe(1_000_000_000,r))
        self.assertFalse(w.observe(1_067_000_000,r));self.assertTrue(w.observe(1_134_000_000,r))
    def test_changed_class_discards_old_proof(self):
        w=ClassWindow(C);a=choose_route(C['class_ids'],[.9,.1],C);b=choose_route(C['class_ids'],[.1,.9],C)
        w.observe(1_000_000_000,a);w.observe(1_067_000_000,a);self.assertFalse(w.observe(1_134_000_000,b));self.assertEqual(len(w.samples),1)
class PayloadTests(unittest.TestCase):
    def test_floor_permission_is_narrow_and_preserves_payload(self):
        m=placement_scene(scene_message(),[0,0,0,0],BASE,True,np.eye(4),False,'bin_a_floor')
        self.assertEqual({a.object.id for a in m.robot_state.attached_collision_objects},TRAY_IDS);self.assertTrue(m.robot_state.is_diff)
        for part in TRAY_IDS:
            self.assertEqual(collision_allowed(m.allowed_collision_matrix,part,'bin_a_floor'),part in SUPPORT_CONTACT_IDS)
            self.assertFalse(collision_allowed(m.allowed_collision_matrix,part,'bin_b_floor'));self.assertFalse(collision_allowed(m.allowed_collision_matrix,part,'support_surface'))
            for link in ('panda_hand','panda_link5'):self.assertFalse(collision_allowed(m.allowed_collision_matrix,part,link))
        deserialize_message(serialize_message(m),PlanningScene)
    def test_destination_without_attachment_fails(self):
        with self.assertRaises(RuntimeError):placement_scene(scene_message(),[0,0,0,0],BASE,True,None,False,'bin_a_floor')
    def test_placed_world_has_eight_parts_and_revoked_floor_pairs(self):
        world=np.eye(4);world[:3,3]=C['bins']['bin_b']['center'];m=placement_scene(scene_message(),[0,0,0,0],BASE,False,None,False,placed=world)
        self.assertFalse(m.robot_state.attached_collision_objects);self.assertTrue(TRAY_IDS<={a.id for a in m.world.collision_objects})
        base=next(a for a in m.world.collision_objects if a.id=='tray_base');self.assertAlmostEqual(base.primitive_poses[0].position.z,.3059)
        for part in TRAY_IDS:
            for b in ('bin_a_floor','bin_b_floor','support_surface'):self.assertFalse(collision_allowed(m.allowed_collision_matrix,part,b))
    def test_full_footprint_rejects_a_center_near_wall_or_rotated_tray(self):
        world=np.eye(4);world[:3,3]=C['bins']['bin_a']['center'];self.assertTrue(footprint_inside_estimate(world,'bin_a',C));world[1,3]+=.03;self.assertFalse(footprint_inside_estimate(world,'bin_a',C))
        world[:3,3]=C['bins']['bin_a']['center'];a=np.pi/4;world[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]];self.assertFalse(footprint_inside_estimate(world,'bin_a',C))
    def test_release_retreat_permission_is_only_fingers_and_handle(self):
        world=np.eye(4);world[:3,3]=C['bins']['bin_a']['center'];m=placement_scene(scene_message(),[0,0,0,0],BASE,True,None,False,placed=world)
        for finger in ('panda_leftfinger','panda_rightfinger'):self.assertTrue(collision_allowed(m.allowed_collision_matrix,'handle_bar',finger))
        self.assertFalse(collision_allowed(m.allowed_collision_matrix,'tray_base','panda_hand'))
class EvidenceTests(unittest.TestCase):
    def test_actual_selection_requires_exact_capture_epoch_track_and_model(self):
        from camera_sort import SortController
        from types import SimpleNamespace as NS
        obj=SortController.__new__(SortController);obj.s=C;obj.epoch=NS(episode_id=2);obj.bound_track=7;obj.model_sha='model';obj.classes={};obj.classified=None;obj.now_ns=lambda:1_140_000_000
        obj.phase='pregrasp_refinement';obj.result={}
        decision=None
        def until(predicate,timeout):
            for capture in (1_000_000_000,1_067_000_000,1_134_000_000):
                b=TargetBelief(episode_id=2,track_id=7,view_id='wrist',valid=True,camera_only=True,class_ids=C['class_ids'],class_probabilities=[.9,.1]);b.header.stamp.sec=capture//10**9;b.header.stamp.nanosec=capture%10**9
                m=ClassObservation(episode_id=2,view_id='wrist',valid=True,class_ids=C['class_ids'],probabilities=[.9,.1],model_sha256='model');m.header=b.header
                obj.class_score(m);obj.classified=TargetBeliefArray(targets=[b]);ready=predicate()
            self.assertTrue(ready)
        obj.until=until;value=obj.select_class();self.assertEqual(value['track_id'],7);self.assertEqual(value['distinct_captures'],3)
        obj.classes.clear();obj.model_sha='changed'
        with self.assertRaisesRegex(RuntimeError,'model changed'):obj.select_class()
    def test_interfaces_serialize_installed_jazzy_types(self):
        m=Decision(episode_id=2,track_id=7,action='PICK_PLACE',proposed_bin='bin_a',remaining_time_s=6.0,estimated_service_time_s=65.0);deserialize_message(serialize_message(m),Decision)
        serialize_message(SetCameraPlacement.Request(episode_id=2,owner_epoch=11,bin_id='bin_a',operation='enable_floor'))
class GateTests(unittest.TestCase):
    def reports(self):
        values=[]
        for cls in ('red_fruit','green_fruit'):
            for covered in (False,True):
                count=1 if covered else 10;bin_id='bin_a' if cls=='red_fruit' else 'bin_b'
                trials=[dict(status='PASS',reset=dict(status='PASS'),decision=dict(class_id=cls),physical=dict(action='PASS' if covered else 'PICK_PLACE',selected_bin=bin_id)) for _ in range(count)]
                values.append(dict(checkpoint='M3D',status='PASS',isolation_audit={'status':'PASS'},expected_class=cls,expected_pass=covered,trials=trials,installed_manifest={'model':'same'},final_reset_status='PASS',final_owner='HOLD',overlap_violations=0,completed_trials=count,aborted_trials=0))
        return values
    def test_complete_two_class_gate_passes_and_keeps_supported_misses(self):
        r=self.reports()
        for i in (0,2):
            for j in (0,1):r[i]['trials'][j]['status']='FAIL'
        out=assess(r);self.assertEqual(out['status'],'PASS');self.assertEqual(out['successful_sorts'],16)
    def test_missing_reset_wrong_bin_overlap_mixed_models_and_partial_batch_fail(self):
        mutations=[lambda r:r[0]['trials'][0].pop('reset'),lambda r:r[0]['trials'][0]['physical'].update(selected_bin='bin_b'),lambda r:r[2].update(overlap_violations=1),lambda r:r[2].update(installed_manifest={'model':'different'}),lambda r:r[0]['trials'].pop(),lambda r:r[1]['trials'][0]['physical'].update(action='PICK_PLACE')]
        for change in mutations:
            r=self.reports();change(r);self.assertEqual(assess(r)['status'],'FAIL')
class PlanningAndIsolationTests(unittest.TestCase):
    def test_real_carried_plan_request_has_payload_and_path_constraints(self):
        from types import SimpleNamespace as NS
        from sort_planning import execute_carried_plan
        from m2b_common import timestamp
        c=Mock();c.s=C;c.last_carried_goal=None;c.settings=yaml.safe_load((R/'src/sorting_control/config/m2b.yaml').read_text());c.q={f'panda_joint{i}':0. for i in range(1,8)}
        c.q.update(panda_finger_joint1=.01,panda_finger_joint2=.01);c.scene_clients=['servo','planner_scene'];c.planner='planning';c.acquire_mode.return_value=(3,timestamp(55_000_000_000));c.hand_pose.return_value=(np.array([.55,0,.64]),[.7071068,.7071068,0.,0.])
        attached=placement_scene(scene_message(),[0,0,0,0],BASE,True,np.eye(4),False).robot_state.attached_collision_objects
        recorded=[]
        def call(client,request,**kwargs):
            if client=='planning':recorded.append(request);raise RuntimeError('recorded_before_live_service')
            return NS(scene=NS(robot_state=NS(attached_collision_objects=attached)))
        c.call.side_effect=call
        with self.assertRaisesRegex(RuntimeError,'recorded_before_live_service'):execute_carried_plan(c,[.52,.02,.64],[.7071068,.7071068,0.,0.],Path('/tmp'),'test')
        req=recorded[0].motion_plan_request;self.assertTrue(req.start_state.is_diff);self.assertEqual({a.object.id for a in req.start_state.attached_collision_objects},TRAY_IDS)
        self.assertEqual(req.path_constraints.orientation_constraints[0].link_name,'panda_hand');self.assertEqual(req.path_constraints.orientation_constraints[0].absolute_x_axis_tolerance,.01)
        self.assertEqual(req.goal_constraints[0].orientation_constraints[0].absolute_x_axis_tolerance,.004)
        self.assertAlmostEqual(req.path_constraints.position_constraints[0].constraint_region.primitives[0].dimensions[2],.01)
        serialize_message(recorded[0])
    def test_sort_controller_has_one_typed_arm_source_and_no_truth_inputs(self):
        from camera_sort import SortController
        from ament_index_python.packages import get_package_share_directory
        node=Mock();node.get_parameter.return_value.value=str(Path(get_package_share_directory('sorting_perception'))/'models/m3c_baseline')
        with patch('camera_pickup.Buffer'),patch('camera_pickup.TransformListener'),patch('camera_pickup.ActionClient'):
            SortController(node)
        subs=[a.args[1] for a in node.create_subscription.call_args_list];pubs=[a.args[1] for a in node.create_publisher.call_args_list]
        self.assertIn('/sorting/classified_targets',subs);self.assertFalse(any('ground_truth' in t or t in ('/sorting/contact_state','/sorting/reset_state') for t in subs));self.assertEqual(pubs.count('/sorting/arm/servo_command'),1);self.assertFalse(any('joint_trajectory' in t for t in pubs))

if __name__=='__main__':unittest.main()
