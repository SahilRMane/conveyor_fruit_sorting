"""Class continuity, default M3 behaviour and the shared conveyor command path."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
import m4a_common
from m4d_core import RouteLease
from m4d_pickup import InspectedSortController
from camera_pickup import Controller
from camera_sort import SortController
from m4d_filter import PickupViewFilter
from sorting_interfaces.msg import HandleObservation
from m4d_evaluator import IntegratedEvaluator


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.c=dict(maximum_capture_age_s=.15,maximum_tracking_gap_s=.3,maximum_capture_binding_error_m=.003)
        self.decision=dict(action='PICK_READY',class_id='red_fruit',proposed_bin='bin_a',confidence=.99,
            probabilities=[.99,.01],episode_id=2,track_id=8,decision_time_ns=1_000_000_000,
            evidence_capture_ns=980_000_000)
        self.target=dict(episode_id=2,track_id=8,capture_ns=1_000_000_000,
                         valid=True,camera_only=True,view_id='overhead',frame_id='world')
        self.lease=RouteLease(self.decision,'model',30.)

    def test_class_can_persist_while_target_feedback_stays_current(self):
        for i in range(1,251):
            now=1_000_000_000+i*40_000_000
            self.target['capture_ns']=now-20_000_000
            route=self.lease.check(self.target,now,self.c)
        self.assertEqual(route['bin_id'],'bin_a')
        self.assertGreater(route['class_age_sim_s'],10.)
        self.assertTrue(route['class_carried_on_continuous_track'])

    def test_route_does_not_mutate_with_input_or_caller_changes(self):
        self.decision['class_id']='green_fruit';self.decision['probabilities'][0]=0.
        self.assertEqual(self.lease.route['class_id'],'red_fruit')
        self.assertEqual(self.lease.route['probabilities'],(.99,.01))
        with self.assertRaises(TypeError):self.lease.route['class_id']='green_fruit'

    def test_invalid_track_cannot_recover_an_old_class(self):
        self.target['valid']=False
        with self.assertRaisesRegex(ValueError,'valid camera'):
            self.lease.check(self.target,1_020_000_000,self.c)
        self.target['valid']=True
        with self.assertRaisesRegex(ValueError,'valid camera'):
            self.lease.check(self.target,1_020_000_000,self.c)

    def test_changed_identity_is_rejected(self):
        for key in ('episode_id','track_id'):
            with self.subTest(key=key):
                lease=RouteLease(self.decision,'model',30.)
                target=deepcopy(self.target);target[key]+=1
                with self.assertRaisesRegex(ValueError,'identity'):
                    lease.check(target,1_020_000_000,self.c)

    def test_stale_future_and_backward_clock_are_rejected(self):
        for now,capture in ((1_150_000_000,1_000_000_000),(1_010_000_000,1_020_000_000),
                            (999_000_000,999_000_000)):
            with self.subTest(now=now,capture=capture):
                lease=RouteLease(self.decision,'model',30.)
                target=deepcopy(self.target);target['capture_ns']=capture
                with self.assertRaises(ValueError):lease.check(target,now,self.c)

    def test_long_gap_is_rejected_even_if_final_capture_is_fresh(self):
        self.lease.check(self.target,1_020_000_000,self.c)
        self.target['capture_ns']=1_320_000_000
        with self.assertRaisesRegex(ValueError,'tracking gap'):
            self.lease.check(self.target,1_340_000_000,self.c)

    def test_class_route_expires(self):
        self.target['capture_ns']=31_000_000_000
        with self.assertRaisesRegex(ValueError,'expired'):
            self.lease.check(self.target,31_000_000_000,self.c)

    def test_pass_cannot_create_a_pickup_route(self):
        self.decision['action']='PASS'
        with self.assertRaisesRegex(ValueError,'PICK_READY'):
            RouteLease(self.decision,'model',30.)

    def test_checked_wrist_transition_retains_the_class_and_track(self):
        self.target.update(view_id='wrist',capture_ns=1_040_000_000,view_bridge=dict(episode_id=2,track_id=8,
            from_view='overhead',to_view='wrist',overhead_capture_ns=1_000_000_000,
            capture_ns=1_040_000_000,binding_error_m=.0005))
        route=self.lease.check(self.target,1_060_000_000,self.c)
        self.assertEqual(route['class_id'],'red_fruit');self.assertEqual(route['track_id'],8)
        self.assertEqual(self.lease.view,'wrist')
        self.target['view_id']='overhead'
        with self.assertRaisesRegex(ValueError,'view changed'):
            self.lease.check(self.target,1_060_000_000,self.c)

    def test_wrist_without_proof_or_with_wrong_spatial_pair_is_rejected(self):
        for bridge in ({},dict(episode_id=2,track_id=8,from_view='overhead',to_view='wrist',binding_error_m=.004)):
            lease=RouteLease(self.decision,'model',30.)
            target=dict(self.target,view_id='wrist',view_bridge=bridge)
            with self.assertRaises(ValueError):lease.check(target,1_020_000_000,self.c)


class ExistingControllerTests(unittest.TestCase):
    def waiting_controller(self):
        c=InspectedSortController.__new__(InspectedSortController)
        c.phase='lower';c.input_enabled=True;c.input_epoch=4;c.input_deadline=3_000_000_000
        c.owner=dict(owner='SERVO',busy=False,epoch=4);c.now_ns=lambda:1_000_000_000
        c.c=dict(maximum_tf_age_s=.1);c.s=dict(supported_release=dict(maximum_wrench_age_s=.05,maximum_wrench_wall_age_s=.2))
        c.integration=dict(wrench_tf_wait_s=.05,maximum_wrench_tf_wait_total_s=.2)
        c.wrench_wait=None;c.wrench_wait_total=0.;c.result={};c.publisher=SimpleNamespace(publish=Mock())
        c.wrist=SimpleNamespace(last_diagnostic=dict(status='UNAVAILABLE',rejected=dict(capture_time_tf=1),
            newest_wrench_stamp_ns=990_000_000,latest_tf_stamp_ns=970_000_000,
            newest_wrench_age_s=.01,newest_receive_age_wall_s=.005,latest_tf_age_s=.03))
        return c

    def test_pending_capture_tf_sends_zero_motion_without_reusing_force(self):
        c=self.waiting_controller()
        self.assertTrue(c.defer_wrench_tf(RuntimeError('Fresh capture-time wrist wrench unavailable')))
        message=c.publisher.publish.call_args.args[0]
        self.assertEqual(message.twist.twist.linear.x,0.)
        self.assertEqual(message.twist.twist.linear.y,0.)
        self.assertEqual(message.twist.twist.linear.z,0.)
        self.assertEqual(message.epoch,4)
        self.assertEqual(message.deadline.sec*1_000_000_000+message.deadline.nanosec,1_050_000_000)

    def test_old_force_or_invalid_force_is_not_hidden_by_a_tf_wait(self):
        c=self.waiting_controller();c.wrist.last_diagnostic['newest_wrench_age_s']=.051
        self.assertFalse(c.defer_wrench_tf(RuntimeError('Fresh capture-time wrist wrench unavailable')))
        self.assertFalse(c.defer_wrench_tf(RuntimeError('Invalid wrist wrench values')))
        c.publisher.publish.assert_not_called()

    def test_capture_tf_wait_is_bounded(self):
        c=self.waiting_controller();error=RuntimeError('Fresh capture-time wrist wrench unavailable')
        c.defer_wrench_tf(error);c.now_ns=lambda:1_051_000_000
        with self.assertRaisesRegex(RuntimeError,'exceeded its bound'):
            c.defer_wrench_tf(error)

    def test_wrist_wait_finishes_only_with_a_fresh_coherent_sample(self):
        c=self.waiting_controller();c.defer_wrench_tf(RuntimeError('Fresh capture-time wrist wrench unavailable'))
        c.finish_wrench_wait();self.assertIsNotNone(c.wrench_wait)
        c.wrist.last_diagnostic=dict(status='PASS',stamp_ns=990_000_000)
        c.finish_wrench_wait();self.assertIsNone(c.wrench_wait)
        self.assertEqual(c.result['wrench_tf_waits'][0]['status'],'RESUMED')

    def test_expired_wait_cannot_publish_normal_motion_after_tf_arrives(self):
        c=self.waiting_controller();c.check_execution=Mock()
        c.defer_wrench_tf(RuntimeError('Fresh capture-time wrist wrench unavailable'))
        c.now_ns=lambda:1_051_000_000
        c.wrist.last_diagnostic=dict(status='PASS',stamp_ns=1_040_000_000)
        c.wrist.sample=Mock(return_value=(1_040_000_000,1.))
        with patch.object(SortController,'send') as send:
            with self.assertRaisesRegex(RuntimeError,'exceeded its bound'):c.send()
            send.assert_not_called()

    def test_physical_evaluator_is_ready_for_its_first_trial(self):
        evaluator=IntegratedEvaluator(Mock(),Mock())
        self.assertEqual(evaluator.trial_error_start,0)
        self.assertEqual(evaluator.maximum_transfer_rise,0.)
        self.assertIsNone(evaluator.grasp)

    def tare_controller(self, force):
        c=InspectedSortController.__new__(InspectedSortController)
        c.empty_tare=1.;c.owner=dict(owner='HOLD',busy=False)
        c.belief=SimpleNamespace(view_id='wrist');c.inspector=SimpleNamespace(pickup_timing=SimpleNamespace(status='PASS'))
        c.q=dict(panda_finger_joint1=.04,panda_finger_joint2=.04);c.scene=dict(estimated_attachment=False)
        c.s=dict(supported_release=dict(tare_maximum_spread_N=.1,tare_span_s=.2))
        c.wrist=SimpleNamespace(tare=None,sample=Mock(side_effect=[(i*100_000_000,force) for i in range(3)]))
        c.result={};c.route=dict(class_id='red_fruit');c.select_class=Mock(return_value=c.route)
        c.publish_decision=Mock()
        def until(predicate, timeout):
            for _ in range(3):
                if predicate():return
            raise TimeoutError('Wrist verification did not pass')
        c.until=until
        return c

    def test_moving_empty_check_does_not_recalibrate_the_sensor(self):
        c=self.tare_controller(1.04);c.before_tracking()
        self.assertEqual(c.wrist.tare,1.)
        self.assertEqual(c.result['moving_empty_wrist_verification'],'PASS')
        c.publish_decision.assert_called_once()

    def test_wrong_empty_load_cannot_reach_closure(self):
        c=self.tare_controller(1.4)
        with self.assertRaises(TimeoutError):c.before_tracking()
        self.assertIsNone(c.wrist.tare)
        c.publish_decision.assert_not_called()

    def test_integrated_handoff_creates_no_second_servo_or_belt_publisher(self):
        c=InspectedSortController.__new__(InspectedSortController)
        c.inspector=SimpleNamespace(servo_publisher=object(),belt=object())
        node=Mock()
        self.assertIs(c.command_publisher(node),c.inspector.servo_publisher)
        self.assertIs(c.belt_publisher(node),c.inspector.belt)
        node.create_publisher.assert_not_called()

    def test_default_publisher_topics_remain_unchanged(self):
        c=Controller.__new__(Controller);node=Mock()
        c.command_publisher(node);c.belt_publisher(node)
        topics=[call.args[1] for call in node.create_publisher.call_args_list]
        self.assertEqual(topics,['/sorting/arm/servo_command','/sorting/conveyor_command'])

    def test_integrated_controller_shares_tf_without_changing_the_default(self):
        c=InspectedSortController.__new__(InspectedSortController)
        c.inspector=SimpleNamespace(buffer=object());node=Mock()
        with patch('camera_pickup.Buffer') as buffer,patch('camera_pickup.TransformListener') as listener:
            self.assertEqual(c.transform_support(node),(c.inspector.buffer,None))
            buffer.assert_not_called();listener.assert_not_called()
            default=Controller.__new__(Controller)
            self.assertEqual(default.transform_support(node),(buffer.return_value,listener.return_value))
            listener.assert_called_once_with(buffer.return_value,node)

    def test_default_pregrasp_keeps_planning_and_refinement(self):
        c=Controller.__new__(Controller)
        c.fc=dict(fixed_handle_z_m=.4229)
        c.settings=dict(grasp_quaternion_xyzw=[.70710678,.70710678,0.,0.])
        c.scene=dict(valid=True)
        c.set_phase=Mock()
        c.target=Mock(return_value=(np.array([.53,0.,.001,0.]),np.eye(4),1))
        c.execute_plan=Mock(return_value={'pipeline':'ompl'})
        c.refine_pose=Mock(return_value={'status':'PASS'})
        c.until=lambda predicate,timeout:self.assertTrue(predicate())
        result=dict(checks={})
        pre,orientation=c.prepare_pregrasp(Path('/tmp'),result)
        self.assertAlmostEqual(pre[2],.5858)
        c.execute_plan.assert_called_once_with(pre,orientation,Path('/tmp'),'camera-pregrasp')
        c.refine_pose.assert_called_once_with(pre,orientation)
        self.assertEqual(result['checks']['camera_pregrasp'],'PASS')

    def test_default_and_inspected_models_are_explicit(self):
        default=SortController.__new__(SortController)
        inspected=InspectedSortController.__new__(InspectedSortController)
        self.assertEqual(default.classifier_directory().name,'m3c_baseline')
        self.assertEqual(inspected.classifier_directory().name,'m4b_baseline')

    def test_belt_speed_changes_use_the_inspection_publisher(self):
        c=InspectedSortController.__new__(InspectedSortController)
        publisher=SimpleNamespace(publish=Mock())
        c.inspector=SimpleNamespace(belt_speed=.001,belt=publisher)
        c.belt=publisher;c.belt_speed=0.;c.enabled=True
        c.send_belt()
        self.assertEqual(c.inspector.belt_speed,0.)
        message=publisher.publish.call_args.args[0]
        self.assertIs(type(message.data),float)
        self.assertEqual(message.data,0.)

    def test_warm_controller_cannot_stop_the_inspection_belt(self):
        c=InspectedSortController.__new__(InspectedSortController)
        c.inspector=SimpleNamespace(belt_speed=.001)
        c.enabled=False;c.belt_speed=0.;c.belt=SimpleNamespace(publish=Mock())
        c.send_belt()
        self.assertEqual(c.inspector.belt_speed,.001)
        c.belt.publish.assert_not_called()


class TransportTests(unittest.TestCase):
    def evaluator(self):
        evaluator=IntegratedEvaluator.__new__(IntegratedEvaluator)
        evaluator.transport=[dict(sim_time=2+i*.1,command_fresh=True,requested_speed_m_s=.001,
            force_x_N=.2,support_force_N=5.,tray_vx_m_s=.001) for i in range(20)]
        return evaluator

    def test_free_tray_speed_is_checked_before_closure(self):
        evaluator=self.evaluator();evaluator.transport[0]['tray_vx_m_s']=-.003
        with self.assertRaisesRegex(RuntimeError,'tray speed'):
            evaluator.transport_checks(0,4_000_000_000,.001,3_000_000_000)

    def test_contact_phase_keeps_belt_command_and_force_limits(self):
        evaluator=self.evaluator();evaluator.transport[10]['tray_vx_m_s']=-.003
        self.assertEqual(evaluator.transport_checks(0,4_000_000_000,.001,3_000_000_000)['status'],'PASS')
        for field,value in (('command_fresh',False),('requested_speed_m_s',0.),('force_x_N',9.)):
            with self.subTest(field=field):
                changed=self.evaluator();changed.transport[10][field]=value
                with self.assertRaises(RuntimeError):changed.transport_checks(0,4_000_000_000,.001,3_000_000_000)

    def test_closure_marker_cannot_bypass_free_transport(self):
        evaluator=self.evaluator()
        for closure in (0,1_100_000_000,4_000_000_000):
            with self.subTest(closure=closure):
                with self.assertRaises(RuntimeError):evaluator.transport_checks(0,4_000_000_000,.001,closure)

    def test_belt_may_stop_after_lift_but_not_during_lift(self):
        evaluator=self.evaluator();evaluator.transport[18]['requested_speed_m_s']=0.
        self.assertEqual(evaluator.transport_checks(0,3_500_000_000,.001,3_000_000_000)['status'],'PASS')
        evaluator.transport[14]['requested_speed_m_s']=0.
        with self.assertRaisesRegex(RuntimeError,'Conveyor stopped'):
            evaluator.transport_checks(0,3_500_000_000,.001,3_000_000_000)


class CameraTransitionTests(unittest.TestCase):
    def node(self):
        c=PickupViewFilter.__new__(PickupViewFilter)
        c.requested_wrist=True;c.bridge=None;c.held=False;c.rewind_episode=None;c.timing_trace=None;c.pending={}
        c.integration=dict(maximum_capture_age_s=.15,maximum_capture_binding_error_m=.003)
        c.c=dict(tracking_view='overhead',acceleration_sigma_m_s2=.005)
        c.track=SimpleNamespace(episode=2,track_id=8,stamp_ns=1_000_000_000,
                               x=np.array([.53,0.,.001,0.]),p=np.eye(4)*1e-8)
        return c

    def observation(self,x):
        m=HandleObservation();m.episode_id=2;m.valid=True;m.view_id='wrist'
        m.header.frame_id='world';m.header.stamp.sec=1;m.position.x=x
        return m

    def test_same_capture_can_bind_views_without_a_second_filter_update(self):
        c=self.node();m=self.observation(.5302);c.receive(m)
        self.assertEqual(c.c['tracking_view'],'wrist')
        self.assertLess(c.bridge['binding_error_m'],.003)
        self.assertEqual(c.pending,{})
        m.header.stamp.nanosec=40_000_000;c.receive(m)
        self.assertIn(1_040_000_000,c.pending)
        self.assertEqual(c.track.track_id,8)

    def test_other_position_cannot_switch_the_camera(self):
        c=self.node();c.receive(self.observation(.534))
        self.assertEqual(c.c['tracking_view'],'overhead')
        self.assertIsNone(c.bridge)
        self.assertEqual(c.pending,{})

    def test_no_camera_switch_without_a_pickup_request(self):
        c=self.node();c.requested_wrist=False;c.receive(self.observation(.5302))
        self.assertEqual(c.c['tracking_view'],'overhead')


if __name__=='__main__':unittest.main()
