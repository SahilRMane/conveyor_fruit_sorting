from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import yaml
from sensor_msgs.msg import Image,CameraInfo
from geometry_msgs.msg import TransformStamped
from sorting_interfaces.msg import CameraEpoch,HandleObservation
from rclpy.serialization import serialize_message,deserialize_message
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/sorting_perception/scripts"))
from camera_core import CaptureGuard
from handle_localizer import HandleLocalizer
from m3a_probe import CameraProbe

ROOT=Path(__file__).resolve().parents[1];C=yaml.safe_load((ROOT/"src/sorting_perception/config/m3a.yaml").read_text())

class CallbackTests(unittest.TestCase):
    def probe(self,age_ns=20_000_000):
        rgb=np.zeros((480,640,3),np.uint8);rgb[220:260,313:327]=[220,10,220]
        image=Image(width=640,height=480,encoding="rgb8",step=1920,data=rgb.tobytes());image.header.stamp.sec=1;image.header.frame_id="overhead_camera_optical_frame"
        info=CameraInfo(width=640,height=480,distortion_model="plumb_bob",d=[0.]*5,k=[500.,0.,320.,0.,500.,240.,0.,0.,1.]);info.header=image.header
        transform=TransformStamped();transform.transform.translation.x=.65;transform.transform.translation.z=1.35;transform.transform.rotation.x=1.;transform.transform.rotation.w=0.
        publisher=Mock();buffer=Mock();buffer.lookup_transform.return_value=transform
        guard=CaptureGuard();guard.reset(0,0)
        return SimpleNamespace(c=C,cameras={"cameras":{"overhead":{"frame":image.header.frame_id}}},guard=guard,
            pending={"overhead":(image,0.)},info={"overhead":{1_000_000_000:info}},sequence=0,
            stats={"overhead":dict(processed=0,rejected=0,valid=0,reasons={})},observation_publishers={"overhead":publisher},
            buffer=buffer,last_clock_ns=0,clock_episode=0,epoch_message=None,
            get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=1_000_000_000+age_ns)))
    def test_callback_looks_up_exact_capture_stamp_and_preserves_it_in_result(self):
        p=self.probe();HandleLocalizer.process(p)
        self.assertEqual(p.buffer.lookup_transform.call_args.args[2].nanoseconds,1_000_000_000)
        result=p.observation_publishers["overhead"].publish.call_args.args[0]
        self.assertTrue(result.valid);self.assertEqual(result.header.stamp.sec,1);self.assertEqual(result.header.frame_id,"world")
        self.assertAlmostEqual(result.position.x,.65);serialize_message(result)
    def test_stale_frame_is_invalid_and_cannot_refresh_capture_order(self):
        p=self.probe(200_000_000);HandleLocalizer.process(p)
        result=p.observation_publishers["overhead"].publish.call_args.args[0]
        self.assertFalse(result.valid);self.assertEqual(result.reason,"capture_age_outside_bound")
        self.assertEqual(p.guard.last_stamp,{});p.buffer.lookup_transform.assert_not_called()
    def test_missing_matching_camera_info_cannot_use_a_newer_frame_calibration(self):
        p=self.probe();p.info={"overhead":{1_010_000_000:next(iter(p.info["overhead"].values()))}}
        HandleLocalizer.process(p);result=p.observation_publishers["overhead"].publish.call_args.args[0]
        self.assertFalse(result.valid);p.buffer.lookup_transform.assert_not_called()
    def test_clock_rewind_requires_reset_identity(self):
        p=self.probe();p.last_clock_ns=2_000_000_000;p.clock_episode=0
        p.stats["overhead"]["last_received_ns"]=2_000_000_000
        HandleLocalizer.process(p);self.assertIsNone(p.guard.episode);self.assertEqual(p.pending,{})
    def test_reset_packet_received_before_clock_rewind_is_retained(self):
        p=self.probe();p.last_clock_ns=2_000_000_000;p.clock_episode=0
        epoch=CameraEpoch(episode_id=1);epoch.header.stamp.sec=1;p.epoch_message=epoch
        HandleLocalizer.process(p);self.assertEqual(p.guard.episode,1);self.assertEqual(p.guard.barrier_ns,1_000_000_000)
    def test_fixed_covariance_and_episode_messages_serialize_with_real_types(self):
        epoch=CameraEpoch(episode_id=3);epoch.header.stamp.sec=7
        self.assertEqual(deserialize_message(serialize_message(epoch),CameraEpoch).episode_id,3)
        observation=HandleObservation(episode_id=3,valid=True,planar_covariance=[1e-6,0.,0.,1e-6])
        np.testing.assert_array_equal(deserialize_message(serialize_message(observation),HandleObservation).planar_covariance,observation.planar_covariance)

class IsolationTests(unittest.TestCase):
    def fake(self,extra=None,consumer="test_m3a"):
        subscriptions=[("/camera/overhead/image_raw",[]),("/tf",[]),("/sorting/episode",[])]
        if extra:subscriptions.append((extra,[]))
        node=SimpleNamespace(get_node_names_and_namespaces=lambda:[("handle_localizer","/"),("test_m3a","/")],
              get_subscriber_names_and_types_by_node=lambda *args:subscriptions,
              get_subscriptions_info_by_topic=lambda *args:[SimpleNamespace(node_name=consumer)])
        return SimpleNamespace(node=node,camera_status={"oracle_input":False})
    def test_image_only_localizer_passes_graph_audit(self):self.assertEqual(CameraProbe.audit(self.fake())["status"],"PASS")
    def test_physical_truth_or_debug_subscription_fails_graph_audit(self):
        for topic in ("/sorting/contact_state","/sorting/ground_truth/camera_sample","/sorting/debug/oracle_target"):
            with self.assertRaises(RuntimeError):CameraProbe.audit(self.fake(topic))
    def test_truth_subscriber_outside_evaluator_fails_graph_audit(self):
        with self.assertRaises(RuntimeError):CameraProbe.audit(self.fake(consumer="handle_localizer"))

if __name__=="__main__":unittest.main()
