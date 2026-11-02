from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy,HistoryPolicy,LivelinessPolicy,QoSCompatibility,qos_check_compatible
from sensor_msgs.msg import Image,CameraInfo
from sorting_interfaces.msg import HandleObservation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/sorting_perception/scripts"))
from camera_qos import camera_subscription_qos
from handle_localizer import HandleLocalizer
from m3a_probe import CameraProbe

class QosTests(unittest.TestCase):
    def test_metadata_reader_matches_the_reliable_upstream_publisher(self):
        image,info=camera_subscription_qos()
        writer=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE,liveliness=LivelinessPolicy.AUTOMATIC)
        self.assertEqual(qos_check_compatible(writer,info)[0],QoSCompatibility.OK)
        self.assertEqual(qos_check_compatible(writer,image)[0],QoSCompatibility.OK)
        self.assertEqual(info.reliability,ReliabilityPolicy.RELIABLE);self.assertEqual(info.depth,64)
        self.assertEqual(info.history,HistoryPolicy.KEEP_LAST);self.assertEqual(info.durability,DurabilityPolicy.VOLATILE)
        self.assertEqual(image.reliability,ReliabilityPolicy.BEST_EFFORT);self.assertEqual(image.depth,1)
    def check_subscriptions(self,calls,image_depth):
        images=[call.args[3] for call in calls if call.args[0] is Image]
        info=[call.args[3] for call in calls if call.args[0] is CameraInfo]
        self.assertEqual(len(images),2);self.assertEqual(len(info),2)
        for qos in images:self.assertEqual(qos.reliability,ReliabilityPolicy.BEST_EFFORT);self.assertEqual(qos.depth,image_depth)
        for qos in info:self.assertEqual(qos.reliability,ReliabilityPolicy.RELIABLE);self.assertEqual(qos.depth,64)
    def test_localizer_constructor_uses_reliable_metadata_and_latest_images(self):
        obj=HandleLocalizer.__new__(HandleLocalizer)
        obj.create_subscription=Mock();obj.create_publisher=Mock();obj.create_timer=Mock()
        obj.declare_parameter=Mock();obj.get_parameter=Mock(side_effect=lambda name:SimpleNamespace(value="cameras.yaml" if name=="camera_config" else "m3a.yaml"))
        with patch("handle_localizer.Node.__init__",return_value=None),patch("handle_localizer.Buffer"),patch("handle_localizer.TransformListener"):
            HandleLocalizer.__init__(obj)
        self.check_subscriptions(obj.create_subscription.call_args_list,1)
    def test_evaluator_constructor_uses_reliable_metadata(self):
        obj=CameraProbe.__new__(CameraProbe);node=Mock()
        with patch("m3a_probe.Probe.__init__",return_value=None),patch("m3a_probe.Buffer"),patch("m3a_probe.TransformListener"),patch("m3a_probe.ResetManager"):
            CameraProbe.__init__(obj,node)
        self.check_subscriptions(node.create_subscription.call_args_list,5)

class LostPacketTests(unittest.TestCase):
    def test_actual_lost_stamp_is_not_replaced_with_neighbouring_metadata(self):
        stamp=28_355_999_999;m=HandleObservation(valid=True,episode_id=0,view_id="overhead")
        m.header.stamp.sec=28;m.header.stamp.nanosec=355_999_999;key=(0,"overhead",stamp)
        p=SimpleNamespace(evaluation_enabled=True,pending={key:(m,0.,"stationary")},truth={key:object()},
            info={("overhead",stamp-104_000_000):object(),("overhead",stamp+104_000_000):object()},
            pairing_diagnostics=[],reported_pair_waits=set(),rejections=[],samples=[],buffer=Mock())
        p.pairing_detail=lambda *args:CameraProbe.pairing_detail(p,*args)
        with patch("m3a_probe.time.monotonic",return_value=2.1):CameraProbe.evaluate(p)
        self.assertEqual(p.pending,{});self.assertEqual(p.samples,[])
        self.assertEqual(p.rejections[0]["capture_ns"],stamp);self.assertFalse(p.rejections[0]["counted"])
        self.assertEqual(p.pairing_diagnostics[0]["status"],"SKIPPED_METADATA")
        self.assertEqual(p.pairing_diagnostics[0]["nearest_camera_info_delta_ns"],104_000_000)
        p.buffer.lookup_transform.assert_not_called()

if __name__=="__main__":unittest.main()
