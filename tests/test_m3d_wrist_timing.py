"""Capture-time force/TF pairing under delay; no ROS node or live graph."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import ast,copy,json,sys,time,unittest
import numpy as np,yaml
from geometry_msgs.msg import TransformStamped,WrenchStamped
from tf2_ros import Buffer,TransformException
from rclpy.time import Time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_control/scripts'))
from wrist_support import WristObserver
from m2b_common import timestamp
CONFIG=yaml.safe_load((ROOT/'src/sorting_control/config/m3d.yaml').read_text())

class WristTimingTests(unittest.TestCase):
    def observer(self,now):
        c=NS(node=Mock(),s=copy.deepcopy(CONFIG),buffer=Buffer(),now_ns=lambda:now)
        return WristObserver(c)
    def tf(self,observer,stamp):
        tf=TransformStamped();tf.header.frame_id='world';tf.child_frame_id='panda_hand'
        tf.header.stamp=timestamp(stamp);tf.transform.rotation.w=1.
        observer.controller.buffer.set_transform(tf,'offline_timing_test')
    def force(self,observer,stamp):
        m=WrenchStamped();m.header.frame_id='panda_hand';m.header.stamp=timestamp(stamp)
        m.wrench.force.z=-12.5568;observer.receive(m)
    def delayed_pairing(self,period):
        failed=[];ages=[]
        # Joint/wrench samples are 100 Hz. TF delivery takes 12 ms and
        # wrench delivery takes 8 ms. Decisions run every 20 ms.
        for now in range(1_064_000_000,1_265_000_000,20_000_000):
            o=self.observer(now)
            for stamp in range(1_000_000_000,now+1,period):
                if stamp+12_000_000<=now:self.tf(o,stamp)
            for stamp in range(1_000_000_000,now+1,10_000_000):
                if stamp+8_000_000<=now:self.force(o,stamp)
            try:
                stamp,value=o.sample();self.assertAlmostEqual(value,-12.5568);ages.append((now-stamp)/1e9)
            except RuntimeError:
                failed.append(o.last_diagnostic)
        return failed,ages
    def test_default_20_hz_tf_reproduces_missing_fresh_pair(self):
        failed,ages=self.delayed_pairing(50_000_000)
        self.assertTrue(failed)
        for d in failed:
            self.assertGreater(d['rejected']['capture_time_tf'],0)
            self.assertLess(d['newest_wrench_age_s'],.05)
            self.assertGreaterEqual(d['latest_tf_age_s'],.05)
    def test_100_hz_tf_pairs_same_inputs_without_relaxing_age(self):
        failed,ages=self.delayed_pairing(10_000_000)
        self.assertEqual(failed,[]);self.assertEqual(len(ages),11)
        self.assertLess(max(ages),.05)
    def test_fresh_wrench_without_capture_tf_reports_tf_failure(self):
        o=self.observer(1_020_000_000);self.tf(o,1_000_000_000);self.force(o,1_010_000_000)
        with self.assertRaisesRegex(RuntimeError,'unavailable'):o.sample()
        d=o.last_diagnostic;self.assertEqual(d['rejected']['capture_time_tf'],1)
        self.assertTrue(d['capture_time_tf_error']);self.assertAlmostEqual(d['latest_tf_age_s'],.02)
        json.dumps(d,allow_nan=False)
    def test_old_force_reports_capture_age_and_is_not_reused(self):
        o=self.observer(1_100_000_000);self.tf(o,1_100_000_000);self.force(o,1_000_000_000)
        with self.assertRaisesRegex(RuntimeError,'unavailable'):o.sample()
        d=o.last_diagnostic;self.assertEqual(d['rejected']['capture_age'],1)
        self.assertEqual(d['rejected']['capture_time_tf'],0);self.assertAlmostEqual(d['newest_wrench_age_s'],.1)
    def test_stopped_force_reports_wall_age(self):
        o=self.observer(1_020_000_000);self.tf(o,1_010_000_000);self.force(o,1_010_000_000)
        m,_=o.frames[1_010_000_000];o.frames[1_010_000_000]=(m,time.monotonic()-1.)
        with self.assertRaisesRegex(RuntimeError,'unavailable'):o.sample()
        self.assertEqual(o.last_diagnostic['rejected']['receive_age'],1)
    def test_empty_stream_is_distinct_from_missing_tf(self):
        o=self.observer(1_020_000_000)
        with self.assertRaisesRegex(RuntimeError,'unavailable'):o.sample()
        d=o.last_diagnostic;self.assertEqual(d['buffered_samples'],0)
        self.assertIsNone(d['newest_wrench_stamp_ns']);self.assertTrue(d['latest_tf_error'])
    def test_diagnostic_retains_one_last_good_sample_without_recursion(self):
        o=self.observer(1_020_000_000);self.tf(o,1_010_000_000);self.force(o,1_010_000_000)
        o.sample();good=copy.deepcopy(o.last_diagnostic);o.controller.now_ns=lambda:1_100_000_000
        for _ in range(3):
            with self.assertRaisesRegex(RuntimeError,'unavailable'):o.sample()
            self.assertEqual(o.last_diagnostic['last_valid_sample'],good)
        json.dumps(o.last_diagnostic,allow_nan=False)
    def test_launch_tf_rate_matches_joint_feedback_and_retains_age_limit(self):
        tree=ast.parse((ROOT/'src/sorting_control/launch/panda_m3d.launch.py').read_text())
        rates=[]
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='Node':continue
            keywords={k.arg:k.value for k in node.keywords}
            if not isinstance(keywords.get('package'),ast.Constant) or keywords['package'].value!='robot_state_publisher':continue
            for item in keywords['parameters'].elts:
                if isinstance(item,ast.Dict):rates.append(ast.literal_eval(item)['publish_frequency'])
        feedback=yaml.safe_load((ROOT/'src/sorting_description/config/panda_controllers.yaml').read_text())['controller_manager']['ros__parameters']['update_rate']
        self.assertEqual(rates,[float(feedback)]);self.assertEqual(rates,[100.])
        self.assertEqual(CONFIG['supported_release']['maximum_wrench_age_s'],.05)

if __name__=='__main__':unittest.main()
