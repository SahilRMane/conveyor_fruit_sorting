"""Physical wrist support, freshness, frame and model contracts; no ROS node."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import copy,sys,time,unittest,xml.etree.ElementTree as ET,subprocess,tempfile
import numpy as np,yaml,xacro
from geometry_msgs.msg import WrenchStamped,TransformStamped
from rclpy.serialization import serialize_message,deserialize_message
from ament_index_python.packages import get_package_prefix
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/sorting_control/scripts')]
from wrist_support import WristObserver,SupportWindow
from sort_core import supported_floor_pose
from m2b_common import timestamp
C=yaml.safe_load((R/'src/sorting_control/config/m3d.yaml').read_text());S=C['supported_release']

class WristTests(unittest.TestCase):
    def observer(self):
        c=NS(node=Mock(),s=C,buffer=Mock(),now_ns=lambda:1_020_000_000)
        tf=TransformStamped();tf.transform.rotation.w=1.;c.buffer.lookup_transform.return_value=tf
        o=WristObserver(c);m=WrenchStamped();m.header.frame_id='panda_hand';m.header.stamp=timestamp(1_000_000_000)
        m.wrench.force.z=7.4556;o.receive(deserialize_message(serialize_message(m),WrenchStamped))
        return o,m
    def test_force_uses_exact_capture_time_transform(self):
        o,m=self.observer();stamp,force=o.sample();self.assertEqual(stamp,1_000_000_000);self.assertAlmostEqual(force,7.4556)
        args=o.controller.buffer.lookup_transform.call_args.args;self.assertEqual(args[:2],('world','panda_hand'))
        self.assertEqual(args[2].nanoseconds,stamp)
    def test_frame_rotation_changes_force_to_world(self):
        o,m=self.observer();tf=o.controller.buffer.lookup_transform.return_value
        tf.transform.rotation.w=0.;tf.transform.rotation.x=1.
        self.assertAlmostEqual(o.sample()[1],-7.4556)
    def test_stale_or_future_force_cannot_be_used(self):
        for now in (1_050_000_000,970_000_000):
            o,m=self.observer();o.controller.now_ns=lambda:now
            with self.assertRaisesRegex(RuntimeError,'unavailable'):o.sample()
    def test_stopped_wrench_source_is_rejected(self):
        o,m=self.observer();o.frames[1_000_000_000]=(m,time.monotonic()-1.)
        with self.assertRaisesRegex(RuntimeError,'unavailable'):o.sample()
    def test_wrong_frame_and_nonfinite_data_fail(self):
        for change in ('frame','nan'):
            o,m=self.observer()
            if change=='frame':m.header.frame_id='world'
            else:m.wrench.force.z=float('nan')
            o.receive(m)
            with self.assertRaises(RuntimeError):o.sample()
    def test_loaded_force_requires_tare_and_known_payload_range(self):
        o,m=self.observer()
        with self.assertRaisesRegex(RuntimeError,'tare'):o.measure_loaded()
        o.tare=7.4556
        with self.assertRaisesRegex(RuntimeError,'calibration'):o.measure_loaded()
        m.wrench.force.z=12.5568;o.receive(m)
        self.assertAlmostEqual(o.measure_loaded(),5.1012);self.assertAlmostEqual(o.payload()[1],5.1012)
    def test_support_requires_distinct_samples_and_time_span(self):
        p=SupportWindow(S)
        self.assertFalse(p.observe(1_000_000_000,1.,5.1012))
        for _ in range(10):self.assertFalse(p.observe(1_000_000_000,1.,5.1012))
        self.assertFalse(p.observe(1_010_000_000,1.,5.1012));self.assertTrue(p.observe(1_020_000_000,1.,5.1012))
    def test_airborne_load_cannot_authorize_opening(self):
        p=SupportWindow(S)
        for stamp in range(1_000_000_000,1_100_000_000,10_000_000):self.assertFalse(p.observe(stamp,5.,5.1012))
        with self.assertRaises(RuntimeError):p.observe(1_200_000_000,float('nan'),5.1012)
    def test_config_cannot_weaken_support_age_or_speed_proof(self):
        for key,value in [('support_fraction',.5),('stable_span_s',.01),('minimum_samples',1),('maximum_wrench_age_s',.15),('approach_speed_m_s',.001)]:
            c=copy.deepcopy(S);c[key]=value
            with self.assertRaises(RuntimeError):SupportWindow(c)
    def test_supported_plane_changes_only_estimate_and_preserves_input(self):
        w=np.eye(4);w[:3,3]=C['bins']['bin_a']['center'];w[2,3]+=.004
        before=w.copy();out=supported_floor_pose(w,'bin_a',C)
        np.testing.assert_array_equal(before,w);self.assertAlmostEqual(out[2,3],.3059)
        self.assertEqual(list(out[:2,3]),list(w[:2,3]))
    def test_supported_plane_requires_destination_and_bounded_correction(self):
        w=np.eye(4);w[:3,3]=C['bins']['bin_a']['center']
        for change in ('height','bin','nan'):
            q=w.copy();name='bin_a'
            if change=='height':q[2,3]+=.007
            elif change=='bin':name='bin_b'
            else:q[2,3]=float('nan')
            with self.assertRaises(RuntimeError):supported_floor_pose(q,name,C)
    def test_sensor_copies_preserve_original_robot_physics(self):
        old=ET.parse(R/'src/sorting_description/mujoco/static_grasp/panda_camera_m3c.xml').getroot()
        new=ET.parse(R/'src/sorting_description/mujoco/static_grasp/panda_camera_m3d.xml').getroot()
        hand=new.find(".//body[@name='hand']");site=hand.find("site[@name='panda_hand_wrench_site']")
        self.assertIsNotNone(site);hand.remove(site);new.remove(new.find('sensor'))
        for root in (new,old):
            for node in root.iter():
                if node.text and not node.text.strip():node.text=None
                if node.tail and not node.tail.strip():node.tail=None
        self.assertEqual(ET.tostring(new),ET.tostring(old))
    def test_ft_ros_interfaces_are_optional_for_earlier_checkpoints(self):
        path=R/'src/sorting_description/urdf/panda_cameras.urdf.xacro'
        for enabled,count in [('true',1),('false',0)]:
            root=ET.fromstring(xacro.process_file(str(path),mappings={'enable_wrist_ft':enabled}).toxml())
            sensors=root.findall('ros2_control/sensor');self.assertEqual(len(sensors),count)
            if count:self.assertEqual({i.get('name') for i in sensors[0].findall('state_interface')},{'force.x','force.y','force.z','torque.x','torque.y','torque.z'})
    def test_controller_and_scene_read_only_physical_wrist_topic(self):
        o,m=self.observer();topic=o.controller.node.create_subscription.call_args.args[1]
        self.assertEqual(topic,'/panda_hand_wrench_broadcaster/wrench')
        self.assertNotIn('ground_truth',topic)
        text=(R/'src/sorting_control/scripts/camera_sort.py').read_text()
        self.assertNotIn('bin_floor_force_N',text);self.assertNotIn('nc_truth',text)
    def test_launch_and_manifest_use_the_wrist_sensor_scene(self):
        launch=(R/'src/sorting_control/launch/panda_m3d.launch.py').read_text()
        for value in ('wrist_sensor.yaml','panda_hand_wrench_broadcaster'):self.assertIn(value,launch)
        from m3d_parameters import model_parameters
        model=ET.fromstring(model_parameters()['robot_description'])
        self.assertEqual(len(model.findall('ros2_control/sensor')),1)
        self.assertIn('scene_camera_m3d_red_clear.xml',ET.tostring(model,encoding='unicode'))
        manifest=(R/'src/sorting_perception/scripts/m3d_manifest.py').read_text()
        self.assertIn('panda_camera_m3d',manifest);self.assertIn('wrist_support.py',manifest)

class DriverBindingTests(unittest.TestCase):
    def verify(self,change=None):
        path=R/'src/sorting_description/urdf/panda_cameras.urdf.xacro'
        root=ET.fromstring(xacro.process_file(str(path),mappings={'enable_wrist_ft':'true'}).toxml())
        sensor=root.find('ros2_control/sensor')
        if change:change(root,sensor)
        model=R/'src/sorting_description/mujoco/static_grasp'
        executable=Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_wrist_model'
        with tempfile.TemporaryDirectory() as folder:
            urdf=Path(folder)/'panda.urdf';ET.ElementTree(root).write(urdf,encoding='unicode')
            return subprocess.run([str(executable),str(urdf),str(model/'scene_camera_m3d_red_clear.xml'),str(model/'scene_camera_m3c_red_clear.xml')],capture_output=True,text=True)
    def test_correct_binding_passes_native_parser_and_model(self):
        result=self.verify();self.assertEqual(result.returncode,0,result.stderr)
    def test_original_missing_type_is_rejected(self):
        result=self.verify(lambda root,s:s.remove(s.find("param[@name='mujoco_type']")))
        self.assertNotEqual(result.returncode,0);self.assertIn('mujoco_type=fts',result.stderr)
    def test_wrong_sensor_type_is_rejected(self):
        result=self.verify(lambda root,s:setattr(s.find("param[@name='mujoco_type']"),'text','imu'))
        self.assertNotEqual(result.returncode,0);self.assertIn('mujoco_type=fts',result.stderr)
    def test_missing_name_override_is_rejected(self):
        result=self.verify(lambda root,s:s.remove(s.find("param[@name='mujoco_sensor_name']")))
        self.assertNotEqual(result.returncode,0);self.assertIn('mujoco_sensor_name=panda_hand_wrench',result.stderr)
    def test_wrong_name_override_is_rejected(self):
        result=self.verify(lambda root,s:setattr(s.find("param[@name='mujoco_sensor_name']"),'text','absent'))
        self.assertNotEqual(result.returncode,0);self.assertIn('mujoco_sensor_name=panda_hand_wrench',result.stderr)
    def test_missing_mjcf_pair_is_rejected(self):
        def change(root,s):
            ET.SubElement(root.find('ros2_control/hardware'),'param',name='force_mjcf_suffix').text='_absent'
        result=self.verify(change)
        self.assertNotEqual(result.returncode,0);self.assertIn('Physical wrist sensor mapping mismatch',result.stderr)

if __name__=='__main__':unittest.main()
