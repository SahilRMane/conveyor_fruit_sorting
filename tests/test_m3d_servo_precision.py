"""Small Servo IK steps and placement result ordering; no ROS node."""
from pathlib import Path
from unittest.mock import patch
import ast,copy,json,subprocess,sys,tempfile,unittest
import xacro,yaml
from std_msgs.msg import String
from ament_index_python.packages import get_package_prefix
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/sorting_control/scripts'),str(ROOT/'src/sorting_perception/scripts')]
from m3d_evaluator import SortEvaluator
from m3b_evaluator import Evaluator

class PrecisionTests(unittest.TestCase):
    def native(self,epsilon):
        executable=Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_servo_precision'
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);urdf=p/'panda.urdf';out=p/'result.json'
            urdf.write_text(xacro.process_file(str(ROOT/'src/sorting_description/urdf/panda_cameras.urdf.xacro')).toxml())
            result=subprocess.run([str(executable),str(urdf),str(epsilon),str(out)],capture_output=True,text=True)
            return result,json.loads(out.read_text()) if out.exists() else None
    def test_native_kdl_resolves_all_eight_small_step_cases(self):
        result,report=self.native(1e-7);self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(report['installed_kdl_default_epsilon'],1e-5)
        self.assertEqual(report['servo_epsilon'],1e-7);self.assertEqual(len(report['cases']),8)
        for case in report['cases']:
            self.assertLess(case['step_m'],1e-5)
            self.assertLess(abs(case['default_descent_m']),1e-10)
            self.assertAlmostEqual(case['corrected_descent_m'],case['speed_m_s'],delta=.05*case['speed_m_s'])
    def test_loose_or_invalid_precision_cannot_pass_new_check(self):
        for epsilon in (1e-5,0.,float('nan')):
            result,report=self.native(epsilon);self.assertNotEqual(result.returncode,0)
            self.assertIsNone(report)
    def test_precision_override_applies_only_to_m3d_servo(self):
        tree=ast.parse((ROOT/'src/sorting_control/launch/panda_m3d.launch.py').read_text());overrides=[]
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='Node':continue
            keywords={k.arg:k.value for k in node.keywords};package=ast.literal_eval(keywords['package'])
            for dictionary in ast.walk(keywords.get('parameters',ast.List(elts=[]))):
                if not isinstance(dictionary,ast.Dict):continue
                for key,value in zip(dictionary.keys,dictionary.values):
                    if isinstance(key,ast.Constant) and key.value=='robot_description_kinematics.panda_arm.epsilon':
                        overrides.append((package,ast.literal_eval(value)))
        self.assertEqual(overrides,[('moveit_servo',1e-7)])

class PlacementPhaseTests(unittest.TestCase):
    def evaluator(self):
        e=SortEvaluator.__new__(SortEvaluator)
        e.s=yaml.safe_load((ROOT/'src/sorting_control/config/m3d.yaml').read_text())
        e.placement_started=True;e.release_seen=False;e.last_placement_phase=None;e.reset_in_progress=False
        e.phase='startup';e.control_state={};e.maximum_transfer_rise=0.;e.maximum_slip=0.;e.grip_trace=[]
        e.grasp=dict(tray_xyz=[.45,.42,.3059],hand_xyz=[.45,.42,.5276],hand_quat_wxyz=[1.,0.,0.,0.],tray_quat_wxyz=[1.,0.,0.,0.])
        e.decision=None
        value=dict(e.grasp,sim_time=147.,tray_xyz=[.45,.42,.30935],hand_xyz=[.45,.42,.53105],
            forbidden_contacts=0,tray_floor_force_N=0.,bin_wall_force_N=[0.,0.],left_force_N=4.98,right_force_N=4.98,
            external_force_x_N=0.,support_force_N=0.,bin_floor_force_N=[0.,0.])
        return e,value
    def state(self,e,phase):e.receive_control(String(data=json.dumps(dict(phase=phase,active=phase!='idle'))))
    def test_idle_before_result_keeps_lower_checks(self):
        e,value=self.evaluator();self.state(e,'lower');self.state(e,'idle')
        self.assertEqual(e.phase,'lower');e.check(value)
        self.assertEqual(e.phase,'lower')
    def test_idle_after_transfer_still_requires_airborne_clearance(self):
        e,value=self.evaluator();self.state(e,'transfer');self.state(e,'idle')
        with self.assertRaisesRegex(RuntimeError,'Transfer clearance'):e.check(value)
    def test_idle_after_release_keeps_release_checks(self):
        e,value=self.evaluator();self.state(e,'release');self.state(e,'idle')
        value['left_force_N']=value['right_force_N']=0.
        e.check(value);self.assertEqual(e.phase,'release')
    def test_selected_floor_force_limit_still_rejects_overload(self):
        e,value=self.evaluator();self.state(e,'lower');self.state(e,'idle')
        from types import SimpleNamespace
        e.decision=SimpleNamespace(proposed_bin='bin_a');value['bin_floor_force_N']=[10.3,0.]
        with self.assertRaisesRegex(RuntimeError,'twice payload weight'):e.check(value)
    def test_forbidden_contact_still_fails_after_idle(self):
        e,value=self.evaluator();self.state(e,'lower');self.state(e,'idle');value['forbidden_contacts']=1
        with self.assertRaisesRegex(RuntimeError,'Forbidden placement'):e.check(value)
    def test_reset_clears_previous_placement_phase(self):
        e,value=self.evaluator();self.state(e,'lower');self.state(e,'idle');e.scene_clients=[]
        with patch.object(Evaluator,'reset_episode',return_value={'status':'PASS'}):e.reset_episode()
        self.assertIsNone(e.last_placement_phase);self.assertFalse(e.placement_started)
        self.state(e,'idle');self.assertEqual(e.phase,'idle')
    def test_trial_start_clears_previous_placement_phase(self):
        e,value=self.evaluator();self.state(e,'lower');e.filter_errors=[]
        e.prepare_trial();self.assertIsNone(e.last_placement_phase);self.assertFalse(e.placement_started)

if __name__=='__main__':unittest.main()
