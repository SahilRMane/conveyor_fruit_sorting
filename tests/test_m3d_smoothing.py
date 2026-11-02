"""Installed plugin discovery, bounded small commands and startup checks; no ROS node."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,create_autospec
import ast,json,subprocess,sys,tempfile,unittest
import yaml
from rcl_interfaces.msg import ParameterValue,ParameterType
from ament_index_python.packages import get_package_prefix
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/sorting_control/scripts'),str(ROOT/'src/sorting_perception/scripts')]
from m3d_components import capture_pairing_check,runtime_configuration,SMOOTHER
from test_arm_ownership import Probe
from m3b_evaluator import Evaluator

class ComponentTests(unittest.TestCase):
    def test_installed_plugin_loads_and_all_bounded_step_cases_pass(self):
        executable=Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_precision_smoothing'
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'result.json'
            result=subprocess.run([str(executable),str(out)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(out.read_text())['cases'],16)
    def test_real_tf_buffer_rejects_slow_pairing_and_accepts_configured_pairing(self):
        settings=yaml.safe_load((ROOT/'src/sorting_control/config/m3d.yaml').read_text())
        report=capture_pairing_check(settings)
        self.assertGreater(report['cases']['negative_20_Hz']['unavailable'],0)
        self.assertEqual(report['cases']['configured_100_Hz']['paired'],11)
        self.assertLess(report['cases']['configured_100_Hz']['maximum_age_s'],.05)
    def runtime(self,override=None):
        expected={
          '/robot_state_publisher':{'use_sim_time':True,'publish_frequency':100.},
          '/sorting_servo':{'use_sim_time':True,'robot_description_kinematics.panda_arm.epsilon':1e-7,
            'moveit_servo.use_smoothing':True,'moveit_servo.smoothing_filter_plugin_name':SMOOTHER,'moveit_servo.publish_period':.01},
          '/controller_manager':{'use_sim_time':True,'update_rate':100}}
        e=NS(node=Mock())
        # Bind against the actual evaluator method signatures. This catches use
        # of a nonexistent timeout argument before a live trial is attempted.
        prototype=Probe.__new__(Probe)
        e.until=create_autospec(prototype.until,side_effect=lambda predicate,timeout: self.assertTrue(predicate()))
        e.future=create_autospec(prototype.future,side_effect=lambda future,timeout:future)
        def client(service,path):
            name=path.removesuffix('/get_parameters');client=Mock();client.service_is_ready.return_value=True
            def call(request):
                values=[]
                for key in request.names:
                    value=expected[name][key]
                    if override and (name,key)==override[:2]:value=override[2]
                    p=ParameterValue()
                    if type(value) is bool:p.type=ParameterType.PARAMETER_BOOL;p.bool_value=value
                    elif type(value) is str:p.type=ParameterType.PARAMETER_STRING;p.string_value=value
                    elif type(value) is int:p.type=ParameterType.PARAMETER_INTEGER;p.integer_value=value
                    else:p.type=ParameterType.PARAMETER_DOUBLE;p.double_value=value
                    values.append(p)
                return NS(values=values)
            client.call_async.side_effect=call;return client
        e.node.create_client.side_effect=client
        return e
    def test_runtime_valid_parameters_pass_and_clients_are_closed(self):
        e=self.runtime();result=runtime_configuration(e)
        self.assertEqual(result['status'],'PASS');self.assertEqual(e.node.destroy_client.call_count,3)
        # Methods used here are the methods supplied by the real evaluator.
        self.assertIs(Evaluator.until,Probe.until);self.assertIs(Evaluator.future,Probe.future)
        for call in e.until.call_args_list+e.future.call_args_list:self.assertEqual(call.args[1],5.)
    def test_old_plugin_loose_ik_wrong_rates_and_wall_clock_fail_before_motion(self):
        failures=[('/sorting_servo','moveit_servo.smoothing_filter_plugin_name','online_signal_smoothing::AccelerationLimitedPlugin'),
          ('/sorting_servo','robot_description_kinematics.panda_arm.epsilon',1e-5),
          ('/sorting_servo','use_sim_time',False),('/sorting_servo','moveit_servo.use_smoothing',False),
          ('/sorting_servo','moveit_servo.publish_period',.02),
          ('/robot_state_publisher','publish_frequency',20.),('/controller_manager','update_rate',50)]
        for failure in failures:
            with self.subTest(failure=failure):
                e=self.runtime(failure)
                with self.assertRaisesRegex(RuntimeError,'Running parameter mismatch'):runtime_configuration(e)
                self.assertGreater(e.node.destroy_client.call_count,0)
    def test_parameter_service_failure_closes_client(self):
        e=self.runtime();e.future.side_effect=RuntimeError('service deadline')
        with self.assertRaisesRegex(RuntimeError,'service deadline'):runtime_configuration(e)
        self.assertEqual(e.node.destroy_client.call_count,1)
    def test_m3d_override_leaves_shared_servo_configuration_unchanged(self):
        tree=ast.parse((ROOT/'src/sorting_control/launch/panda_m3d.launch.py').read_text())
        assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Subscript) and 'smoothing_filter_plugin_name' in ast.unparse(x) for x in n.targets)]
        self.assertEqual(len(assignments),1);self.assertEqual(ast.literal_eval(assignments[0].value),SMOOTHER)
        shared=yaml.safe_load((ROOT/'src/sorting_control/config/servo.yaml').read_text())
        self.assertEqual(shared['moveit_servo']['smoothing_filter_plugin_name'],'online_signal_smoothing::AccelerationLimitedPlugin')
    def test_components_only_report_cannot_authorize_physical_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'validation.json').write_text(json.dumps(dict(checkpoint='M3D-components',status='PASS',installed_manifest={})))
            # The parser must reject this report before rclpy.init is reached.
            result=subprocess.run([sys.executable,str(ROOT/'src/sorting_perception/scripts/test_m3d.py'),
                '--preflight-dir',str(p),'--expected-class','red_fruit'],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Run a fresh installed M3D preflight',result.stderr)

if __name__=='__main__':unittest.main()
