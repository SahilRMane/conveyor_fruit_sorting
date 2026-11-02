"""Real native post-processing regression and carried-command admission checks."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import copy,importlib.util,json,subprocess,sys,tempfile,unittest,xml.etree.ElementTree as ET
import numpy as np,xacro,yaml
from ament_index_python.packages import get_package_prefix,get_package_share_directory
from moveit_msgs.srv import GetMotionPlan
from trajectory_msgs.msg import JointTrajectoryPoint
from rcl_interfaces.msg import ParameterValue,ParameterType
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/sorting_control/scripts'),str(ROOT/'src/sorting_perception/scripts')]
from sort_planning import execute_carried_plan,carried_request
from carried_checks import fixtures,planning_configuration
from ownership_core import NAMES
from m2b_common import timestamp
from m3d_manifest import installed_manifest

class CarriedChecks(unittest.TestCase):
    def native(self,profile_override=None,group_override=None):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);d=Path(get_package_share_directory('sorting_description'));m=Path(get_package_share_directory('moveit_resources_panda_moveit_config'))
            (p/'panda.urdf').write_text(xacro.process_file(str(d/'urdf/panda_cameras.urdf.xacro'),mappings={'enable_wrist_ft':'true'}).toxml())
            srdf=ET.parse(m/'config/panda.srdf').getroot()
            for item in srdf.findall('virtual_joint'):srdf.remove(item)
            ET.ElementTree(srdf).write(p/'panda.srdf',encoding='unicode');fixtures(p/'fixtures')
            profile=yaml.safe_load((ROOT/'src/sorting_control/config/carried_planning.yaml').read_text())
            if profile_override:profile['profile'].update(profile_override)
            if group_override:profile['group'].update(group_override)
            (p/'profile.yaml').write_text(yaml.safe_dump(profile))
            exe=Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_carried_motion'
            r=subprocess.run([str(exe),str(p/'panda.urdf'),str(p/'panda.srdf'),str(m/'config/joint_limits.yaml'),str(p/'fixtures/cases.yaml'),str(p/'profile.yaml'),str(p/'validation.json')],capture_output=True,text=True,timeout=60)
            return r,json.loads((p/'validation.json').read_text()) if (p/'validation.json').exists() else None
    def test_native_rejects_failed_path_and_validates_twenty_retimed_paths(self):
        r,v=self.native();self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(v['corrected_paths'],120);self.assertEqual(v['old_invalid_states'],723)
        self.assertEqual(v['recorded_invalid_states'],729);self.assertEqual(v['recorded_failed_paths'],3)
        self.assertEqual(v['interior_margin_checks'],32);self.assertEqual(v['margin_plugin_load'],'PASS')
    def test_missing_joint_space_enforcement_cannot_pass_native_gate(self):
        r,v=self.native(group_override=dict(enforce_joint_model_state_space=False));self.assertNotEqual(r.returncode,0);self.assertIsNone(v)
    def test_installed_ros_boolean_conversion_is_checked(self):
        r,v=self.native();self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(v['parameter_conversion'],dict(status='PASS',planner_bool_text='true',planner_bool_rejected=True,group_bool_text='1'))
    def test_planner_boolean_override_is_rejected_even_with_valid_group(self):
        r,v=self.native(dict(enforce_joint_model_state_space=True))
        self.assertNotEqual(r.returncode,0);self.assertIsNone(v);self.assertIn('Remove planner-level bool',r.stderr)
    def test_launch_places_boolean_at_group_level_without_planner_override(self):
        from launch import LaunchContext
        path=ROOT/'src/sorting_control/launch/panda_m3d.launch.py'
        spec=importlib.util.spec_from_file_location('m3d_launch_check',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context=LaunchContext();context.launch_configurations.update(headless='true',scene_fixture='red_clear')
        with patch.object(module,'Node') as node:
            module.setup(context)
        planning=next(c.kwargs['parameters'][1]['ompl'] for c in node.call_args_list if c.kwargs['package']=='moveit_ros_move_group')
        self.assertIs(planning['panda_arm']['enforce_joint_model_state_space'],True)
        self.assertNotIn('enforce_joint_model_state_space',planning['planner_configs']['CarriedRRTConnect'])
        self.assertEqual(planning['planning_plugins'],['sorting_control/CarriedMarginPlanner'])
    def test_regression_fixtures_preserve_both_green_retract_failures(self):
        from rclpy.serialization import deserialize_message
        from moveit_msgs.msg import MotionPlanRequest
        with tempfile.TemporaryDirectory() as folder:
            items=fixtures(folder);self.assertEqual(len(items),4)
            self.assertEqual(sum(item['native_repeats'] for item in items),120)
            for item in items[2:]:
                req=deserialize_message(Path(item['request']).read_bytes(),MotionPlanRequest)
                self.assertEqual(len(req.start_state.attached_collision_objects),8)
                self.assertAlmostEqual(req.goal_constraints[0].position_constraints[0].constraint_region.primitive_poses[0].position.x,.5499896446083137)
                self.assertAlmostEqual(req.path_constraints.position_constraints[0].constraint_region.primitives[0].dimensions[2],.010000210440329504)
                self.assertEqual(len(req.path_constraints.orientation_constraints),1)
    def test_old_twenty_plan_proof_cannot_authorize_a_physical_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);pre=p/'pre';pre.mkdir();proof=p/'proof';proof.mkdir();manifest=installed_manifest()
            (pre/'validation.json').write_text(json.dumps(dict(checkpoint='M3D',status='PASS',installed_manifest=manifest)))
            (proof/'validation.json').write_text(json.dumps(dict(checkpoint='M3D-carried-planning',status='PASS',successful_plans=20,installed_manifest=manifest)))
            result=subprocess.run([sys.executable,str(ROOT/'src/sorting_perception/scripts/test_m3d.py'),'--preflight-dir',str(pre),'--planning-probe-dir',str(proof),'--expected-class','green_fruit'],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0);self.assertIn('fresh installed planning-only probe',result.stderr)
    def controller(self,codes):
        c=NS();c.s=yaml.safe_load((ROOT/'src/sorting_control/config/m3d.yaml').read_text());c.settings=yaml.safe_load((ROOT/'src/sorting_control/config/m2b.yaml').read_text())
        c.q=dict(zip(NAMES,[0.,0.,0.,-1.57,0.,1.57,-.785]));c.q.update(panda_finger_joint1=.01,panda_finger_joint2=.01)
        c.hand_pose=lambda:(np.array([.48,.26,.645]),c.settings['grasp_quaternion_xyzw'])
        c.last_carried_goal=None;c.now_ns=lambda:1_000_000_000;c.acquire_mode=lambda *args:(1,timestamp(56_000_000_000));c.wait_sim=lambda s:None;c.ensure_oracle_current=Mock();c.scene_clients=['servo_scene','planner_scene'];c.planner='planner';c.validity='validity';c.result={};c.input_enabled=False
        with tempfile.TemporaryDirectory() as folder:
            from rclpy.serialization import deserialize_message
            from moveit_msgs.msg import PlanningScene
            item=fixtures(folder)[0];attached=deserialize_message(Path(item['scene']).read_bytes(),PlanningScene).robot_state.attached_collision_objects
        c.calls=[];c.valid=True;c.move_after_error=False
        def call(client,request,**kwargs):
            c.calls.append(client)
            if client=='planner':
                response=GetMotionPlan.Response();code=codes.pop(0);response.motion_plan_response.error_code.val=code
                trajectory=response.motion_plan_response.trajectory.joint_trajectory;trajectory.joint_names=NAMES
                for sec in (0,1):
                    point=JointTrajectoryPoint(positions=[c.q[n] for n in NAMES]);point.time_from_start.sec=sec;trajectory.points.append(point)
                if code!=1 and c.move_after_error:c.q[NAMES[0]]+=.01
                return response
            if client=='validity':return NS(valid=c.valid)
            return NS(scene=NS(robot_state=NS(attached_collision_objects=copy.deepcopy(attached))))
        c.call=call;c.future=lambda value,*args:value
        outcome=NS(status=4,result=NS(success=True,reason='unit outcome'))
        handle=NS(accepted=True,get_result_async=lambda:outcome)
        c.action=NS(send_goal_async=Mock(return_value=handle));return c
    def run_plan(self,c):
        with tempfile.TemporaryDirectory() as folder:
            result=execute_carried_plan(c,[.48,.26,.645],c.settings['grasp_quaternion_xyzw'],Path(folder),'test')
            self.assertTrue((Path(folder)/'test-attempt-01-request.json').is_file())
            return result
    def test_retry_executes_only_the_validated_second_plan(self):
        c=self.controller([99999,1]);self.run_plan(c);self.assertEqual(c.calls.count('planner'),2);self.assertEqual(c.action.send_goal_async.call_count,1)
        self.assertEqual([v['code'] for v in c.result['carried_planning_attempts']],[99999,1])
    def test_two_invalid_plans_never_reach_execution(self):
        c=self.controller([99999,99999])
        with self.assertRaisesRegex(RuntimeError,'Carried MoveIt plan failed'):self.run_plan(c)
        c.action.send_goal_async.assert_not_called();self.assertEqual(c.calls.count('planner'),2)
    def test_start_collision_is_not_retried(self):
        c=self.controller([-10])
        with self.assertRaisesRegex(RuntimeError,'Carried MoveIt plan failed'):self.run_plan(c)
        self.assertEqual(c.calls.count('planner'),1);c.action.send_goal_async.assert_not_called()
    def test_moved_start_blocks_retry(self):
        c=self.controller([99999]);c.move_after_error=True
        with self.assertRaisesRegex(RuntimeError,'start moved before retry'):self.run_plan(c)
        c.action.send_goal_async.assert_not_called()
    def test_short_lease_blocks_retry(self):
        c=self.controller([99999]);c.acquire_mode=lambda *args:(1,timestamp(20_000_000_000))
        with self.assertRaisesRegex(RuntimeError,'lease is too short for retry'):self.run_plan(c)
        c.action.send_goal_async.assert_not_called()
    def test_invalid_start_blocks_retry(self):
        c=self.controller([99999]);c.valid=False
        with self.assertRaisesRegex(RuntimeError,'start invalid before retry'):self.run_plan(c)
        c.action.send_goal_async.assert_not_called()
    def test_unbounded_retry_setting_is_rejected(self):
        c=self.controller([1]);profile=yaml.safe_load((ROOT/'src/sorting_control/config/carried_planning.yaml').read_text());profile['maximum_attempts']=3
        with patch('yaml.safe_load',return_value=profile):
            with self.assertRaisesRegex(RuntimeError,'attempts must be 1 or 2'):self.run_plan(c)
        c.action.send_goal_async.assert_not_called()
    def test_request_preserves_eight_parts_and_original_path_bounds(self):
        from rclpy.serialization import deserialize_message
        from moveit_msgs.msg import MotionPlanRequest
        with tempfile.TemporaryDirectory() as folder:
            item=fixtures(folder)[0];r=deserialize_message(Path(item['request']).read_bytes(),MotionPlanRequest)
            self.assertEqual(r.planner_id,'CarriedRRTConnect');self.assertEqual(len(r.start_state.attached_collision_objects),8);self.assertTrue(r.start_state.is_diff)
            self.assertEqual(r.path_constraints.orientation_constraints[0].absolute_x_axis_tolerance,.01)
            self.assertAlmostEqual(r.path_constraints.position_constraints[0].constraint_region.primitives[0].dimensions[2],.010143548189974316)
    def test_live_pilot_requires_planning_probe_before_node_start(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'validation.json').write_text(json.dumps(dict(checkpoint='M3D',status='PASS',installed_manifest=installed_manifest())))
            r=subprocess.run([sys.executable,str(ROOT/'src/sorting_perception/scripts/test_m3d.py'),'--preflight-dir',str(p),'--expected-class','red_fruit'],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(r.returncode,0);self.assertIn('planning-only probe',r.stderr)
    def probe_run(self,details=None,returncode=0,missing=False):
        import probe_m3d_planning as probe
        from contextlib import redirect_stdout
        import io
        calls=[]
        def run(command,**kwargs):
            calls.append((command,kwargs));folder=Path(command[-1]);parameters=yaml.safe_load(Path(command[1]).read_text())['/m3d_fixture_planner']['ros__parameters']
            self.assertFalse(parameters['use_sim_time']);self.assertFalse(parameters['allow_trajectory_execution'])
            self.assertEqual(parameters['ompl']['planning_plugins'],['sorting_control/CarriedMarginPlanner'])
            self.assertEqual(kwargs['env']['ROS_DOMAIN_ID'],'67')
            self.assertEqual(Path(command[0]).name,'verify_carried_pipeline')
            self.assertEqual(len(yaml.safe_load(Path(command[2]).read_text())['cases']),4)
            if not missing:
                data=dict(status='PASS',robot_commanded=False,scene_context='private_fixture_scenes',successful_plans=40,cases=[])
                if details:data.update(details)
                (folder/'pipeline-report.yaml').write_text(yaml.safe_dump(data))
            return NS(returncode=returncode,stdout='component log',stderr='component diagnostic')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);pre=p/'pre';pre.mkdir();out=p/'out'
            (pre/'validation.json').write_text(json.dumps(dict(checkpoint='M3D-components',status='PASS',installed_manifest=installed_manifest(),native_carried_motion=dict(status='PASS'))))
            with patch.object(probe.subprocess,'run',side_effect=run),patch.object(sys,'argv',['probe','--preflight-dir',str(pre),'--output-dir',str(out)]),redirect_stdout(io.StringIO()):
                code=probe.main()
            report=json.loads((out/'validation.json').read_text())
            self.assertIn('component log',(out/'pipeline-log.txt').read_text())
        self.assertEqual(len(calls),1)
        return code,report
    def test_planning_probe_uses_complete_private_scenes_without_live_services(self):
        code,report=self.probe_run();self.assertEqual(code,0);self.assertFalse(report['robot_commanded'])
        self.assertEqual(report['successful_plans'],40);self.assertEqual(report['scene_context'],'private_fixture_scenes')
    def test_private_pipeline_failure_preserves_diagnostic(self):
        code,report=self.probe_run(dict(status='FAIL',successful_plans=20,error='Saved requested start is in collision'),1)
        self.assertEqual(code,1);self.assertEqual(report['successful_plans'],20);self.assertIn('in collision',report['error'])
    def test_private_pipeline_missing_report_points_to_process_log(self):
        code,report=self.probe_run(returncode=1,missing=True)
        self.assertEqual(code,1);self.assertIn('pipeline-log.txt',report['error']);self.assertIn('component diagnostic',report['error'])
    def test_proof_from_live_scene_context_is_rejected(self):
        code,report=self.probe_run(dict(scene_context='live_robot_scene'))
        self.assertEqual(code,1);self.assertEqual(report['status'],'FAIL')
    def test_component_parameters_pass_the_installed_ros_yaml_parser_without_a_node(self):
        import probe_m3d_planning as probe
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);parameters=p/'parameters.yaml';parameters.write_text(yaml.safe_dump(probe.pipeline_parameters()))
            exe=Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_carried_pipeline'
            result=subprocess.run([str(exe),'--check-parameters',str(parameters),str(p/'report.json')],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr);report=json.loads((p/'report.json').read_text())
            self.assertEqual(report['status'],'PASS');self.assertGreater(report['parameters'],20)
    def test_install_manifest_binds_shared_parameters_and_pipeline_binary(self):
        manifest=installed_manifest()
        self.assertIn('sorting_control/m3d_parameters.py',manifest)
        self.assertIn('carried_pipeline_check',manifest)
        self.assertEqual(len(manifest['sorting_control/m3d_parameters.py']),64)
    def test_probe_and_live_launch_share_model_and_pipeline_parameters(self):
        import probe_m3d_planning as probe
        from m3d_parameters import model_parameters,planning_parameters
        expected=dict(model_parameters(),**planning_parameters());actual=probe.pipeline_parameters()['/m3d_fixture_planner']['ros__parameters']
        actual['use_sim_time']=True;self.assertEqual(actual,expected)
    def test_previous_trial_reset_cannot_mark_current_failed_trial_as_reset(self):
        import test_m3d as runner
        from contextlib import redirect_stdout
        import io
        class FakeEvaluator:
            def __init__(self,node):
                self.owner=dict(owner='HOLD',ready=True,stats=dict(planned_accepted=0,servo_input_accepted=0,overlap_violations=0))
                self.contact=dict(support_force_N=0);self.scene=dict(valid=True);self.plan={};self.index=0;self.phase='transfer';self.maximum_rise=.12;self.history=[];self.grip_trace=[];self.control_result=None;self.control_state=dict(active=False);self.start=object();self.audit_diagnostic=None
            def until(self,predicate,timeout=30.):self_test.assertTrue(predicate())
            def current(self):return NS(episode_id=0,track_id=1)
            def now_ns(self):return 1_000_000_000
            def audit(self):return dict(status='PASS')
            def prepare_trial(self):self.index+=1;self.control_result=None
            def call(self,client,request):
                self.control_result=dict(trial_id=self.index,status='PASS' if self.index==1 else 'FAIL',error='test planning failure',class_decision=dict(model_sha256='test-model'))
                return NS(accepted=True,reason='unit response')
            def verify_sort(self,*args):return dict(status='PASS')
            def reset_episode(self):return dict(status='PASS')
            def stop_after_failure(self):return dict(hold='PASS',paused=True)
        self_test=self
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);pre=p/'pre';pre.mkdir();proof=p/'probe';proof.mkdir();out=p/'out';manifest=installed_manifest()
            (pre/'validation.json').write_text(json.dumps(dict(checkpoint='M3D',status='PASS',installed_manifest=manifest,model_sha256='test-model')))
            (proof/'validation.json').write_text(json.dumps(dict(checkpoint='M3D-carried-planning',status='PASS',successful_plans=40,scene_context='private_fixture_scenes',installed_manifest=manifest)))
            args=['test','--preflight-dir',str(pre),'--planning-probe-dir',str(proof),'--expected-class','red_fruit','--trials','10','--output-dir',str(out)]
            with patch.object(runner,'SortEvaluator',FakeEvaluator),patch.object(runner,'Node',return_value=Mock()),patch.object(runner.rclpy,'init'),patch.object(runner.rclpy,'try_shutdown'),patch.object(runner,'runtime_configuration',return_value=dict(status='PASS')),patch.object(runner,'planning_configuration',return_value=dict(status='PASS')),patch.object(sys,'argv',args),redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(),1)
            report=json.loads((out/'summary.json').read_text());self.assertEqual(report['successful_trials'],1);self.assertEqual(report['completed_trials'],2)
            self.assertEqual(report['trials'][0]['reset']['status'],'PASS');self.assertEqual(report['final_reset_status'],'NOT_RUN');self.assertTrue(report['stopped_early'])
    def test_runtime_old_sampling_or_missing_validator_is_rejected(self):
        config=yaml.safe_load((ROOT/'src/sorting_control/config/carried_planning.yaml').read_text());prefix='ompl.planner_configs.CarriedRRTConnect.'
        values={'ompl.planning_plugins':['sorting_control/CarriedMarginPlanner'],
            'ompl.carried_margin.position_m':.0005,'ompl.carried_margin.orientation_rad':.001,
            'ompl.panda_arm.enforce_joint_model_state_space':True,prefix+'enforce_joint_model_state_space':None,prefix+'range':.02,prefix+'longest_valid_segment_fraction':.0005,'ompl.totg.path_tolerance':.0001,'ompl.totg.resample_dt':.01,'ompl.response_adapters':['default_planning_response_adapters/AddTimeOptimalParameterization','default_planning_response_adapters/ValidateSolution']}
        def observer(override):
            entries=dict(values,**override);client=Mock();client.service_is_ready.return_value=True
            def response(request):
                output=[]
                for key in request.names:
                    want=entries[key];v=ParameterValue()
                    if want is None:v.type=ParameterType.PARAMETER_NOT_SET
                    elif type(want) is bool:v.type=ParameterType.PARAMETER_BOOL;v.bool_value=want
                    elif type(want) is list:v.type=ParameterType.PARAMETER_STRING_ARRAY;v.string_array_value=want
                    else:v.type=ParameterType.PARAMETER_DOUBLE;v.double_value=want
                    output.append(v)
                return NS(values=output)
            client.call_async.side_effect=response;e=NS(node=Mock(),until=lambda f,t:self.assertTrue(f()),future=lambda f,t:f);e.node.create_client.return_value=client;return e
        self.assertEqual(planning_configuration(observer({}))['status'],'PASS')
        for changes in ({'ompl.panda_arm.enforce_joint_model_state_space':False},{prefix+'enforce_joint_model_state_space':True},{'ompl.totg.path_tolerance':.1},{'ompl.response_adapters':[]},{'ompl.planning_plugins':['ompl_interface/OMPLPlanner']},{'ompl.carried_margin.position_m':0.},{'ompl.response_adapters':['default_planning_response_adapters/ValidateSolution','default_planning_response_adapters/AddTimeOptimalParameterization']}):
            with self.assertRaises(RuntimeError):planning_configuration(observer(changes))

if __name__=='__main__':unittest.main()
