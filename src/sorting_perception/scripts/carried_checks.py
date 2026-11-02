"""Prepare production carried scenes and bounded requests without a ROS node."""
import json
from pathlib import Path
import numpy as np,yaml
from ament_index_python.packages import get_package_share_directory
from rosidl_runtime_py.set_message import set_message_fields
from rclpy.serialization import serialize_message
from moveit_msgs.msg import RobotState,AttachedCollisionObject,MotionPlanRequest
from trajectory_msgs.msg import JointTrajectory
from publish_static_fixtures import scene_message
from sort_core import placement_scene
from sort_planning import carried_request
from ownership_core import NAMES

def fixtures(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    control=Path(get_package_share_directory('sorting_control'));perception=Path(get_package_share_directory('sorting_perception'))
    config=yaml.safe_load((control/'config/m2b.yaml').read_text());camera=yaml.safe_load((control/'config/m3b.yaml').read_text());sort=yaml.safe_load((control/'config/m3d.yaml').read_text());profile=yaml.safe_load((control/'config/carried_planning.yaml').read_text())
    data=json.loads((perception/'config/m3d_carried_fixtures.json').read_text());records=[]
    for case in data['cases']:
        scene=placement_scene(scene_message(),[0,0,0,0],camera,True,np.asarray(case.get('relative',np.eye(4))),False)
        if 'state' in case:
            state=RobotState();set_message_fields(state,case['state']);scene.robot_state=state;scene.robot_state.is_diff=True
        else:
            scene.robot_state.joint_state.name=NAMES+['panda_finger_joint1','panda_finger_joint2'];scene.robot_state.joint_state.position=case['joint_positions']
        state=scene.robot_state;q=dict(zip(state.joint_state.name,state.joint_state.position))
        request=carried_request(q,case['goal_xyz'],config['grasp_quaternion_xyzw'],config,state.attached_collision_objects,case['start_xyz'],sort['transfer_path_orientation_tolerance_rad'],profile)
        if 'request' in case:
            exact=MotionPlanRequest();set_message_fields(exact,case['request']);request.motion_plan_request=exact
        label=case['label'];record=dict(label=label,scene=str(folder/(label+'-scene.cdr')),request=str(folder/(label+'-request.cdr')))
        Path(record['scene']).write_bytes(serialize_message(scene));Path(record['request']).write_bytes(serialize_message(request.motion_plan_request))
        if 'bad_trajectory' in case:
            bad=JointTrajectory();set_message_fields(bad,case['bad_trajectory']);record['bad']=str(folder/(label+'-bad.cdr'));Path(record['bad']).write_bytes(serialize_message(bad))
        record['native_repeats']=case.get('native_repeats',10)
        records.append(record)
    (folder/'cases.yaml').write_text(yaml.safe_dump(dict(cases=records)))
    return records

def planning_configuration(e):
    from rcl_interfaces.srv import GetParameters
    from rcl_interfaces.msg import ParameterType
    from ownership_core import require
    profile=yaml.safe_load((Path(get_package_share_directory('sorting_control'))/'config/carried_planning.yaml').read_text())
    name='ompl.planner_configs.'+profile['planner_id']
    expected={'ompl.planning_plugins':['sorting_control/CarriedMarginPlanner'],
              'ompl.carried_margin.position_m':.0005,'ompl.carried_margin.orientation_rad':.001,
              'ompl.panda_arm.enforce_joint_model_state_space':True,
              name+'.enforce_joint_model_state_space':None,name+'.range':profile['profile']['range'],
              name+'.longest_valid_segment_fraction':profile['profile']['longest_valid_segment_fraction'],
              'ompl.totg.path_tolerance':profile['totg']['path_tolerance'],'ompl.totg.resample_dt':profile['totg']['resample_dt']}
    client=e.node.create_client(GetParameters,'/m2b_planner/move_group/get_parameters')
    try:
        e.until(client.service_is_ready,5.)
        result=e.future(client.call_async(GetParameters.Request(names=list(expected)+['ompl.response_adapters'])),5.)
        require(len(result.values)==len(expected)+1,'Incomplete carried planner parameter response')
        observed={}
        for (key,want),value in zip(expected.items(),result.values):
            if want is None:ok=value.type==ParameterType.PARAMETER_NOT_SET;actual=None
            elif type(want) is list:ok=value.type==ParameterType.PARAMETER_STRING_ARRAY;actual=list(value.string_array_value)
            elif type(want) is bool:ok=value.type==ParameterType.PARAMETER_BOOL;actual=value.bool_value
            else:ok=value.type==ParameterType.PARAMETER_DOUBLE;actual=value.double_value
            require(ok and actual==want,'Running carried planner parameter mismatch: '+key);observed[key]=actual
        adapters=result.values[-1]
        timing='default_planning_response_adapters/AddTimeOptimalParameterization';validation='default_planning_response_adapters/ValidateSolution'
        require(adapters.type==ParameterType.PARAMETER_STRING_ARRAY and timing in adapters.string_array_value and
            validation in adapters.string_array_value and list(adapters.string_array_value).index(timing)<list(adapters.string_array_value).index(validation),
            'Final MoveIt path validation must follow time parameterization')
        observed['ompl.response_adapters']=list(adapters.string_array_value)
        return dict(status='PASS',parameters=observed)
    finally:e.node.destroy_client(client)
