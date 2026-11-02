"""MoveIt plans with the actual carried model preserved in the start state."""
import copy,json,time
import numpy as np
from action_msgs.msg import GoalStatus
from moveit_msgs.msg import PlanningSceneComponents,OrientationConstraint,PositionConstraint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPlanningScene,GetStateValidity
from sorting_interfaces.action import ExecuteArmTrajectory
from m2b_runtime import plan_request,robot_state,normalize_plan
from ownership_core import NAMES,require,ns,validate_trajectory
from prediction_core import angle_error

def carried_request(q,position,quaternion,settings,attached,start,path_tolerance,profile):
    """Build the same bounded request for runtime and planning-only probes."""
    request=plan_request(q,position,quaternion,settings);request.motion_plan_request.start_state.is_diff=True
    request.motion_plan_request.planner_id=profile["planner_id"]
    request.motion_plan_request.start_state.attached_collision_objects=attached
    # A goal-only orientation constraint permits the payload to tilt mid-path.
    attitude=OrientationConstraint();attitude.header.frame_id='world';attitude.link_name='panda_hand'
    attitude.orientation.x,attitude.orientation.y,attitude.orientation.z,attitude.orientation.w=map(float,quaternion)
    attitude.absolute_x_axis_tolerance=attitude.absolute_y_axis_tolerance=attitude.absolute_z_axis_tolerance=float(path_tolerance);attitude.weight=1.;attitude.parameterization=OrientationConstraint.ROTATION_VECTOR
    request.motion_plan_request.path_constraints.orientation_constraints=[attitude]
    for goal in request.motion_plan_request.goal_constraints:
        for orientation in goal.orientation_constraints:
            orientation.absolute_x_axis_tolerance=orientation.absolute_y_axis_tolerance=orientation.absolute_z_axis_tolerance=.004
    start=np.asarray(start);end=np.asarray(position);lo=np.minimum(start,end)-[.04,.04,.005];hi=np.maximum(start,end)+[.04,.04,.005]
    corridor=PositionConstraint();corridor.header.frame_id='world';corridor.link_name='panda_hand';corridor.weight=1.
    box=SolidPrimitive(type=SolidPrimitive.BOX,dimensions=(hi-lo).tolist());pose=Pose();pose.orientation.w=1.
    pose.position.x,pose.position.y,pose.position.z=map(float,(lo+hi)/2)
    corridor.constraint_region.primitives=[box];corridor.constraint_region.primitive_poses=[pose]
    request.motion_plan_request.path_constraints.position_constraints=[corridor]
    return request

def execute_carried_plan(c,position,quaternion,folder,label):
    from rosidl_runtime_py.convert import message_to_ordereddict
    # Correct compliance at the previous endpoint before forming a new plan.
    # This reads encoder FK and keeps planning within the original tilt bound.
    hand,attitude=c.hand_pose();previous=getattr(c,'last_carried_goal',None)
    displaced=previous is not None and np.linalg.norm(hand-previous)>c.s['placement_position_tolerance_m']
    if displaced or angle_error(attitude,quaternion)>c.s['transfer_start_orientation_tolerance_rad']:
        c.servo_to((hand if previous is None else previous).tolist(),1.5,
                   orientation_tolerance=c.s['transfer_start_orientation_tolerance_rad'])
    c.input_enabled=False;epoch,deadline=c.acquire_mode('PLANNED');c.wait_sim(.2);c.ensure_oracle_current()
    require(angle_error(c.hand_pose()[1],quaternion)<c.s['transfer_start_orientation_tolerance_rad'],
            'Loaded hand orientation changed during the planning handoff')
    query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
    attached=copy.deepcopy(c.call(c.scene_clients[1],query).scene.robot_state.attached_collision_objects)
    from camera_scene import TRAY_IDS
    require({a.object.id for a in attached}==TRAY_IDS and len(attached)==8,'Plan must include all carried tray parts')
    from pathlib import Path
    import yaml
    from ament_index_python.packages import get_package_share_directory
    profile=yaml.safe_load((Path(get_package_share_directory('sorting_control'))/'config/carried_planning.yaml').read_text())
    require(type(profile['maximum_attempts']) is int and 1<=profile['maximum_attempts']<=2,'Carried planning attempts must be 1 or 2')
    begin=c.q.copy();request=carried_request(begin,position,quaternion,c.settings,attached,c.hand_pose()[0],
        c.s['transfer_path_orientation_tolerance_rad'],profile)
    started=time.monotonic();result=None
    for attempt in range(1,int(profile['maximum_attempts'])+1):
        (folder/(label+f'-attempt-{attempt:02d}-request.json')).write_text(json.dumps(message_to_ordereddict(request),indent=2)+'\n')
        response=c.call(c.planner,request,timeout=12.);result=response.motion_plan_response
        saved=json.dumps(message_to_ordereddict(response),indent=2)+'\n'
        (folder/(label+'-moveit-response.json')).write_text(saved)
        (folder/(label+f'-attempt-{attempt:02d}-response.json')).write_text(saved)
        c.result.setdefault('carried_planning_attempts',[]).append(dict(label=label,attempt=attempt,
            code=result.error_code.val,message=result.error_code.message,source=result.error_code.source,
            planner_id=request.motion_plan_request.planner_id,points=len(result.trajectory.joint_trajectory.points)))
        if result.error_code.val==1:break
        retryable=result.error_code.val in (-1,-2,-6,99999)
        if not retryable or attempt==int(profile['maximum_attempts']):break
        c.ensure_oracle_current()
        require(max(abs(c.q[n]-begin[n]) for n in NAMES)<.002,'Carried planning start moved before retry')
        require(c.now_ns()+round((c.settings['planning_time_s']+c.settings['maximum_plan_duration_s']+2.)*1e9)<ns(deadline),
                'Carried planning lease is too short for retry')
        state=robot_state(c.q);state.is_diff=True;state.attached_collision_objects=attached
        require(c.call(c.validity,GetStateValidity.Request(group_name='panda_arm',robot_state=state)).valid,
                'Carried planning start invalid before retry')
    require(result.error_code.val==1,'Carried MoveIt plan failed: '+str(result.error_code.val)+' '+result.error_code.message)
    trajectory=normalize_plan(result.trajectory.joint_trajectory)
    validate_trajectory(trajectory,c.now_ns(),ns(deadline));duration=ns(trajectory.points[-1].time_from_start)/1e9
    require(0<duration<=c.settings['maximum_plan_duration_s'],'Carried plan duration outside bound')
    require(max(abs(c.q[n]-trajectory.points[0].positions[i]) for i,n in enumerate(NAMES))<c.settings['maximum_plan_seed_error_rad'],'Carried plan seed moved')
    state=robot_state(c.q);state.is_diff=True;state.attached_collision_objects=attached
    require(c.call(c.validity,GetStateValidity.Request(group_name='panda_arm',robot_state=state)).valid,'Carried plan start invalid')
    handle=c.future(c.action.send_goal_async(ExecuteArmTrajectory.Goal(epoch=int(epoch),deadline=deadline,trajectory=trajectory)));require(handle.accepted,'Gate rejected carried plan')
    outcome=c.future(handle.get_result_async(),45.);require(outcome.status==GoalStatus.STATUS_SUCCEEDED and outcome.result.success,outcome.result.reason)
    c.wait_sim(.3);p,q=c.hand_pose();require(np.linalg.norm(p-position)<.015 and angle_error(q,quaternion)<.05,'Carried plan endpoint outside compliance bound')
    c.last_carried_goal=np.asarray(position).copy();c.acquire_mode('HOLD');return dict(pipeline='ompl',carried_parts=8,start_state_is_diff=True,points=len(trajectory.points),duration_sim_s=duration,planning_wall_s=time.monotonic()-started)
