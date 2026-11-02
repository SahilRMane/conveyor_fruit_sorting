"""Actual controller methods on a native plant, with mocked ROS transport."""
import copy,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np,yaml
from native_camera_pickup import run
from lift_support_fixtures import hand_matrix
from camera_sort import SortController
from sort_planning import execute_carried_plan
from sort_core import placement_scene
from publish_static_fixtures import scene_message
from ownership_core import NAMES
from m2b_common import timestamp
from rclpy.serialization import serialize_message

class PlanRecorded(RuntimeError):pass
class ClearanceViolation(RuntimeError):pass

def run_probe(library,scene,perception,control,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);records={}
    for mode in ('old_endpoint_replay','completed_endpoint'):
        path=folder/mode;path.mkdir(exist_ok=True)
        def exercise(plant,reference,grasp):
            c=SortController.__new__(SortController)
            c.s=yaml.safe_load((control/'config/m3d.yaml').read_text())
            c.settings=yaml.safe_load((control/'config/m2b.yaml').read_text())
            c.c=yaml.safe_load((control/'config/m3b.yaml').read_text())
            c.scene={'estimated_attachment':True};c.result={};c.phase='transfer_retract';c.last_carried_goal=None
            c.input_enabled=False;c.owner={};c.command_function=None;rows=[];requests=[]
            root=np.eye(4);root[:3,3]=reference['camera_root_xyz']
            relative=np.linalg.inv(hand_matrix(reference))@root
            c.now_ns=lambda:round(plant.feedback()[0]*1e9)
            def hand_pose():
                fb=plant.feedback();q=fb[4:8]
                return fb[1:4].copy(),[float(q[1]),float(q[2]),float(q[3]),float(q[0])]
            c.hand_pose=hand_pose
            def body():
                fb=plant.feedback()
                return hand_matrix(dict(hand_xyz=fb[1:4],hand_quat_wxyz=fb[4:8]))@relative
            def score():
                # Simulator truth enters only this grader, not controller inputs.
                value=plant.truth();rise=value['tray_xyz'][2]-grasp['tray_xyz'][2]
                rows.append(dict(phase=c.phase,rise_m=rise,**value))
                assert value['forbidden_contacts']==0 and sum(value['bin_wall_force_N'])<.1
                assert min(value['left_force_N'],value['right_force_N'])>.2 and max(value['left_force_N'],value['right_force_N'])<15.
                assert abs(np.dot(value['tray_quat_wxyz'],grasp['tray_quat_wxyz']))>np.cos(c.s['maximum_tilt_rad']/2)
                if c.phase=='transfer' and not (rise>c.s['minimum_airborne_transfer_rise_m'] and value['support_force_N']+sum(value['bin_floor_force_N'])<.1):
                    raise ClearanceViolation('Transfer clearance/support condition failed')
            def acquire(mode,*args):
                c.owner=dict(owner=mode,busy=False);c.input_deadline=c.now_ns()+30_000_000_000
                if mode=='SERVO':plant.call('nc_resume')
                else:
                    plant.call('nc_hold_reference')
                    for _ in range(30):plant.call('nc_step',.01,0.);score()
                return 1,timestamp(c.input_deadline)
            c.acquire_mode=acquire
            def step():
                if c.input_enabled:
                    goal,velocity,_=c.command_function(c.now_ns());plant.control(goal,velocity,0.)
                else:plant.call('nc_step',.01,0.)
                c.q=dict(zip(NAMES,plant.feedback()[8:15]));c.q.update(panda_finger_joint1=float(plant.feedback()[15]),panda_finger_joint2=float(plant.feedback()[16]));score()
            def until(predicate,timeout):
                end=c.now_ns()+round(timeout*1e9)
                while not predicate():
                    if c.now_ns()>=end:raise TimeoutError('Native controller endpoint did not settle')
                    step()
            c.until=until
            def wait_sim(seconds):
                end=c.now_ns()+round(seconds*1e9);until(lambda:c.now_ns()>=end,seconds+1.)
            c.wait_sim=wait_sim;c.ensure_oracle_current=lambda:None;c.scene_clients=['servo_scene','planner_scene'];c.planner='planner'
            def call(client,request,**kwargs):
                if client==c.planner:
                    serialize_message(request);requests.append(request);raise PlanRecorded('next plan request captured')
                attached=placement_scene(scene_message(),[0,0,0,0],c.c,True,relative,False).robot_state.attached_collision_objects
                return NS(scene=NS(robot_state=NS(attached_collision_objects=copy.deepcopy(attached))))
            c.call=call
            try:
                p,_=c.hand_pose();world=body();retract=p+[c.s['retract_root_x_m']-world[0,3],-world[1,3],0.]
                c.servo_to(retract.tolist(),6.);old=c.last_carried_goal.copy()
                c.phase='transfer_raise';p,_=c.hand_pose();raise_goal=p+[0,0,c.s['transfer_root_z_m']-body()[2,3]]
                c.servo_to(raise_goal.tolist(),3.)
                if mode=='old_endpoint_replay':c.last_carried_goal=old
                c.phase='transfer';p,_=c.hand_pose();next_goal=p+[0,.04,0]
                try:execute_carried_plan(c,next_goal.tolist(),c.settings['grasp_quaternion_xyzw'],path,'next-transfer')
                except PlanRecorded:pass
                result=dict(status='PASS',planner_request_recorded=bool(requests),hand_goal_xyz=c.last_carried_goal.tolist())
            except ClearanceViolation as error:result=dict(status='FAIL',error=str(error),planner_request_recorded=bool(requests))
            finally:(path/'endpoint-physical-trace.json').write_text(json.dumps(rows,indent=2)+'\n')
            result.update(minimum_transfer_rise_m=min((r['rise_m'] for r in rows if r['phase']=='transfer'),default=None),carried_goal_history=c.result.get('carried_goal_history',[]))
            records[mode]=result;return result
        run(library,scene,perception,control,path,after_pickup=exercise,preserve_hold_reference=True)
    assert records['old_endpoint_replay']['status']=='FAIL' and not records['old_endpoint_replay']['planner_request_recorded'],'Old endpoint did not reproduce the clearance failure'
    assert records['completed_endpoint']['status']=='PASS' and records['completed_endpoint']['planner_request_recorded'],'Completed endpoint did not retain transfer height'
    result=dict(status='PASS',records=records,scope='Actual SortController.servo_to and plan preparation on native free-payload physics; ROS transport mocked, live ROS NOT_RUN')
    (folder/'endpoint-regression.json').write_text(json.dumps(result,indent=2)+'\n');return result
