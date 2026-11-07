#!/usr/bin/env python3
"""RGB class -> bin -> physical moving pickup and placement; no truth inputs."""
import copy,json,time,hashlib
from pathlib import Path
import numpy as np,yaml,rclpy
from rclpy.node import Node
from control_msgs.action import GripperCommand
from action_msgs.msg import GoalStatus
from sorting_interfaces.msg import ClassObservation,TargetBeliefArray,Decision
from sorting_interfaces.srv import SetCameraPlacement
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from ament_index_python.packages import get_package_share_directory
from camera_pickup import Controller
from wrist_support import WristObserver,SupportWindow,coarse_lower_transition
from camera_scene import TRAY_IDS,SUPPORT_CONTACT_IDS
from camera_scene_checks import collision_allowed
from sort_core import choose_route,ClassWindow
from sort_planning import execute_carried_plan
from ownership_core import require,ns
from prediction_core import angle_error
from m2b_common import timestamp

class SortController(Controller):
    def __init__(self,node):
        self.s=yaml.safe_load((Path(get_package_share_directory('sorting_control'))/'config/m3d.yaml').read_text());self.classes={};self.classified=None;self.route=None;self.last_carried_goal=None
        super().__init__(node);self.settings['pregrasp_orientation_tolerance_rad']=.015
        self.configure_stationary_camera_recovery()
        self.wrist=WristObserver(self);SupportWindow(self.s['supported_release'])
        installed=self.classifier_directory()
        node.declare_parameter('model_directory',str(installed))
        model_dir=Path(node.get_parameter('model_directory').value)
        self.model_sha=hashlib.sha256((model_dir/'weights.npz').read_bytes()).hexdigest()
        require(self.model_sha==hashlib.sha256((installed/'weights.npz').read_bytes()).hexdigest() and (model_dir/'model.json').read_bytes()==(installed/'model.json').read_bytes(),'Use the installed, preflight-validated model')
        classifier_config=yaml.safe_load((Path(get_package_share_directory('sorting_perception'))/'config/m3c.yaml').read_text())
        require(self.s['class_ids']==classifier_config['class_ids'] and self.s['minimum_confidence']>=classifier_config['acceptance_confidence'],'Routing cannot bypass classifier abstention')
        node.create_subscription(ClassObservation,'/sorting/perception/class_observation',self.class_score,10)
        node.create_subscription(TargetBeliefArray,'/sorting/classified_targets',lambda m:setattr(self,'classified',m),1)
        self.decision=node.create_publisher(Decision,'/sorting/decisions',1);self.place_client=node.create_client(SetCameraPlacement,'/sorting/perception/set_camera_placement')
    def classifier_directory(self):
        return Path(get_package_share_directory('sorting_perception'))/'models/m3c_baseline'
    def configure_stationary_camera_recovery(self):
        recovery=self.s['stationary_camera_recovery']
        keys=('resume_max_capture_age_s','maximum_wait_sim_s','maximum_wait_wall_s','maximum_total_wait_wall_s')
        require(all(np.isfinite(recovery[key]) and recovery[key]>0 for key in keys),'Invalid stationary camera recovery limits')
        require(self.c['pregrasp_start_max_capture_age_s']<=recovery['resume_max_capture_age_s']<=self.c['maximum_capture_age_s'],
                'Stationary resume must retain the tracking capture-age limit')
        require(recovery['maximum_total_wait_wall_s']>=recovery['maximum_wait_wall_s'],'Total camera wait is shorter than a single wait')
        self.c.update(stationary_camera_resume_max_capture_age_s=recovery['resume_max_capture_age_s'],
            maximum_stationary_camera_wait_sim_s=recovery['maximum_wait_sim_s'],
            maximum_stationary_camera_wait_wall_s=recovery['maximum_wait_wall_s'],
            maximum_stationary_camera_wait_total_wall_s=recovery['maximum_total_wait_wall_s'])
    def camera_resume_ready(self):
        # Only restart the stopped-belt, open-finger correction. Moving phases
        # still reject old captures in Controller.target(), without this wait.
        self.require_stationary_camera_context()
        stamp=ns(self.belief.header.stamp)
        return (stamp>self.camera_wait['capture_ns'] and
                self.scene.get('episode_id')==self.epoch.episode_id and
                self.scene.get('track_id')==self.belief.track_id and
                self.refinement_ready(self.c['stationary_camera_resume_max_capture_age_s']))
    def class_score(self,m):
        if self.epoch and m.episode_id==self.epoch.episode_id and m.valid:
            self.classes[(m.episode_id,m.view_id,ns(m.header.stamp))]=m
            while len(self.classes)>8:self.classes.pop(next(iter(self.classes)))
    def record_class_pair(self,b,m,status):
        self.result['last_class_pairing']=dict(status=status,phase=self.phase,time_ns=self.now_ns(),
            episode_id=b.episode_id,track_id=b.track_id,view_id=b.view_id,capture_ns=ns(b.header.stamp),
            capture_age_s=(self.now_ns()-ns(b.header.stamp))/1e9,
            target_class_ids=list(b.class_ids),target_probabilities=list(b.class_probabilities),
            score_class_ids=list(m.class_ids) if m is not None else None,
            score_probabilities=list(m.probabilities) if m is not None else None,
            score_model_sha256=m.model_sha256 if m is not None else None)
    def diagnostic(self):
        result=super().diagnostic()
        result['class_pairing']=copy.deepcopy(self.result.get('last_class_pairing'))
        result['wrist_wrench']=copy.deepcopy(self.wrist.last_diagnostic) if hasattr(self,'wrist') else None
        return result
    def select_class(self,required_class=None):
        proof=ClassWindow(self.s);decision=None;identity=None
        def ready():
            nonlocal decision,identity,proof
            if self.classified is None or self.epoch is None:return False
            for b in self.classified.targets:
                if not(b.valid and b.camera_only and b.episode_id==self.epoch.episode_id):continue
                if self.bound_track is not None and b.track_id!=self.bound_track:continue
                if b.view_id!='wrist':continue
                current=(b.episode_id,b.track_id)
                if identity!=current:identity=current;proof=ClassWindow(self.s)
                capture=ns(b.header.stamp);m=self.classes.get((b.episode_id,b.view_id,capture))
                if not -.02<=(self.now_ns()-capture)/1e9<self.s['maximum_capture_age_s']:continue
                if m is None:
                    self.record_class_pair(b,None,'waiting_for_score');continue
                self.record_class_pair(b,m,'checking_pair')
                require(m.model_sha256==self.model_sha,'Classifier model changed')
                # EvidenceStore publishes localisation with empty class fields
                # until this capture's score is available. Cross-topic arrival
                # order can deliver that pending message after the score.
                if not b.class_ids and not b.class_probabilities:
                    self.record_class_pair(b,m,'waiting_for_target_enrichment');continue
                matched=(list(b.class_ids)==list(m.class_ids) and len(b.class_probabilities)==len(m.probabilities) and
                         np.allclose(b.class_probabilities,m.probabilities,atol=1e-12,rtol=0.))
                self.record_class_pair(b,m,'matched' if matched else 'mismatch')
                require(matched,'Class/target evidence mismatch')
                value=choose_route(b.class_ids,b.class_probabilities,self.s)
                if required_class is not None:require(value['class_id']==required_class and value['action']=='PICK_PLACE','Class evidence changed at pregrasp')
                if proof.observe(capture,value):decision=dict(value,capture_ns=capture,episode_id=b.episode_id,track_id=b.track_id,model_sha256=m.model_sha256,distinct_captures=len(proof.samples));return True
            return False
        self.until(ready,self.s['class_wait_wall_s']);return decision
    def publish_decision(self,route):
        msg=Decision();msg.header.frame_id='world';msg.header.stamp=self.node.get_clock().now().to_msg();msg.episode_id=route['episode_id'];msg.track_id=route['track_id'];msg.action=route['action'];msg.selected_view='wrist';msg.class_id=route['class_id'];msg.proposed_bin=route['bin_id'];msg.confidence=route['confidence'];msg.class_probabilities=route['probabilities'];msg.evidence_capture_time=timestamp(route['capture_ns']);msg.model_sha256=self.model_sha;msg.reason=route['reason'];msg.estimated_service_time_s=self.s['estimated_full_service_s']
        x,cov,_=self.target();speed=max(self.c['belt_speed_m_s'],float(x[2]+2*np.sqrt(cov[2,2])),.001)
        remaining=min(self.s['maximum_exit_estimate_s'],max(0.,(self.settings['safe_x_interval_m'][1]-x[0]-2*np.sqrt(cov[0,0]))/speed))
        msg.remaining_time_s=float(remaining);msg.deadline=timestamp(self.now_ns()+round(remaining*1e9))
        if route['action']=='PICK_PLACE':require(remaining>self.c['descend_duration_s']+self.settings['gripper_closure_estimate_s']+self.c['deadline_reserve_s'],'Insufficient camera-estimated interception time')
        self.decision.publish(msg)
    def before_tracking(self):
        self.wrist.calibrate()
        self.route=self.select_class(self.route['class_id']);self.result['class_decision']=copy.deepcopy(self.route);self.publish_decision(self.route)
    def body(self):
        self.ensure_oracle_current();value=self.scene.get('estimated_body_matrix');require(value is not None,'Estimated carried body unavailable');return np.asarray(value)
    def servo_to(self,goal,duration=2.,hold=True,reuse=False,orientation_tolerance=None,lease_s=30.):
        if reuse:
            require(self.owner.get('owner')=='SERVO' and self.input_deadline>self.now_ns()+round((duration+8.)*1e9),'Release Servo lease is too short for retreat')
        else:self.input_enabled=False;self.acquire_mode('SERVO',lease_s,1)
        tolerance=self.s['placement_orientation_tolerance_rad'] if orientation_tolerance is None else orientation_tolerance
        p0,_=self.hand_pose();start=self.now_ns();orientation=self.settings['grasp_quaternion_xyzw']
        from grasp_core import smooth
        def reference(now):
            t=(now-start)/1e9;s,ds=smooth(t,duration);delta=np.asarray(goal)-p0
            return (p0+delta*s).tolist(),(delta*ds).tolist(),orientation
        self.command_function=reference;self.input_enabled=True
        self.until(lambda:self.now_ns()-start>duration*1e9 and np.linalg.norm(self.hand_pose()[0]-goal)<self.s['placement_position_tolerance_m'] and angle_error(self.hand_pose()[1],orientation)<tolerance,20.)
        self.wait_sim(.3)
        if hold:self.input_enabled=False;self.acquire_mode('HOLD')
        if self.scene.get('estimated_attachment'):
            # A Servo raise replaces the earlier planned retract endpoint.
            # The next plan must correct towards this completed goal.
            self.last_carried_goal=np.asarray(goal,dtype=float).copy()
            self.result.setdefault('carried_goal_history',[]).append(dict(
                phase=self.phase,time_ns=self.now_ns(),hand_goal_xyz=self.last_carried_goal.tolist(),source='completed_servo'))
    def place_service(self,operation):
        response=self.call(self.place_client,SetCameraPlacement.Request(episode_id=self.epoch.episode_id,owner_epoch=self.owner['epoch'],bin_id=self.route['bin_id'],operation=operation));require(response.success,response.reason)
    def approach_bin_with_wrist(self,goal,orientation):
        self.wrist.measure_loaded()
        self.input_enabled=False;self.acquire_mode('SERVO',self.s['release_motion_lease_s'],1)
        start_pose,_=self.hand_pose();begin=self.now_ns();duration=self.s['lower_duration_s']
        from grasp_core import smooth
        def reference(now):
            blend,ds=smooth((now-begin)/1e9,duration);delta=np.asarray(goal)-start_pose
            return (start_pose+delta*blend).tolist(),(delta*ds).tolist(),orientation
        self.command_function=reference;self.input_enabled=True
        def reached_or_contact():
            stamp,payload=self.wrist.payload();position,_=self.hand_pose()
            transition=coarse_lower_transition(position,goal,(self.now_ns()-begin)/1e9,duration,payload,
                self.wrist.loaded,self.s['placement_position_tolerance_m'])
            self.result['coarse_lower_transition']=dict(transition,time_ns=self.now_ns(),wrench_stamp_ns=stamp)
            return transition['ready']
        self.until(reached_or_contact,20.)
        self.lower_to_wrist_support(orientation)

    def lower_to_wrist_support(self,orientation):
        settings=self.s['supported_release'];window=SupportWindow(settings)
        loaded=self.wrist.loaded;require(loaded is not None,'Wrist payload calibration missing');self.result['wrist_loaded_payload_N']=loaded
        start_pose,_=self.hand_pose();begin=self.now_ns();started_wall=time.monotonic()
        self.result['wrist_approach_start']=dict(time_ns=begin,hand_xyz=start_pose.tolist(),
            transition=copy.deepcopy(self.result.get('coarse_lower_transition')))
        def reference(now):
            elapsed=(now-begin)/1e9;p,_,_,age,velocity=self.sample_hand(now)
            corrected=p+velocity*max(age,0.) if velocity is not None else p
            distance=abs(float(corrected[2]-start_pose[2]))
            self.result['wrist_approach_progress']=dict(time_ns=now,elapsed_sim_s=elapsed,distance_m=distance,
                maximum_sim_s=settings['maximum_approach_sim_s'],maximum_distance_m=settings['maximum_approach_distance_m'])
            require(self.phase=='release' or elapsed<settings['maximum_approach_sim_s'],'Wrist support approach exceeded time bound')
            require(distance<settings['maximum_approach_distance_m'],'Wrist support approach exceeded distance bound')
            _,payload=self.wrist.payload()
            speed=-float(np.clip(settings['force_velocity_gain_m_per_N_s']*payload,-settings['approach_speed_m_s'],settings['approach_speed_m_s']))
            return [float(start_pose[0]),float(start_pose[1]),float(corrected[2])],[0.,0.,speed],orientation
        self.command_function=reference;self.input_enabled=True;self.result['wrist_support_trace']=[]
        def supported():
            require(time.monotonic()-started_wall<settings['maximum_approach_wall_s'],'Wrist support approach wall timeout')
            stamp,payload=self.wrist.payload();ready=window.observe(stamp,payload,loaded)
            self.result['wrist_support_trace'].append(dict(time_ns=self.now_ns(),wrench_stamp_ns=stamp,payload_N=payload,loaded_N=loaded,ready=ready))
            return ready
        self.until(supported,settings['maximum_approach_wall_s'])
        # Keep wrist load regulation during opening; it prevents a force handoff.
        self.wait_sim(.1);self.place_service('register_floor_support')
        self.result['wrist_floor_support']=dict(status='PASS',wrench_stamp_ns=window.last,samples=window.count,
            span_s=(window.last-window.first)/1e9,source='physical_wrist_wrench',force_threshold_N=settings['support_fraction']*loaded)

    def release_hand_step(self,position,unload=False):
        require(self.phase=='release' and self.input_enabled and self.owner.get('owner')=='SERVO' and
                not self.owner.get('busy') and self.now_ns()<self.input_deadline,'Release requires active Servo pose support')
        goal=GripperCommand.Goal();goal.command.position=float(position);goal.command.max_effort=20.
        self.result.setdefault('release_command_starts',[]).append(dict(time_ns=self.now_ns(),
            position_command_m=float(position),stage='unload' if unload else 'open'))
        action=self.future(self.hand.send_goal_async(goal));require(action.accepted,'Gripper rejected release')
        outcome=self.future(action.get_result_async(),10.)
        require(outcome.status==GoalStatus.STATUS_SUCCEEDED and
                (outcome.result.reached_goal or (unload and outcome.result.stalled)),
                'Gripper did not complete release command')
        self.result.setdefault('release_commands',[]).append(dict(time_ns=self.now_ns(),position_command_m=float(position),
            stage='unload' if unload else 'open',stalled=bool(outcome.result.stalled),reached_goal=bool(outcome.result.reached_goal)))
        self.wait_sim(self.s['release_unload_settle_s'] if unload else .15)
    def pickup(self,folder):
        self.reset_joint_guard()
        self.active=True;self.cleanup=False;self.abort=False;self.target_required=True;self.bound_track=None;self.grip_candidate=False
        self.events=[];self.trace=[];self.closure_trace=[];self.camera_wait_trace=[];self.camera_wait_count=0;self.camera_wait_total_wall_s=0.
        self.result=dict(status='FAIL',trial_id=self.pending.trial_id,episode_id=self.epoch.episode_id,policy_completed=False)
        self.route=None;self.last_carried_goal=None;self.set_phase('classify_initial');self.route=self.select_class();self.publish_decision(self.route)
        if self.route['action']=='PASS':
            require(self.owner.get('owner')=='HOLD' and not self.owner.get('busy') and all(self.q.get(n,0)>.039 for n in ('panda_finger_joint1','panda_finger_joint2')),'PASS requires open fingers and exclusive HOLD')
            self.active=True;self.result=dict(status='PASS',trial_id=self.pending.trial_id,episode_id=self.epoch.episode_id,action='PASS',class_decision=self.route,camera_only=True,oracle_input=False,policy_completed=True,checks={'low_confidence_no_grasp':'PASS'})
            self.set_phase('pass_supported');self.belt_speed=self.c['belt_speed_m_s'];self.send_belt();self.wait_sim(self.s['pass_transport_s'])
            self.belt_speed=0.;self.send_belt();self.wait_sim(.3)
            require(self.owner.get('owner')=='HOLD' and not self.input_enabled,'PASS must keep exclusive HOLD');return self.result
        super().pickup(folder);self.result['status']='FAIL';self.result['policy_completed']=False;self.result['action']='PICK_PLACE';self.result['pickup_complete_ns']=self.now_ns()
        self.belt_speed=0.;self.send_belt();self.target_required=False;orientation=self.settings['grasp_quaternion_xyzw'];self.result['transfer_plans']=[]
        self.set_phase('transfer_retract');body=self.body();hand,_=self.hand_pose()
        retract=hand+np.array([self.s['retract_root_x_m']-body[0,3],-body[1,3],0.])
        self.result['transfer_plans'].append(execute_carried_plan(self,retract.tolist(),orientation,folder,'transfer-retract'))
        self.set_phase('transfer_raise');body=self.body();p,_=self.hand_pose();goal=p+np.array([0,0,self.s['transfer_root_z_m']-body[2,3]])
        self.servo_to(goal.tolist(),3.)
        destination=np.asarray(self.s['bins'][self.route['bin_id']]['center']);body=self.body();p,_=self.hand_pose();hover=p+np.array([destination[0]-body[0,3],destination[1]-body[1,3],self.s['transfer_root_z_m']-body[2,3]])
        self.set_phase('transfer');steps=max(1,int(np.ceil(np.linalg.norm(hover-p)/self.s['transfer_segment_m'])))
        for i in range(1,steps+1):
            point=p+(hover-p)*i/steps;self.result['transfer_plans'].append(execute_carried_plan(self,point.tolist(),orientation,folder,f'transfer-{i:02d}'))
        self.servo_to(hover.tolist(),2.)
        self.place_service('enable_floor')
        for client in self.scene_clients:
            query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
            def committed():return all(collision_allowed(self.call(client,query).scene.allowed_collision_matrix,part,self.route['bin_id']+'_floor') for part in SUPPORT_CONTACT_IDS)
            self.until(committed,5.)
        release_destination=destination.copy();release_destination[2]=self.s['supported_release']['approach_start_root_z_m']
        self.set_phase('lower');body=self.body();hand,_=self.hand_pose();target=hand+(release_destination-body[:3,3])
        self.approach_bin_with_wrist(target.tolist(),orientation)
        self.set_phase('release')
        for position in self.s['supported_release']['supported_unload_positions_m']:self.release_hand_step(position,unload=True)
        self.until(lambda:self.wrist.payload()[1]<self.s['supported_release']['maximum_open_payload_fraction']*self.wrist.loaded,5.)
        self.result['wrist_unloaded_before_open']='PASS'
        for gap in self.s['release_finger_positions_m']:
            self.release_hand_step(gap)
        self.wait_sim(.3);self.place_service('detach')
        for client in self.scene_clients:
            query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS|PlanningSceneComponents.WORLD_OBJECT_NAMES
            def detached():
                scene=self.call(client,query).scene
                return not any(a.object.id in TRAY_IDS for a in scene.robot_state.attached_collision_objects) and TRAY_IDS<={o.id for o in scene.world.collision_objects}
            self.until(detached,5.)
        self.set_phase('retreat');start_pose,_=self.hand_pose();self.servo_to((start_pose+[0.,0.,.10]).tolist(),3.,reuse=False);self.set_phase('verify_placement');self.wait_sim(self.s['placement_settle_s'])
        require(self.owner.get('owner')=='HOLD' and self.owner['stats']['overlap_violations']==0,'Placement must end in exclusive HOLD')
        self.result.update(status='PASS',policy_completed=True,physical_outcome='EVALUATOR_REQUIRED');self.result['checks'].update(class_driven_bin='PASS',planned_carried_transfer='PASS',release_and_retreat='PASS',exclusive_final_hold='PASS');return self.result

def main():
    from datetime import datetime
    rclpy.init();node=Node('camera_sort_controller');c=SortController(node);folder=Path('results')/f'M3D-controller-{datetime.now():%Y%m%d-%H%M%S}';folder.mkdir(parents=True,exist_ok=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.02)
            if c.pending is not None:
                trial=folder/f'trial-{c.pending.trial_id}';trial.mkdir(exist_ok=True);c.run_pending(trial)
    except KeyboardInterrupt:pass
    finally:c.input_enabled=False;c.belt_speed=0.;c.send_belt();node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
