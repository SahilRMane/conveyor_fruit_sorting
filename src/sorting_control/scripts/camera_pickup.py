#!/usr/bin/env python3
"""Camera-only pickup controller. Physical outcome is evaluated elsewhere."""
import copy
import json
from pathlib import Path
import sys
import time
import math
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock,ClockType
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from ament_index_python.packages import get_package_share_directory,get_package_prefix
from tf2_ros import Buffer,TransformListener
from sensor_msgs.msg import JointState
from std_msgs.msg import String,Float64
from std_srvs.srv import Trigger
from control_msgs.action import GripperCommand
from action_msgs.msg import GoalStatus
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetMotionPlan,GetPositionFK,GetStateValidity,GetPlanningScene
from sorting_interfaces.msg import TargetBeliefArray,CameraEpoch,ArmServoCommand
from sorting_interfaces.srv import SetArmOwner,SetHandleContact,SetEstimatedGrasp,StartCameraPickup
from sorting_interfaces.action import ExecuteArmTrajectory
from ownership_core import NAMES,ns,require
from m2b_runtime import M2BProbe
from m2b_common import xyz,xyzw,timestamp
from prediction_core import angle_error,tracking_twist
from grasp_core import VelocityWindow,smooth,braking_x
from camera_grasp_core import CameraClosureWindow,ClosureProof,FingerWindow,predict_belief

class CameraCaptureWait(ValueError):
    """A new capture can resolve this; malformed data and identity errors cannot."""

class Controller:
    # Reuse the passed planning/arbiter functions, without constructing the
    # M2 oracle observer or any of its subscriptions.
    call=M2BProbe.call;acquire_mode=M2BProbe.acquire_mode;execute_plan=M2BProbe.execute_plan
    def __init__(self,node):
        self.node=node;share=Path(get_package_share_directory("sorting_control"))
        self.c=yaml.safe_load((share/"config/m3b.yaml").read_text())
        self.settings=yaml.safe_load((share/"config/m2b.yaml").read_text())
        self.fc=self.c
        self.settings.update(maximum_linear_speed_m_s=self.c["maximum_servo_linear_speed_m_s"],maximum_angular_speed_rad_s=self.c["maximum_servo_angular_speed_rad_s"])
        self.q={};self.v={};self.q_stamp_ns=0;self.previous_q=None;self.owner={};self.epoch=None;self.belief=None;self.received=0.
        self.scene={};self.active=False;self.pending=None;self.abort=False;self.cleanup=False;self.phase="idle"
        self.camera_wait=None;self.camera_wait_trace=[];self.camera_wait_count=0;self.camera_wait_total_wall_s=0.
        self.input_enabled=False;self.input_epoch=0;self.input_deadline=0;self.command_function=None;self.belt_speed=0.
        self.target_required=True;self.grip_candidate=False;self.bound_track=None;self.velocity_window=VelocityWindow()
        self.trace=[];self.closure_proof=None;self.closure_trace=[];self.tracking_start=0;self.motion_lease=0;self.last_tracking={};self.events=[];self.result={};self.maximum_joint_speed=0.;self.maximum_joint_step=0.
        self.buffer=Buffer(cache_time=Duration(seconds=20.));self.listener=TransformListener(self.buffer,node)
        durable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(CameraEpoch,"/sorting/episode",self.receive_epoch,durable)
        node.create_subscription(TargetBeliefArray,"/sorting/targets",self.receive_belief,1)
        node.create_subscription(JointState,"/joint_states",self.receive_joints,QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
        node.create_subscription(String,"/sorting/arm/owner_state",lambda m:setattr(self,"owner",json.loads(m.data)),1)
        node.create_subscription(String,"/sorting/perception/grasp_scene_state",lambda m:setattr(self,"scene",json.loads(m.data)),1)
        self.service=node.create_client(SetArmOwner,"/sorting/arm/set_owner")
        self.action=ActionClient(node,ExecuteArmTrajectory,"/sorting/arm/execute_trajectory")
        self.hand=ActionClient(node,GripperCommand,"/panda_hand_controller/gripper_cmd")
        self.planner=node.create_client(GetMotionPlan,"/m2b_planner/plan_kinematic_path")
        self.planner_params=node.create_client(GetParameters,"/m2b_planner/move_group/get_parameters")
        self.fk_client=node.create_client(GetPositionFK,"/m2b_planner/compute_fk")
        self.validity=node.create_client(GetStateValidity,"/m2b_planner/check_state_validity")
        self.scene_clients=[node.create_client(GetPlanningScene,p) for p in ("/get_planning_scene","/m2b_planner/get_planning_scene")]
        self.contact_service=node.create_client(SetHandleContact,"/sorting/perception/set_handle_contact")
        self.grasp_service=node.create_client(SetEstimatedGrasp,"/sorting/perception/set_estimated_grasp")
        self.publisher=node.create_publisher(ArmServoCommand,"/sorting/arm/servo_command",1)
        self.belt=node.create_publisher(Float64,"/sorting/conveyor_command",1)
        self.state_pub=node.create_publisher(String,"/sorting/control/pickup_state",1)
        self.result_pub=node.create_publisher(String,"/sorting/control/pickup_result",durable)
        node.create_service(StartCameraPickup,"/sorting/control/start_camera_pickup",self.start)
        node.create_service(Trigger,"/sorting/control/abort_camera_pickup",self.stop_request)
        wall=Clock(clock_type=ClockType.STEADY_TIME)
        node.create_timer(.02,self.send,clock=wall);node.create_timer(.1,self.publish_state,clock=wall)
        node.create_timer(.1,self.send_belt,clock=wall)
    def now_ns(self):return self.node.get_clock().now().nanoseconds
    def reset_joint_guard(self):
        # Compare samples within one trial. Reset motion occurs while inactive.
        require(not self.active,"Cannot reset the joint guard during an active trial")
        self.previous_q=None;self.previous_q_stamp_ns=None
        self.maximum_joint_speed=0.;self.maximum_joint_step=0.;self.joint_feedback_diagnostic=None
    def receive_epoch(self,m):
        if self.epoch is not None and m.episode_id<=self.epoch.episode_id:return
        if self.active:self.abort=True
        else:self.reset_joint_guard()
        self.epoch=m;self.belief=None;self.bound_track=None;self.grip_candidate=False
    def receive_belief(self,m):
        if self.epoch is None or m.episode_id!=self.epoch.episode_id:return
        targets=[b for b in m.targets if b.valid and b.camera_only]
        if targets:self.belief=targets[0];self.received=time.monotonic()
    def receive_joints(self,m):
        stamp=ns(m.header.stamp)
        if stamp<=self.q_stamp_ns:return
        self.q=dict(zip(m.name,m.position));self.v=dict(zip(m.name,m.velocity));self.q_stamp_ns=stamp
        if not self.active or self.cleanup:return
        require(all(n in self.q and n in self.v for n in NAMES),"Incomplete joint feedback")
        values=[self.q[n] for n in NAMES];velocities=[self.v[n] for n in NAMES]
        require(all(math.isfinite(x) for x in values+velocities),"Invalid joint feedback")
        speed=max(abs(x) for x in velocities);self.maximum_joint_speed=max(self.maximum_joint_speed,speed)
        step=max(abs(a-b) for a,b in zip(values,self.previous_q)) if self.previous_q is not None else None
        self.joint_feedback_diagnostic=dict(names=list(NAMES),stamp_ns=stamp,
            previous_stamp_ns=getattr(self,'previous_q_stamp_ns',None),positions_rad=values,
            previous_positions_rad=copy.deepcopy(self.previous_q),maximum_step_rad=step,
            maximum_speed_rad_s=speed,step_limit_rad=.02,speed_limit_rad_s=self.c["maximum_joint_speed_rad_s"])
        require(speed<self.c["maximum_joint_speed_rad_s"],"Camera pickup joint speed bound")
        if self.previous_q is not None:
            self.maximum_joint_step=max(self.maximum_joint_step,step)
            require(step<.02,"Camera pickup joint feedback step")
        self.previous_q=values;self.previous_q_stamp_ns=stamp
    def target(self,at_ns=None):
        require(self.epoch is not None and self.belief is not None and self.belief.episode_id==self.epoch.episode_id,"No current camera target")
        if self.bound_track is not None:require(self.belief.track_id==self.bound_track,"Camera track identity changed during pickup")
        require(time.monotonic()-self.received<self.c["maximum_belief_wall_age_s"],"Camera belief source stopped")
        require(abs(self.belief.fixed_handle_z-self.fc["fixed_handle_z_m"])<1e-8,"Camera and controller height priors differ")
        require(self.belief.valid and self.belief.camera_only and self.belief.header.frame_id=="world","Invalid camera belief flags or frame")
        decision=self.now_ns() if at_ns is None else at_ns
        age=(decision-ns(self.belief.header.stamp))/1e9
        if age>=self.c["maximum_capture_age_s"]:
            raise CameraCaptureWait(f"Camera capture age {age:.3f} s exceeds {self.c['maximum_capture_age_s']:.3f} s")
        return predict_belief(self.belief,decision,self.fc)
    def ensure_oracle_current(self):
        # Adapter name required by the unchanged M2 planning helper. Only
        # camera beliefs are checked here; no oracle object or topic exists.
        def dependencies():
            if self.target_required:self.target()
            if (not self.scene.get("valid") and self.scene.get("camera_only") and self.scene.get("reason")=="Camera belief is invalid, stale or future" and
                    (self.scene.get("capture_age_s") or 0.)>=self.c["maximum_capture_age_s"]):
                raise CameraCaptureWait("Camera collision geometry awaits a fresh capture")
            require(self.scene.get("valid") and self.scene.get("camera_only"),"Camera-derived collision scene is invalid: "+str(self.scene.get("reason")))
            require(-.02<=(self.now_ns()-self.scene.get("time_ns",0))/1e9<.15,"Camera collision scene source stopped")
        if not self.input_enabled:
            def ready():
                try:dependencies();return True
                except (ValueError,RuntimeError) as error:
                    if any(text in str(error) for text in ("identity changed","priors differ")):raise
                    return False
            self.until(ready,5.)
        dependencies()
    def refinement_ready(self,maximum_capture_age_s=None):
        x,p,stamp=self.target()
        limit=self.c["pregrasp_start_max_capture_age_s"] if maximum_capture_age_s is None else maximum_capture_age_s
        require(0<limit<=self.c["maximum_capture_age_s"],"Refinement age limit exceeds the tracking limit")
        return ((self.now_ns()-stamp)/1e9<limit and
                self.scene.get("valid") and self.scene.get("camera_only") and
                -.02<=(self.now_ns()-self.scene.get("time_ns",0))/1e9<.05 and
                self.scene.get("model_capture_ns",0)>=stamp)
    def camera_resume_ready(self):
        # Earlier checkpoints keep their original restart rule.
        return self.refinement_ready()
    def wait_refinement_ready(self):
        require(not self.input_enabled and self.belt_speed==0.,"Refinement wait requires stopped belt and disabled input")
        def ready():
            try:return self.refinement_ready()
            except (ValueError,RuntimeError) as error:
                if any(text in str(error) for text in ("identity changed","priors differ")):raise
                return False
        self.until(ready,10.)
    def refine_pose(self,position,quaternion):
        self.set_phase("pregrasp_refinement_wait")
        self.acquire_mode("SERVO",seconds=20.,kind=1)
        # Wait after handoff. A pre-handoff camera check can age during the stop.
        self.wait_refinement_ready()
        self.command_function=lambda now:(position,[0.,0.,0.],quaternion)
        self.set_phase("pregrasp_refinement");self.input_enabled=True;start=self.now_ns()
        self.until(lambda:self.camera_wait is None and self.now_ns()-start>300_000_000 and
            np.linalg.norm(self.hand_pose()[0]-np.asarray(position))<self.settings["pregrasp_position_tolerance_m"] and
            angle_error(self.hand_pose()[1],quaternion)<self.settings["pregrasp_orientation_tolerance_rad"],10.)
        self.wait_sim(.2)
        self.until(lambda:self.camera_wait is None,2.)
        p,q=self.hand_pose()
        self.input_enabled=False;self.acquire_mode("HOLD")
        return dict(position_error_m=float(np.linalg.norm(p-np.asarray(position))),orientation_error_rad=angle_error(q,quaternion),
                    post_handoff_camera_wait=True)
    def hand_pose(self,stamp=None):
        at=Time(nanoseconds=stamp) if stamp is not None else Time()
        tf=self.buffer.lookup_transform("world","panda_hand",at)
        require(stamp is not None or -.02<=(self.now_ns()-ns(tf.header.stamp))/1e9<self.c["maximum_tf_age_s"],"Hand TF is stale")
        return np.asarray(xyz(tf.transform.translation)),xyzw(tf.transform.rotation)
    def sample_hand(self,at_ns=None):
        tf=self.buffer.lookup_transform("world","panda_hand",Time());stamp=ns(tf.header.stamp);age=((self.now_ns() if at_ns is None else at_ns)-stamp)/1e9
        require(-.02<=age<self.c["maximum_tf_age_s"],"Hand TF age")
        p=np.asarray(xyz(tf.transform.translation));velocity=self.velocity_window.observe(stamp,p)
        return p,xyzw(tf.transform.rotation),stamp,age,velocity
    def until(self,predicate,timeout=30.):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            rclpy.spin_once(self.node,timeout_sec=.01)
            if self.active and not self.cleanup:
                require(not self.abort,"Camera pickup aborted")
                require(self.owner.get("owner")!="FAULT","Arm gate fault: "+self.owner.get("reason",""))
                if self.input_enabled:require(self.owner.get("owner")=="SERVO" and not self.owner.get("busy"),"Servo ownership was stopped during "+self.phase+": "+str(self.owner.get("reason")))
            if predicate():return
        raise TimeoutError("Camera pickup operation timed out in "+self.phase)
    def future(self,future,timeout=30.):self.until(future.done,timeout);return future.result()
    def wait_sim(self,seconds):
        start=self.now_ns();self.until(lambda:self.now_ns()-start>=round(seconds*1e9))
    def publish_state(self):
        self.state_pub.publish(String(data=json.dumps(dict(phase=self.phase,active=self.active,
            episode_id=self.epoch.episode_id if self.epoch else None,track_id=self.bound_track,grip_candidate=self.grip_candidate,
            camera_only=True,belt_speed_m_s=self.belt_speed,decision_time_ns=self.now_ns()))))
    def set_phase(self,phase):self.phase=phase;self.events.append(dict(phase=phase,time_ns=self.now_ns()));self.publish_state()
    def send_belt(self):self.belt.publish(Float64(data=float(self.belt_speed)))
    def check_camera_wait_bounds(self,now,wall):
        elapsed_wall=wall-self.camera_wait["start_wall"];elapsed_sim=(now-self.camera_wait["start_ns"])/1e9
        require(0<=elapsed_sim<self.c["maximum_stationary_camera_wait_sim_s"] and
                0<=elapsed_wall<self.c["maximum_stationary_camera_wait_wall_s"] and
                self.camera_wait_total_wall_s+elapsed_wall<self.c["maximum_stationary_camera_wait_total_wall_s"],
                f"Stationary camera wait exceeded its bound: sim={elapsed_sim:.3f}/{self.c['maximum_stationary_camera_wait_sim_s']:.3f} s, "
                f"wall={elapsed_wall:.3f}/{self.c['maximum_stationary_camera_wait_wall_s']:.3f} s, "
                f"total_wall={self.camera_wait_total_wall_s+elapsed_wall:.3f}/{self.c['maximum_stationary_camera_wait_total_wall_s']:.3f} s")
        require(self.epoch.episode_id==self.camera_wait["episode_id"] and self.belief.track_id==self.camera_wait["track_id"],
                "Camera identity changed during stationary wait")
        return elapsed_wall,elapsed_sim
    def require_stationary_camera_context(self):
        require(self.phase=="pregrasp_refinement" and self.belt_speed==0. and self.target_required and not self.grip_candidate,
                "Camera wait is allowed only during stationary open-finger pregrasp")
        require(all(self.q.get(n,0.)>.039 for n in ("panda_finger_joint1","panda_finger_joint2")),"Camera wait requires open fingers")
        require(not self.scene.get("estimated_attachment"),"Camera wait cannot hold an estimated payload")
        require(self.owner.get("owner")=="SERVO" and not self.owner.get("busy") and self.owner.get("epoch")==self.input_epoch,
                "Camera wait requires the current SERVO owner")
        require(self.now_ns()<self.input_deadline,"Camera wait motion lease expired")
    def send_stationary_camera_brake(self,reason):
        # No camera prediction or nonzero reference is used while data is old.
        self.require_stationary_camera_context();now=self.now_ns();wall=time.monotonic()
        self.sample_hand(now) # Keep the unchanged measured-hand freshness check.
        if self.camera_wait is None:
            self.camera_wait_count+=1
            require(self.camera_wait_count<=self.c["maximum_stationary_camera_waits"],"Too many stationary camera waits")
            self.camera_wait=dict(start_ns=now,start_wall=wall,reason=str(reason),episode_id=self.epoch.episode_id,track_id=self.belief.track_id,
                                  capture_ns=ns(self.belief.header.stamp))
            self.events.append(dict(phase="stationary_camera_wait",time_ns=now,reason=str(reason)))
        elapsed_wall,elapsed_sim=self.check_camera_wait_bounds(now,wall)
        message=ArmServoCommand(epoch=int(self.input_epoch),deadline=timestamp(self.input_deadline),command_type=ArmServoCommand.TWIST)
        message.twist.header.frame_id="world";message.twist.header.stamp=self.node.get_clock().now().to_msg()
        self.publisher.publish(message) # All six twist components remain zero.
        self.camera_wait_trace.append(dict(time_ns=now,capture_ns=ns(self.belief.header.stamp) if self.belief else None,
            elapsed_wall_s=elapsed_wall,elapsed_sim_s=elapsed_sim,reason=str(reason),zero_twist=True,phase=self.phase,
            scene=copy.deepcopy(self.scene),capture_age_s=(now-ns(self.belief.header.stamp))/1e9,
            start_max_capture_age_s=self.c['pregrasp_start_max_capture_age_s'],
            resume_max_capture_age_s=self.c.get('stationary_camera_resume_max_capture_age_s',self.c['pregrasp_start_max_capture_age_s'])))
    def send(self):
        if not self.input_enabled:return
        try:
            self.ensure_oracle_current()
            if self.camera_wait is not None and not self.camera_resume_ready():raise CameraCaptureWait("Waiting for a fresh capture and matching camera scene")
        except CameraCaptureWait as error:
            if self.phase!="pregrasp_refinement":raise
            self.send_stationary_camera_brake(error);return
        if self.camera_wait is not None:
            self.require_stationary_camera_context()
            elapsed_wall,_=self.check_camera_wait_bounds(self.now_ns(),time.monotonic())
            self.camera_wait_total_wall_s+=elapsed_wall
            self.events.append(dict(phase="stationary_camera_resume",time_ns=self.now_ns(),capture_ns=ns(self.belief.header.stamp)))
            self.camera_wait=None;self.velocity_window=VelocityWindow()
        decision_ns=self.now_ns();p,q,stamp,age,velocity=self.sample_hand(decision_ns)
        corrected=p+velocity*max(age,0) if velocity is not None else p
        target,ff,orientation=self.command_function(decision_ns)
        linear,angular=tracking_twist(target,ff,orientation,corrected,q,self.settings)
        message=ArmServoCommand(epoch=int(self.input_epoch),deadline=timestamp(self.input_deadline),command_type=ArmServoCommand.TWIST)
        message.twist.header.frame_id="world";message.twist.header.stamp=self.node.get_clock().now().to_msg()
        message.twist.twist.linear.x,message.twist.twist.linear.y,message.twist.twist.linear.z=map(float,linear)
        message.twist.twist.angular.x,message.twist.twist.angular.y,message.twist.twist.angular.z=map(float,angular)
        self.last_tracking=dict(time_ns=decision_ns,tf_stamp_ns=stamp,tf_age_s=age,hand_xyz=corrected.tolist(),target_xyz=list(map(float,target)),
            camera_capture_ns=ns(self.belief.header.stamp) if self.belief else None,
            camera_capture_age_s=(decision_ns-ns(self.belief.header.stamp))/1e9 if self.belief else None,
            scene_model_capture_ns=self.scene.get('model_capture_ns'),
            scene_model_capture_age_s=(decision_ns-self.scene['model_capture_ns'])/1e9 if self.scene.get('model_capture_ns') else None,
            position_error_m=(np.asarray(target)-corrected).tolist(),orientation_error_rad=angle_error(orientation,q),
            hand_velocity_m_s=velocity.tolist() if velocity is not None else None,target_velocity_m_s=list(map(float,ff)),phase=self.phase)
        self.trace.append(copy.deepcopy(self.last_tracking))
        if self.phase=="track" and self.closure_proof is not None and velocity is not None:
            # The hand error, target velocity and covariance use this same time.
            x,cov,capture=predict_belief(self.belief,decision_ns,self.fc)
            sigma=float(np.sqrt(max(np.linalg.eigvalsh(cov[:2,:2]))))
            upper_speed=max(.001,x[2]+2*np.sqrt(cov[2,2]))
            remaining=(self.settings["safe_x_interval_m"][1]-x[0]-2*sigma)/upper_speed
            self.input_deadline=min(self.motion_lease,decision_ns+round(remaining*1e9))
            sample=dict(self.last_tracking,relative_velocity_m_s=(velocity-[x[2],x[3],0.]).tolist(),
                predicted_covariance=cov.reshape(-1).tolist(),capture_ns=capture,capture_age_s=(decision_ns-capture)/1e9,
                remaining_s=remaining,descent_complete=decision_ns-self.tracking_start>=round(self.c["descend_duration_s"]*1e9))
            ready=self.closure_proof.observe(sample)
            diagnostic=dict(sample,ready=ready,reasons=list(self.closure_proof.window.reasons),
                accepted_distinct_tf_samples=self.closure_proof.window.count,
                distinct_camera_captures=len(self.closure_proof.window.captures))
            self.closure_trace.append(diagnostic);self.result["last_closure_check"]=diagnostic
            if remaining<=self.c["deadline_reserve_s"]:
                raise RuntimeError("Camera closure proof was not complete before the predicted safe-exit deadline")
        message.deadline=timestamp(self.input_deadline)
        self.publisher.publish(message)
    def start(self,request,response):
        try:
            require(not self.active and self.pending is None,"controller_busy")
            self.target();require(request.episode_id==self.epoch.episode_id and request.track_id==self.belief.track_id,"target_identity_mismatch")
            require(self.owner.get("owner")=="HOLD" and self.owner.get("ready") and not self.owner.get("busy"),"ready_hold_required")
            self.pending=copy.deepcopy(request);response.accepted=True;response.reason="accepted"
        except Exception as e:response.accepted=False;response.reason=str(e)
        return response
    def stop_request(self,request,response):self.abort=True;response.success=True;response.message="abort_requested";return response
    def allow_contact(self):
        request=SetHandleContact.Request(episode_id=self.epoch.episode_id,owner_epoch=self.owner["epoch"],allow_handle_contact=True)
        response=self.call(self.contact_service,request);require(response.success,response.reason)
        for client in self.scene_clients:
            def committed():
                r=GetPlanningScene.Request();r.components.components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
                a=self.call(client,r).scene.allowed_collision_matrix;n=a.entry_names
                return all(f in n for f in ("handle_bar","panda_leftfinger","panda_rightfinger")) and all(a.entry_values[n.index("handle_bar")].enabled[n.index(f)] for f in ("panda_leftfinger","panda_rightfinger"))
            end=time.monotonic()+5.
            while not committed():require(time.monotonic()<end,"Camera contact allowance not committed");self.wait_sim(.05)
    def before_tracking(self):pass
    def pickup(self,folder):
        if not self.active:self.reset_joint_guard()
        self.active=True;self.cleanup=False;self.abort=False;self.target_required=True;self.grip_candidate=False
        self.camera_wait=None;self.camera_wait_trace=[];self.camera_wait_count=0;self.camera_wait_total_wall_s=0.
        self.trace=[];self.events=[];self.closure_proof=None;self.closure_trace=[];self.velocity_window=VelocityWindow()
        self.bound_track=None # Bind strictly after the stopped-belt pregrasp.
        result=dict(status="FAIL",camera_only=True,oracle_input=False,forced_attachment=False,checks={},episode_id=self.pending.episode_id,track_id=self.bound_track,trial_id=self.pending.trial_id)
        self.result=result
        require(self.owner.get("controller_streaming_verified") and self.owner.get("collision_scene_verified"),"Arm gate readiness is incomplete")
        request=GetParameters.Request(names=["allow_trajectory_execution","moveit_manage_controllers","moveit_controller_manager"])
        parameters=self.call(self.planner_params,request).values
        require(len(parameters)==3 and all(v.type==ParameterType.PARAMETER_BOOL and v.bool_value is False for v in parameters[:2]) and
            parameters[2].type==ParameterType.PARAMETER_STRING and parameters[2].string_value=="sorting_control/PlanningOnlyControllers","MoveIt execution must remain disabled with the planning-only controller manager")
        self.until(self.hand.server_is_ready)
        self.set_phase("pregrasp");target,_,_=self.target()
        pre=[target[0],target[1],self.fc["fixed_handle_z_m"]+.1029+.06];orientation=self.settings["grasp_quaternion_xyzw"]
        result["pregrasp_plan"]=self.execute_plan(pre,orientation,folder,"camera-pregrasp")
        def camera_ready():
            try:
                x,_,_=self.target()
                return self.scene.get("valid") and np.linalg.norm(x[:2]-np.asarray(pre)[:2])<.002
            except (ValueError,RuntimeError):return False
        self.until(camera_ready,10.)
        result["pregrasp_refinement"]=self.refine_pose(pre,orientation);result["checks"]["camera_pregrasp"]="PASS"
        self.bound_track=self.belief.track_id;result["motion_track_id"]=self.bound_track
        self.before_tracking()
        self.acquire_mode("SERVO",self.c["motion_lease_s"],1);lease=self.input_deadline
        self.command_function=lambda now:(pre,[0.,0.,0.],orientation);self.input_enabled=True;self.allow_contact()
        self.belt_speed=self.c["belt_speed_m_s"];self.send_belt();self.set_phase("track");start=self.now_ns();self.tracking_start=start;self.motion_lease=lease;self.closure_proof=ClosureProof(self.c);self.closure_trace=[]
        def reference(now):
            x,p,capture=self.target(now);t=(now-start)/1e9;s,ds=smooth(t,self.c["descend_duration_s"])
            return [x[0],x[1],self.fc["fixed_handle_z_m"]+.1029+.06*(1-s)],[x[2],x[3],-.06*ds],orientation
        self.command_function=reference
        self.until(lambda:self.closure_proof.authorized(self.now_ns()),self.c["tracking_timeout_wall_s"])
        self.ensure_oracle_current()
        require(self.closure_proof.authorized(self.now_ns()),"Closure authorization expired before the gripper request")
        result["closure"]=copy.deepcopy(self.closure_proof.sample)
        result["checks"]["camera_closure_authorized"]="PASS";self.set_phase("close")
        goal=GripperCommand.Goal();goal.command.position=0.;goal.command.max_effort=20.
        require(self.hand.server_is_ready(),"Gripper action server became unavailable")
        require(self.closure_proof.authorized(self.now_ns()),"Closure authorization expired before sending the gripper goal")
        handle=self.future(self.hand.send_goal_async(goal));require(handle.accepted,"Gripper rejected closure")
        hand_result=handle.get_result_async();finger=FingerWindow(self.c)
        self.until(lambda:finger.observe(self.q_stamp_ns,[self.q[n] for n in ("panda_finger_joint1","panda_finger_joint2")],
                        [self.v[n] for n in ("panda_finger_joint1","panda_finger_joint2")]),self.c["grip_candidate_timeout_wall_s"])
        snapshot,_,_=self.target();snapshot_time=self.now_ns()
        request=SetEstimatedGrasp.Request(episode_id=self.epoch.episode_id,track_id=self.bound_track,owner_epoch=self.owner["epoch"],held=True)
        response=self.call(self.grasp_service,request);require(response.success,response.reason)
        self.grip_candidate=True;self.target_required=False;self.input_deadline=lease
        result["checks"]["encoder_grip_candidate"]="PASS";self.set_phase("grip_candidate")
        def candidate_reference(now):
            dt=(now-snapshot_time)/1e9;require(0<=dt<.4,"Estimated-attachment commit exceeded bounded camera prediction")
            return [snapshot[0]+snapshot[2]*dt,snapshot[1]+snapshot[3]*dt,self.fc["fixed_handle_z_m"]+.1029],[snapshot[2],snapshot[3],0.],orientation
        self.command_function=candidate_reference
        from camera_scene import TRAY_IDS,SUPPORT_CONTACT_IDS
        for client in self.scene_clients:
            end=time.monotonic()+5.
            while True:
                r=GetPlanningScene.Request();r.components.components=PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS|PlanningSceneComponents.WORLD_OBJECT_NAMES
                scene=self.call(client,r).scene
                attached={a.object.id for a in scene.robot_state.attached_collision_objects};world={o.id for o in scene.world.collision_objects}
                if TRAY_IDS<=attached and not TRAY_IDS.intersection(world):
                    r.components.components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
                    from camera_scene_checks import collision_allowed
                    matrix=self.call(client,r).scene.allowed_collision_matrix
                    require(all(collision_allowed(matrix,part,"support_surface") for part in SUPPORT_CONTACT_IDS),"Clear-lift support permissions were not committed")
                    require(all(collision_allowed(matrix,a,b) for a in TRAY_IDS for b in TRAY_IDS if a!=b),"Rigid tray internal distance permissions were not committed")
                    break
                require(time.monotonic()<end,"Estimated carried geometry was not committed");self.wait_sim(.02)
        result["checks"]["estimated_collision_attachment_committed"]="PASS";self.set_phase("lift")
        p0,_=self.hand_pose();vx=float(snapshot[2]);begin=self.now_ns();brake={"t":None,"x":None}
        def lift(now):
            t=(now-begin)/1e9;s,ds=smooth(t,self.c["lift_duration_s"]);p,_=self.hand_pose()
            require(t<self.c["clear_lift_progress_check_s"] or p[2]-p0[2]>self.c["minimum_clear_lift_progress_m"],
                    "Clear lift made insufficient measured progress; Servo may be scaling motion")
            xx=p0[0]+vx*t;vv=vx
            if p[2]-p0[2]>.005 and brake["t"] is None:brake.update(t=t,x=xx)
            if brake["t"] is not None:xx,vv=braking_x(brake["x"],vx,t-brake["t"],self.c["horizontal_brake_duration_s"])
            return [xx,p0[1],p0[2]+self.c["lift_height_m"]*s],[vv,0.,self.c["lift_height_m"]*ds],orientation
        self.command_function=lift;self.until(lambda:(self.now_ns()-begin)/1e9>=self.c["lift_duration_s"],15.)
        require(hand_result.done(),"Gripper result did not finish during lift");h=hand_result.result()
        require(h.status==GoalStatus.STATUS_SUCCEEDED and h.result.stalled and not h.result.reached_goal,"Gripper did not report a stalled handle candidate")
        result["gripper_result"]=dict(stalled=h.result.stalled,reached_goal=h.result.reached_goal,position_m=h.result.position)
        self.set_phase("hold");self.wait_sim(self.c["hold_duration_s"])
        require(self.scene.get("support_cleared") and not self.scene.get("allow_tray_support_contact"),"Clear-lift support allowance was not revoked")
        for client in self.scene_clients:
            r=GetPlanningScene.Request();r.components.components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
            from camera_scene_checks import collision_allowed
            matrix=self.call(client,r).scene.allowed_collision_matrix
            require(not any(collision_allowed(matrix,part,"support_surface") for part in TRAY_IDS),"Planning scene still permits tray/support contact after clear lift")
        result["checks"]["clear_lift_permission_revoked"]="PASS"
        result["checks"]["commanded_lift_and_hold"]="PASS"
        self.input_enabled=False;self.set_phase("final_hold");self.acquire_mode("HOLD")
        require(self.owner["stats"]["overlap_violations"]==0,"Arm source overlap")
        result["checks"]["exclusive_final_hold"]="PASS";result.update(status="PASS",policy_completed=True,physical_outcome="EVALUATOR_REQUIRED")
        return result
    def diagnostic(self):
        return dict(phase=self.phase,time_ns=self.now_ns(),scene=copy.deepcopy(self.scene),
            joint_feedback=copy.deepcopy(getattr(self,'joint_feedback_diagnostic',None)),
            scene_report_age_s=(self.now_ns()-self.scene.get('time_ns',0))/1e9,
            scene_model_capture_age_s=(self.now_ns()-self.scene['model_capture_ns'])/1e9 if self.scene.get('model_capture_ns') else None,
            camera_wait=copy.deepcopy(self.camera_wait),
            stationary_camera_recovery={key:self.c[key] for key in ('pregrasp_start_max_capture_age_s','maximum_capture_age_s',
                'maximum_stationary_camera_wait_sim_s','maximum_stationary_camera_wait_wall_s','maximum_stationary_camera_wait_total_wall_s')},
            belief_capture_ns=ns(self.belief.header.stamp) if self.belief else None,
            belief_capture_age_s=(self.now_ns()-ns(self.belief.header.stamp))/1e9 if self.belief else None,
            belief_track_id=self.belief.track_id if self.belief else None,bound_track_id=self.bound_track)
    def run_pending(self,folder):
        try:self.pickup(folder)
        except Exception as e:
            self.result.update(status="FAIL",error=str(e),policy_completed=False,failure_diagnostic=self.diagnostic())
            self.input_enabled=False;self.target_required=False;self.cleanup=True;self.abort=False;self.camera_wait=None
            try:
                self.until(lambda:not self.owner.get("busy"),12.)
                if self.owner.get("owner")!="FAULT":self.acquire_mode("HOLD")
            except Exception as stop:self.result["hold_error"]=str(stop)
        finally:
            self.input_enabled=False;self.belt_speed=0.;self.send_belt();self.active=False;self.cleanup=False;self.phase="idle"
            self.result.update(trace=self.trace,closure_trace=self.closure_trace,camera_wait_trace=self.camera_wait_trace,
                stationary_camera_waits=self.camera_wait_count,stationary_camera_wait_total_wall_s=self.camera_wait_total_wall_s,events=self.events,final_owner=self.owner,maximum_joint_speed_rad_s=self.maximum_joint_speed,
                maximum_joint_step_rad=self.maximum_joint_step,joint_feedback=copy.deepcopy(getattr(self,'joint_feedback_diagnostic',None)))
            (folder/"controller-result.json").write_text(json.dumps(self.result,indent=2)+"\n")
            self.result_pub.publish(String(data=json.dumps(self.result)));self.pending=None;self.publish_state()

def main():
    rclpy.init();node=Node("camera_pickup_controller");controller=Controller(node)
    from datetime import datetime
    folder=Path("results")/f"M3B-controller-{datetime.now():%Y%m%d-%H%M%S}";folder.mkdir(parents=True,exist_ok=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.02)
            if controller.pending is not None:
                trial=folder/f"trial-{controller.pending.trial_id}";trial.mkdir(exist_ok=True);controller.run_pending(trial)
    except KeyboardInterrupt:pass
    finally:controller.input_enabled=False;controller.belt_speed=0.;controller.send_belt();node.destroy_node();rclpy.try_shutdown()
if __name__=="__main__":main()
