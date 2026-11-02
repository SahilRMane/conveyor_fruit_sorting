"""Evaluator for M3A camera calibration. Truth stays in this test process."""
from collections import deque
import json
import math
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_prefix,get_package_share_directory
sys.path.insert(0,str(Path(get_package_prefix("sorting_control"))/"lib/sorting_control"))
from test_arm_ownership import Probe
from ownership_core import NAMES,require
from m2c_reset import ResetManager
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import Float64,String
from std_srvs.srv import SetBool
from sorting_interfaces.msg import CameraEpoch,CameraTruth,HandleObservation
from tf2_ros import Buffer,TransformListener,TransformException
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.clock import Clock,ClockType
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from camera_core import image_rgb,quaternion_rotation
from camera_qos import camera_subscription_qos

def ns(stamp):return stamp.sec*1_000_000_000+stamp.nanosec
def xyz(p):return [p.x,p.y,p.z]
def quat(q):return [q.x,q.y,q.z,q.w]

class CameraProbe(Probe):
    def __init__(self,node):
        self.c=yaml.safe_load((Path(get_package_share_directory("sorting_perception"))/"config/m3a.yaml").read_text())
        self.camera_config=yaml.safe_load((Path(get_package_share_directory("sorting_perception"))/"config/cameras.yaml").read_text())
        self.joint_updates=0;self.reset_record={};self.epoch=None;self.phase="startup";self.phase_begin_ns=0
        self.frames=[];self.latest_images={};self.info={};self.truth={};self.pending={};self.samples=[];self.rejections=[]
        self.camera_status={};self.belt_speed=0.;self.history=[];self.plan={"initial_state":{}}
        self.pairing_diagnostics=[];self.reported_pair_waits=set()
        self.camera_wait_diagnostic=None
        self.evaluation_enabled=False;self.reset_in_progress=False;self.reset_report=None
        self.failed_camera_sample=None;self.reset_stage="not_started"
        super().__init__(node);self.names=NAMES
        self.buffer=Buffer(cache_time=Duration(seconds=20.));self.listener=TransformListener(self.buffer,node)
        qos,metadata_qos=camera_subscription_qos(image_depth=5)
        durable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(CameraEpoch,"/sorting/episode",self.receive_epoch,durable)
        node.create_subscription(String,"/sorting/reset_state",self.receive_reset,durable)
        node.create_subscription(CameraTruth,"/sorting/ground_truth/camera_sample",self.receive_truth,200)
        node.create_subscription(String,"/sorting/perception/camera_status",lambda m:setattr(self,"camera_status",json.loads(m.data)),1)
        for view in ("overhead","wrist"):
            node.create_subscription(Image,f"/camera/{view}/image_raw",lambda m,v=view:self.receive_image(v,m),qos)
            node.create_subscription(CameraInfo,f"/camera/{view}/camera_info",lambda m,v=view:self.receive_info(v,m),metadata_qos)
            node.create_subscription(HandleObservation,f"/sorting/perception/{view}/handle_observation",self.receive_observation,10)
        self.belt=node.create_publisher(Float64,"/sorting/conveyor_command",1)
        self.reset_guard=node.create_client(SetBool,"/sorting/arm/reset_guard")
        self.reset_manager=ResetManager(self)
        wall=Clock(clock_type=ClockType.STEADY_TIME)
        node.create_timer(0.1,self.send_belt,clock=wall);node.create_timer(0.01,self.evaluate,clock=wall)

    @property
    def state(self):return self.contact
    @property
    def joints(self):return self.q
    @property
    def joint_velocities(self):return self.v

    def receive_joints(self,message):
        super().receive_joints(message);self.joint_updates+=1

    def check(self,value):
        require(value["forbidden_contacts"]==0,"Unexpected physical contact in camera check")
        require(value["left_force_N"]+value["right_force_N"]<0.1,"Camera check must keep the hand open")
        require(min(value["finger_positions_m"])>0.039,"Camera check fingers are not open")
        require(value["support_force_N"]>0.5,"Camera localization requires a supported tray")
        require(abs(value["external_force_x_N"])<=8.0001,"Conveyor force exceeded its bound")
        require(all(math.isfinite(v) for v in value["tray_xyz"]),"Invalid physical state")

    def receive_contact(self,message):
        self.contact=json.loads(message.data)
        if self.guarded:self.check(self.contact)
    def receive_epoch(self,message):self.epoch=message
    def receive_reset(self,message):self.reset_record=json.loads(message.data)
    def receive_truth(self,message):
        if not self.evaluation_enabled:return
        self.truth[(message.episode_id,message.view_id,ns(message.header.stamp))]=message
        while len(self.truth)>2000:self.truth.pop(next(iter(self.truth)))
    def receive_info(self,view,message):
        if not self.evaluation_enabled:return
        self.info[(view,ns(message.header.stamp))]=message
        while len(self.info)>1000:self.info.pop(next(iter(self.info)))
    def receive_image(self,view,message):
        stamp=ns(message.header.stamp);now=self.now_ns();rgb=image_rgb(message)
        self.frames.append(dict(view_id=view,capture_ns=stamp,receipt_ns=now,phase=self.phase,
                                width=message.width,height=message.height,encoding=message.encoding,
                                camera_frame=message.header.frame_id,image_std=float(rgb.std()),age_s=(now-stamp)/1e9))
        self.latest_images[view]=(message,rgb)
    def receive_observation(self,message):
        stamp=ns(message.header.stamp)
        if message.valid:
            # Volatile truth/CameraInfo packets captured before this phase can
            # precede the evaluator's subscriptions. They are not scoreable.
            if self.phase=="startup" or stamp<=self.phase_begin_ns:
                self.rejections.append(dict(episode_id=message.episode_id,view_id=message.view_id,capture_ns=stamp,
                    reason="evaluator_startup_or_previous_phase_capture",phase=self.phase,localizer_valid=True))
                return
            if not self.evaluation_enabled:
                self.rejections.append(dict(episode_id=message.episode_id,view_id=message.view_id,capture_ns=stamp,
                    reason="evaluator_reset_or_failure_paused",phase=self.phase,localizer_valid=True))
                return
            if self.epoch is not None and message.episode_id!=self.epoch.episode_id:
                self.rejections.append(dict(episode_id=message.episode_id,view_id=message.view_id,capture_ns=stamp,
                    reason="evaluator_episode_mismatch",phase=self.phase,localizer_valid=True))
                return
            self.pending[(message.episode_id,message.view_id,stamp)]=(message,time.monotonic(),self.phase)
        else:self.rejections.append(dict(episode_id=message.episode_id,view_id=message.view_id,capture_ns=stamp,reason=message.reason,phase=self.phase))

    def pairing_detail(self,key,phase,missing,tf_error,wait_wall_s):
        episode,view,stamp=key
        truth_times=[t for e,v,t in self.truth if e==episode and v==view]
        info_times=[t for v,t in self.info if v==view]
        return dict(episode_id=episode,view_id=view,capture_ns=stamp,phase=phase,missing=missing,
                    tf_error=tf_error,wait_wall_s=wait_wall_s,
                    nearest_truth_delta_ns=min((abs(t-stamp) for t in truth_times),default=None),
                    nearest_camera_info_delta_ns=min((abs(t-stamp) for t in info_times),default=None),
                    truth_capture_range_ns=[min(truth_times),max(truth_times)] if truth_times else None)

    def evaluate(self):
        if not self.evaluation_enabled:return
        for key,(observation,received,phase) in list(self.pending.items()):
            episode,view,stamp=key
            truth=self.truth.get(key)
            if truth is None:
                # The bridge and rclcpp Duration can differ by one nanosecond.
                for delta in (-2,-1,1,2):
                    truth=self.truth.get((episode,view,stamp+delta))
                    if truth is not None:break
            info=self.info.get((view,stamp))
            missing=[];tf_error=None
            if truth is None:missing.append("truth")
            if info is None:missing.append("camera_info")
            try:
                if missing:raise LookupError()
                transform=self.buffer.lookup_transform("world",observation.camera_frame,Time.from_msg(observation.header.stamp))
            except (LookupError,TransformException) as error:
                if isinstance(error,TransformException):missing.append("capture_time_tf");tf_error=str(error)
                wait=time.monotonic()-received
                if key not in self.reported_pair_waits or wait>=2.:
                    detail=self.pairing_detail(key,phase,missing,tf_error,wait)
                    detail.update(resolved=False,status="TIMEOUT" if wait>=2. else "WAITING")
                    self.pairing_diagnostics.append(detail);self.reported_pair_waits.add(key)
                if wait>=2. and missing==["camera_info"]:
                    # A valid policy observation does not guarantee that this
                    # separate evaluator received every calibration packet.
                    # Preserve the missing pair as unscored; never substitute
                    # another timestamp or count it toward minimum evidence.
                    detail.update(status="SKIPPED_METADATA",counted=False)
                    self.pending.pop(key,None)
                    self.rejections.append(dict(episode_id=episode,view_id=view,capture_ns=stamp,phase=phase,
                        reason="evaluator_camera_info_unavailable",localizer_valid=True,counted=False))
                    continue
                require(wait<2.,f"Evaluator capture pair timed out: view={view}, capture_ns={stamp}, missing={','.join(missing)}; see pairing-diagnostics.json")
                continue
            self.pending.pop(key,None)
            if key in self.reported_pair_waits:
                for detail in reversed(self.pairing_diagnostics):
                    if (detail["episode_id"],detail["view_id"],detail["capture_ns"])==key:
                        detail.update(resolved=True,status="RESOLVED",resolved_after_wall_s=time.monotonic()-received)
                        break
            position=np.asarray(xyz(observation.position));target=np.asarray(xyz(truth.handle_position))
            tf=transform.transform;rotation=quaternion_rotation(quat(tf.rotation));expected=quaternion_rotation(quat(truth.camera_pose.orientation))
            cosine=np.clip((np.trace(rotation@expected.T)-1)/2,-1,1)
            covariance=np.asarray(observation.planar_covariance).reshape(2,2)
            sample=dict(episode_id=episode,view_id=view,capture_ns=stamp,phase=phase,position=position.tolist(),
                        true_handle_position=target.tolist(),error_xy_m=(position[:2]-target[:2]).tolist(),
                        planar_error_m=float(np.linalg.norm(position[:2]-target[:2])),
                        camera_position_error_m=float(np.linalg.norm(np.asarray(xyz(tf.translation))-xyz(truth.camera_pose.position))),
                        camera_orientation_error_rad=float(np.arccos(cosine)),planar_covariance=observation.planar_covariance.tolist(),
                        pixel_u=observation.pixel_u,pixel_v=observation.pixel_v,area_px=observation.area_px,
                        compute_wall_s=observation.compute_wall_s,capture_frame=info.header.frame_id)
            scored=((view=="overhead" and phase in {"stationary","moving_belt","post_reset"}) or
                    (view=="wrist" and phase in {"wrist_inspection","post_reset"}))
            sample["localization_scored"]=scored
            try:
                require(info.header.frame_id==observation.camera_frame,"CameraInfo optical frame mismatch")
                require(np.all(np.isfinite(covariance)) and np.allclose(covariance,covariance.T) and np.linalg.eigvalsh(covariance).min()>=0,"Invalid observation covariance")
                require(sample["camera_position_error_m"]<self.c["maximum_camera_position_error_m"] and
                        sample["camera_orientation_error_rad"]<self.c["maximum_camera_orientation_error_rad"],"Capture-time optical TF differs from physics")
                if scored:require(sample["planar_error_m"]<self.c["maximum_planar_error_m"],"Image localization exceeds 2 mm")
            except Exception as error:
                self.failed_camera_sample=dict(sample,error=str(error))
                raise
            self.samples.append(sample)

    def send_belt(self):self.belt.publish(Float64(data=float(self.belt_speed)))
    def begin(self,phase):
        self.phase=phase;self.phase_begin_ns=self.now_ns()
        self.evaluation_enabled=phase not in {"startup","reset","failure"}
    def phase_samples(self,view):return [s for s in self.samples if s["phase"]==self.phase and s["view_id"]==view and s["capture_ns"]>self.phase_begin_ns]
    def collect(self,view,count,timeout=12.):
        try:self.until(lambda:len(self.phase_samples(view))>=count,timeout)
        except TimeoutError as error:
            samples=self.phase_samples(view)
            frames=[f for f in self.frames if f["phase"]==self.phase and f["view_id"]==view and f["capture_ns"]>self.phase_begin_ns]
            ages=[f["age_s"] for f in frames]
            stats=self.camera_status.get("views",{}).get(view,{})
            self.camera_wait_diagnostic=dict(phase=self.phase,view_id=view,paired_samples=len(samples),
                required_samples=count,raw_frames=len(frames),maximum_allowed_image_age_s=self.c["maximum_image_age_s"],
                median_received_image_age_s=float(np.median(ages)) if ages else None,
                maximum_received_image_age_s=max(ages) if ages else None,
                localizer_rejection_reasons=stats.get("reasons",{}),localizer_received=stats.get("received",0),
                localizer_valid=stats.get("valid",0),timeout_wall_s=timeout)
            raise TimeoutError(f"Camera observation wait timed out: phase={self.phase}, view={view}, paired={len(samples)}/{count}; see camera-wait-diagnostic.json") from error
        return self.phase_samples(view)
    def save_images(self,folder,label):
        for view,(_,rgb) in self.latest_images.items():cv2.imwrite(str(folder/f"{label}-{view}.png"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))

    def audit(self):
        names=[n for n,_ in self.node.get_node_names_and_namespaces()]
        require(names.count("handle_localizer")==1,"Require exactly one handle localizer")
        require(not any("oracle" in n for n in names),"An oracle producer is running in the M3 launch")
        allowed={"/tf","/tf_static","/clock","/parameter_events","/sorting/episode"}
        allowed|={f"/camera/{v}/{t}" for v in ("overhead","wrist") for t in ("image_raw","camera_info")}
        subscriptions=[name for name,_ in self.node.get_subscriber_names_and_types_by_node("handle_localizer","/")]
        require(set(subscriptions)<=allowed,"Perception has a forbidden non-camera subscription: "+str(set(subscriptions)-allowed))
        consumers=self.node.get_subscriptions_info_by_topic("/sorting/ground_truth/camera_sample")
        require(consumers and all(p.node_name=="test_m3a" for p in consumers),"Camera truth has a non-evaluator subscriber")
        require(self.camera_status.get("oracle_input") is False,"Localizer did not confirm image-only input")
        return dict(localizer_subscriptions=subscriptions,truth_consumers=[p.node_name for p in consumers],oracle_producers=[],status="PASS")

    def move(self,positions,duration):
        epoch,deadline=self.acquire("PLANNED",seconds=20.)
        handle=self.trajectory(epoch,deadline,positions,duration)
        require(handle.accepted,"Camera calibration motion was rejected")
        result=self.future(handle.get_result_async())
        require(result.status==GoalStatus.STATUS_SUCCEEDED and result.result.success,"Camera calibration motion failed: "+result.result.reason)
        self.acquire("HOLD");self.wait_sim(0.2)

    def wait_future(self,future,timeout=30.):return self.future(future,timeout)
    def confirm_controller(self):require(self.owner.get("controller_streaming_verified") is True,"Controller settings are not verified")
    def prepare_for_reset(self):
        self.begin("reset")
        self.guarded=False;self.previous_q=None;self.q.clear();self.v.clear();self.pending.clear();self.reference=None
        self.truth.clear();self.info.clear()
    def call(self,client,request):
        self.until(client.service_is_ready);return self.future(client.call_async(request))
    def reset_episode(self):
        self.belt_speed=0.;self.send_belt();self.until(lambda:abs(self.contact.get("external_force_x_N",1))<1e-6)
        self.begin("reset");self.pending.clear();self.reset_stage="request_guard"
        previous=self.epoch.episode_id;response=self.call(self.reset_guard,SetBool.Request(data=True));require(response.success,response.message)
        self.reset_in_progress=True;self.reset_stage="physical_reset"
        reset=self.reset_manager.reset_home();self.reset_report=reset
        require(reset["status"]=="PASS",reset.get("error","Reset failed"))
        self.reset_stage="guard_revalidation"
        response=self.call(self.reset_guard,SetBool.Request(data=False));require(response.success,response.message)
        self.reset_in_progress=False
        self.until(lambda:self.owner.get("owner")=="HOLD" and self.owner.get("ready") and not self.owner.get("busy") and
                   self.epoch.episode_id>previous and self.camera_status.get("episode_id")==self.epoch.episode_id)
        require(self.epoch.episode_id==reset["reset_seq"],"Camera epoch differs from physical reset identity")
        self.pending.clear();self.truth.clear();self.info.clear()
        self.reset_stage="post_reset_camera_evidence";self.begin("post_reset")
        return reset
    def stop_after_failure(self):
        from mujoco_ros2_control_msgs.srv import SetPause
        # Cleanup spins callbacks. Close the evaluator first to prevent a
        # second validation exception from interrupting the stop protocol.
        self.evaluation_enabled=False;self.pending.clear()
        self.guarded=False;self.belt_speed=0.;self.send_belt();out={}
        if self.reset_in_progress or self.owner.get("owner")=="RESETTING":
            out.update(hold="NOT_REQUESTED",retained_owner=self.owner.get("owner"),reason="Retain closed reset guard after an incomplete reset")
        else:
            try:self.safe_hold();out["hold"]="PASS"
            except Exception as error:out["hold_error"]=str(error)
        try:
            result=self.reset_manager.call(self.reset_manager.client(SetPause),SetPause.Request(paused=True))
            require(result.success,result.message);out["paused"]=True
        except Exception as error:out["pause_error"]=str(error)
        return out
