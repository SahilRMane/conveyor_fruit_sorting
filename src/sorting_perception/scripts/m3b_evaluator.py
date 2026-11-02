"""Separate physical evaluator and reset runner. No arm or belt publishers."""
import copy
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory,get_package_prefix
sys.path.insert(0,str(Path(get_package_prefix("sorting_control"))/"lib/sorting_control"))
from test_arm_ownership import Probe
from m2c_reset import ResetManager
from m2c_runtime import MovingGrasp
from ownership_core import NAMES,require,ns
from camera_scene_checks import reset_scene_diagnostic
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool,Trigger
from sorting_interfaces.msg import CameraEpoch,CameraTruth,TargetBeliefArray
from sorting_interfaces.srv import SetArmOwner,StartCameraPickup
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningSceneComponents
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy

class Evaluator:
    until=Probe.until;future=Probe.future;wait_sim=Probe.wait_sim;safe_hold=Probe.safe_hold
    def __init__(self,node):
        self.node=node;self.c=yaml.safe_load((Path(get_package_share_directory("sorting_control"))/"config/m3b.yaml").read_text())
        self.owner={};self.q={};self.v={};self.contact={};self.epoch=None;self.reset_record={};self.joint_updates=0
        self.guarded=True;self.reference=None;self.previous_q=None;self.input_enabled=False;self.maximum_joint_speed=0.;self.maximum_joint_step=0.
        self.phase="startup";self.grasp=None;self.maximum_slip=0.;self.maximum_world_offset_change=0.;self.grip_trace=[];self.history=[]
        self.names=NAMES;self.plan={"initial_state":{}};self.events=[];self.control_state={};self.control_result=None
        self.targets=None;self.camera_status={};self.filter_status={};self.scene={};self.belief_log=[];self.truth={};self.filter_errors=[];self.seen_beliefs=set();self.pending_errors={}
        self.finger_start=None;self.maximum_rise=0.;self.lifted_samples=[];self.evaluation_enabled=True;self.reset_in_progress=False
        durable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String,"/sorting/arm/owner_state",self.receive_owner,1)
        node.create_subscription(String,"/sorting/arm/events",lambda m:self.events.append(json.loads(m.data)),100)
        node.create_subscription(JointState,"/joint_states",self.receive_joints,QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
        node.create_subscription(String,"/sorting/contact_state",self.receive_contact,10)
        node.create_subscription(CameraEpoch,"/sorting/episode",lambda m:setattr(self,"epoch",m),durable)
        node.create_subscription(String,"/sorting/reset_state",lambda m:setattr(self,"reset_record",json.loads(m.data)),durable)
        node.create_subscription(CameraTruth,"/sorting/ground_truth/camera_sample",self.receive_truth,200)
        node.create_subscription(TargetBeliefArray,"/sorting/targets",self.receive_targets,1)
        node.create_subscription(String,"/sorting/perception/camera_status",lambda m:setattr(self,"camera_status",json.loads(m.data)),1)
        node.create_subscription(String,"/sorting/perception/filter_status",lambda m:setattr(self,"filter_status",json.loads(m.data)),1)
        node.create_subscription(String,"/sorting/perception/grasp_scene_state",lambda m:setattr(self,"scene",json.loads(m.data)),1)
        node.create_subscription(String,"/sorting/control/pickup_state",self.receive_control,1)
        node.create_subscription(String,"/sorting/control/pickup_result",lambda m:setattr(self,"control_result",json.loads(m.data)),durable)
        self.service=node.create_client(SetArmOwner,"/sorting/arm/set_owner")
        self.start=node.create_client(StartCameraPickup,"/sorting/control/start_camera_pickup")
        self.abort=node.create_client(Trigger,"/sorting/control/abort_camera_pickup")
        self.reset_guard=node.create_client(SetBool,"/sorting/arm/reset_guard")
        self.reset_manager=ResetManager(self)
        self.reset_diagnostics=[];self.last_reset_report=None
        self.scene_clients=[node.create_client(GetPlanningScene,p) for p in ("/get_planning_scene","/m2b_planner/get_planning_scene")]
    def now_ns(self):return self.node.get_clock().now().nanoseconds
    @property
    def state(self):return self.contact
    @property
    def joints(self):return self.q
    @property
    def joint_velocities(self):return self.v
    def receive_owner(self,m):
        self.owner=json.loads(m.data)
        if self.guarded:
            require(self.owner.get("owner")!="FAULT","Arm gate fault: "+self.owner.get("reason",""))
            require(self.owner.get("stats",{}).get("overlap_violations",0)==0,"Arm source overlap")
    def receive_joints(self,m):
        self.q=dict(zip(m.name,m.position));self.v=dict(zip(m.name,m.velocity));self.joint_updates+=1
        if self.guarded and all(n in self.q and n in self.v for n in NAMES):
            values=[self.q[n] for n in NAMES];speed=max(abs(self.v[n]) for n in NAMES)
            self.maximum_joint_speed=max(self.maximum_joint_speed,speed)
            require(speed<self.c["maximum_joint_speed_rad_s"],"Physical joint speed exceeds 0.5 rad/s")
            if self.previous_q is not None:require(max(abs(a-b) for a,b in zip(values,self.previous_q))<.02,"Physical joint feedback jump")
            self.previous_q=values
    def receive_control(self,m):
        self.control_state=json.loads(m.data)
        if not self.reset_in_progress:self.phase=self.control_state.get("phase","startup")
    def check(self,value):
        phase=self.phase
        if phase in {"pregrasp_refinement_wait","pregrasp_refinement"}:self.phase="pregrasp"
        try:MovingGrasp.check(self,value)
        finally:self.phase=phase
    def receive_contact(self,m):
        self.contact=json.loads(m.data);self.history.append(copy.deepcopy(self.contact))
        if not self.guarded:return
        if self.grasp is None and self.phase in {"close","grip_candidate","lift","hold","final_hold"}:
            if min(self.contact["left_force_N"],self.contact["right_force_N"])>.2:
                if self.finger_start is None:self.finger_start=self.contact["sim_time"]
                if self.contact["sim_time"]-self.finger_start>=.2:self.grasp=copy.deepcopy(self.contact)
            else:self.finger_start=None
        self.check(self.contact)
        if self.grasp is not None:
            rise=self.contact["tray_xyz"][2]-self.grasp["tray_xyz"][2];self.maximum_rise=max(self.maximum_rise,rise)
            if self.phase=="hold":self.lifted_samples.append(copy.deepcopy(self.contact))
    def receive_truth(self,m):
        if not self.evaluation_enabled:return
        self.truth[(m.episode_id,ns(m.header.stamp))]=m
        self.pair_errors()
        while len(self.truth)>1000:self.truth.pop(next(iter(self.truth)))
    def receive_targets(self,m):
        self.targets=m
        if not self.evaluation_enabled:return
        for b in m.targets:
            key=(b.episode_id,b.track_id,ns(b.header.stamp))
            if key in self.seen_beliefs or not b.valid:continue
            self.seen_beliefs.add(key)
            from rosidl_runtime_py.convert import message_to_ordereddict
            self.belief_log.append(message_to_ordereddict(b))
            self.pending_errors[key]=copy.deepcopy(b)
        self.pair_errors()
    def pair_errors(self):
        for key,b in list(self.pending_errors.items()):
            truth=self.truth.get((b.episode_id,ns(b.header.stamp)))
            if truth:
                self.filter_errors.append(dict(episode_id=b.episode_id,track_id=b.track_id,capture_ns=ns(b.header.stamp),
                    error_xy_m=[b.state[0]-truth.handle_position.x,b.state[1]-truth.handle_position.y],view_id=b.view_id))
                self.pending_errors.pop(key)
        while len(self.pending_errors)>200:self.pending_errors.pop(next(iter(self.pending_errors)))
    def current(self):
        if self.targets is None or self.epoch is None:return None
        return next((b for b in self.targets.targets if b.valid and b.episode_id==self.epoch.episode_id),None)
    def call(self,client,request):self.until(client.service_is_ready);return self.future(client.call_async(request))
    def wait_future(self,future,timeout=30.):return self.future(future,timeout)
    def send_belt(self):pass # The camera controller is the sole belt publisher.
    def confirm_controller(self):require(self.owner.get("controller_streaming_verified"),"Controller parameters not verified")
    def audit(self):
        names=[n for n,_ in self.node.get_node_names_and_namespaces()]
        policies={"handle_localizer","camera_motion_filter","camera_grasp_scene","camera_pickup_controller"}
        require(policies<=set(names),"Camera pipeline nodes are missing")
        require(not any("oracle" in n for n in names),"Oracle node present in camera launch")
        subscriptions={}
        for n in policies:
            topics=[t for t,_ in self.node.get_subscriber_names_and_types_by_node(n,"/")]
            bad=[t for t in topics if t.startswith("/sorting/ground_truth/") or t.startswith("/sorting/debug/") or t in {"/sorting/contact_state","/sorting/reset_state"}]
            require(not bad,"Policy has simulator-truth input: "+n+str(bad));subscriptions[n]=topics
        for topic in ("/sorting/contact_state","/sorting/ground_truth/camera_sample","/sorting/reset_state"):
            require(all(p.node_name=="test_m3b" for p in self.node.get_subscriptions_info_by_topic(topic)),"Truth subscriber outside evaluator: "+topic)
        return dict(status="PASS",policy_subscriptions=subscriptions,oracle_input=False)
    def prepare_trial(self):
        self.grasp=None;self.finger_start=None;self.maximum_slip=0.;self.maximum_world_offset_change=0.;self.maximum_rise=0.
        self.grip_trace=[];self.history=[];self.lifted_samples=[];self.control_result=None;self.trial_error_start=len(self.filter_errors);self.evaluation_enabled=True;self.guarded=True
    def verify_physical(self,policy):
        require(policy["status"]=="PASS",policy.get("error","Policy failed"))
        require(self.grasp is not None,"No stable bilateral physical contact")
        require(self.maximum_rise>=self.c["minimum_lift_rise_m"],"Actual tray did not lift 45 mm")
        require(self.lifted_samples and min(s["tray_xyz"][2]-self.grasp["tray_xyz"][2] for s in self.lifted_samples)>=.045,"Physical tray was not held 45 mm above support")
        require(self.contact["tray_xyz"][2]-self.grasp["tray_xyz"][2]>.03 and self.contact["support_force_N"]<.1,"Final HOLD did not retain the physical tray")
        require(abs(self.contact["external_force_x_N"])<1e-6,"Conveyor traction did not cease after lift")
        errors=[p for p in self.filter_errors[self.trial_error_start:] if p["capture_ns"]<=policy["closure"]["capture_ns"]]
        require(len(errors)>=10,"Too few evaluator camera/filter pairs")
        max_filter_error=max(np.linalg.norm(p["error_xy_m"]) for p in errors)
        require(max_filter_error<.002,"Supported camera/filter position error exceeds2mm")
        closure=policy["closure"];sample=min(self.history,key=lambda s:abs(round(s["sim_time"]*1e9)-closure["time_ns"]))
        require(abs(round(sample["sim_time"]*1e9)-closure["time_ns"])<30_000_000,"Closure truth sample is not time aligned")
        from prediction_core import rotation
        q=sample["tray_quat_wxyz"];handle=np.asarray(sample["tray_xyz"])+rotation([q[1],q[2],q[3],q[0]])@np.array([0.,0.,.117])
        error=np.asarray(sample["hand_xyz"])-(handle+[0,0,.1029]);relative=np.asarray(sample["hand_linear_velocity_m_s"])-sample["tray_linear_velocity_m_s"]
        require(np.linalg.norm(error[:2])<.002 and abs(error[2])<.002,"True closure position error exceeds2mm")
        require(np.linalg.norm(relative)<.005,"True closure relative velocity exceeds 5 mm/s")
        return dict(status="PASS",true_closure_position_error_m=error.tolist(),true_closure_relative_velocity_m_s=relative.tolist(),
            maximum_supported_filter_error_xy_m=float(max_filter_error),evaluator_camera_pairs=len(errors),maximum_lift_rise_m=self.maximum_rise,maximum_hand_frame_slip_m=self.maximum_slip,grasp_state=self.grasp,final_contact_state=self.contact)
    def prepare_for_reset(self):
        self.reset_in_progress=True;self.evaluation_enabled=False;self.guarded=False;self.phase="reset"
        self.previous_q=None;self.q.clear();self.v.clear();self.grasp=None;self.finger_start=None;self.reference=None;self.truth.clear();self.pending_errors.clear()
    def reset_episode(self):
        self.until(lambda:not self.control_state.get("active") and self.owner.get("owner")=="HOLD" and not self.owner.get("busy"))
        self.until(lambda:abs(self.contact.get("external_force_x_N",1))<1e-6)
        response=self.call(self.reset_guard,SetBool.Request(data=True));require(response.success,response.message)
        reset=self.reset_manager.reset_home();self.last_reset_report=reset;require(reset["status"]=="PASS",reset.get("error","Physical reset failed"))
        response=self.call(self.reset_guard,SetBool.Request(data=False));require(response.success,response.message)
        self.until(lambda:self.owner.get("owner")=="HOLD" and self.owner.get("ready") and not self.owner.get("busy") and
            self.current() is not None and self.current().episode_id==reset["reset_seq"] and self.scene.get("valid") and self.scene.get("episode_id")==reset["reset_seq"] and not self.scene.get("estimated_attachment"))
        from camera_scene import TRAY_IDS
        end=time.monotonic()+5.
        for client in self.scene_clients:
            while True:
                r=GetPlanningScene.Request();r.components.components=PlanningSceneComponents.WORLD_OBJECT_NAMES|PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS|PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
                scene=self.call(client,r).scene;world={o.id for o in scene.world.collision_objects};attached={a.object.id for a in scene.robot_state.attached_collision_objects}
                diagnostic=reset_scene_diagnostic(scene,TRAY_IDS);diagnostic.update(service=client.srv_name,time_ns=self.now_ns(),episode_id=reset["reset_seq"])
                self.reset_diagnostics.append(diagnostic)
                if diagnostic["valid"]:break
                require(time.monotonic()<end,"Reset collision geometry or contact permission was not restored: "+json.dumps(diagnostic));self.wait_sim(.02)
        reset["camera_collision_reset"]="PASS"
        self.reset_in_progress=False;self.evaluation_enabled=True;self.guarded=True;self.phase="startup"
        return reset
    def stop_after_failure(self):
        from mujoco_ros2_control_msgs.srv import SetPause
        self.guarded=False;self.evaluation_enabled=False;out={}
        try:
            if self.control_state.get("active"):
                self.call(self.abort,Trigger.Request());self.until(lambda:not self.control_state.get("active"),15.)
            if self.owner.get("owner")!="RESETTING":self.safe_hold();out["hold"]="PASS"
        except Exception as e:out["hold_error"]=str(e)
        try:
            r=self.reset_manager.call(self.reset_manager.client(SetPause),SetPause.Request(paused=True));out["paused"]=bool(r.success)
        except Exception as e:out["pause_error"]=str(e)
        return out
