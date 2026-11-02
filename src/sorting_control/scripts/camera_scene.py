#!/usr/bin/env python3
"""Camera-derived collision geometry. The plant remains physically free."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock,ClockType
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from moveit_msgs.msg import PlanningScene,AllowedCollisionEntry,AttachedCollisionObject,CollisionObject
from sorting_interfaces.msg import TargetBeliefArray,CameraEpoch
from sorting_interfaces.srv import SetHandleContact,SetEstimatedGrasp
from tf2_ros import Buffer,TransformListener
from ament_index_python.packages import get_package_share_directory
from camera_grasp_core import predict_belief
from prediction_core import rotation as quaternion_rotation
from publish_static_fixtures import scene_message
TRAY_IDS={"tray_base","tray_wall_x_minus","tray_wall_x_plus","tray_wall_y_minus","tray_wall_y_plus","handle_post_minus","handle_post_plus","handle_bar"}
# The bar is well above the support. The seven lower parts are one rigid tray.
SUPPORT_CONTACT_IDS=TRAY_IDS-{"handle_bar"}
from ownership_core import require

def xyz(p):return np.array([p.x,p.y,p.z])
def rotation(q):return quaternion_rotation([q.x,q.y,q.z,q.w])
def hand_transform(buffer):
    m=buffer.lookup_transform("world","panda_hand",Time());t=np.eye(4)
    t[:3,:3]=rotation(m.transform.rotation);t[:3,3]=xyz(m.transform.translation)
    return t,m.header.stamp.sec*1_000_000_000+m.header.stamp.nanosec

def scene_for_camera(template,target,settings,allow,relative=None,allow_support_contact=False):
    scene=copy.deepcopy(template);scene.robot_state.is_diff=True
    root=np.array([target[0],target[1],settings["fixed_handle_z_m"]-.117])
    for obj in scene.world.collision_objects:
        if obj.id not in TRAY_IDS:continue
        nominal=xyz(obj.primitive_poses[0].position)-[.55,0,.307]
        p=root+nominal
        if relative is None:
            obj.primitive_poses[0].position.x,obj.primitive_poses[0].position.y,obj.primitive_poses[0].position.z=map(float,p)
        else:
            # MoveIt model attachment only. Nothing is sent to MuJoCo.
            obj.operation=CollisionObject.REMOVE
            attached=AttachedCollisionObject();attached.link_name="panda_hand"
            attached.object=copy.deepcopy(obj);attached.object.operation=CollisionObject.ADD
            attached.object.header.frame_id="panda_hand"
            local=relative[:3,:3]@nominal+relative[:3,3]
            attached.object.primitive_poses[0].position.x,attached.object.primitive_poses[0].position.y,attached.object.primitive_poses[0].position.z=map(float,local)
            # The carried tray orientation is the validated horizontal prior,
            # rotated into the measured hand frame at the grip candidate.
            from scipy.spatial.transform import Rotation
            q=Rotation.from_matrix(relative[:3,:3]).as_quat()
            pose=attached.object.primitive_poses[0]
            pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,q)
            attached.touch_links=["panda_leftfinger","panda_rightfinger"] if obj.id=="handle_bar" else []
            scene.robot_state.attached_collision_objects.append(attached)
    if relative is not None:scene.world.collision_objects=[o for o in scene.world.collision_objects if o.id not in TRAY_IDS]
    old=template.allowed_collision_matrix;names=sorted(set(old.entry_names)|TRAY_IDS|{"panda_leftfinger","panda_rightfinger","support_surface"})
    pairs={(a,b) for i,a in enumerate(old.entry_names) for j,b in enumerate(old.entry_names) if old.entry_values[i].enabled[j]}
    if allow:pairs|={("handle_bar",f) for f in ("panda_leftfinger","panda_rightfinger")}|{(f,"handle_bar") for f in ("panda_leftfinger","panda_rightfinger")}
    if relative is not None:
        # MoveIt 2.12.4 collision checks skip bodies on the same attached link,
        # but its distance callback does not. These are parts of one rigid body.
        pairs|={(a,b) for a in TRAY_IDS for b in TRAY_IDS}
    if allow_support_contact:
        require(relative is not None,"Support contact allowance requires the estimated carried model")
        pairs|={(part,"support_surface") for part in SUPPORT_CONTACT_IDS}|{("support_surface",part) for part in SUPPORT_CONTACT_IDS}
    scene.allowed_collision_matrix.entry_names=names
    scene.allowed_collision_matrix.entry_values=[AllowedCollisionEntry(enabled=[a==b or (a,b) in pairs for b in names]) for a in names]
    return scene

def support_contact_state(world,now,grip_time,cleared,owner,settings):
    """Bound expected contact between the rigid tray and its support from measured hand FK."""
    require(np.isfinite(world).all(),"Estimated carried transform is not finite")
    bottom=float(world[2,3]-np.abs(world[2,:3])@np.array([.12,.09,.006]))
    gap=bottom-.30
    require(gap>-.003,"Estimated tray penetrates the support beyond the grip corridor")
    cleared=bool(cleared or gap>settings["clear_lift_support_gap_m"])
    elapsed=now-grip_time
    allowed=(not cleared and owner.get("owner")=="SERVO" and not owner.get("busy") and
             0<=elapsed<round(settings["maximum_clear_lift_contact_s"]*1e9))
    return gap,cleared,allowed

def lightweight_camera_update(template,target,settings,allow,permissions_changed=False):
    # Permission changes still use the complete matrix. Ordinary MOVE updates
    # need only eight poses, not copies of the static world and its matrix.
    if permissions_changed:
        scene=scene_for_camera(template,target,settings,allow)
        scene.world.collision_objects=[o for o in scene.world.collision_objects if o.id in TRAY_IDS]
        for obj in scene.world.collision_objects:
            obj.operation=CollisionObject.MOVE;obj.primitives=[]
        return scene
    scene=PlanningScene(is_diff=True);scene.robot_state.is_diff=True
    root=np.array([target[0],target[1],settings["fixed_handle_z_m"]-.117])
    for source in template.world.collision_objects:
        if source.id not in TRAY_IDS:continue
        obj=CollisionObject(id=source.id,operation=CollisionObject.MOVE)
        obj.header=copy.deepcopy(source.header);obj.pose=copy.deepcopy(source.pose)
        obj.primitive_poses=copy.deepcopy(source.primitive_poses)
        p=root+(xyz(source.primitive_poses[0].position)-[.55,0,.307])
        obj.primitive_poses[0].position.x,obj.primitive_poses[0].position.y,obj.primitive_poses[0].position.z=map(float,p)
        scene.world.collision_objects.append(obj)
    return scene

class CameraScene(Node):
    def __init__(self):
        super().__init__("camera_grasp_scene")
        self.c=yaml.safe_load((Path(get_package_share_directory("sorting_control"))/"config/m3b.yaml").read_text())
        self.template=scene_message();self.belief=None;self.received=0.;self.owner={};self.q={};self.episode=None
        self.allow=False;self.relative=None;self.attached_ids=False;self.carried_published=False;self.model_capture_ns=0;self.model_time_ns=0;self.last_geometry_wall=0.;self.initial_geometry_sent=False;self.last_allow=None;self.support_contact=False;self.support_cleared=False;self.grip_time_ns=0;self.last_support_sent=None;self.support_gap=None
        self.buffer=Buffer(cache_time=Duration(seconds=20.));self.listener=TransformListener(self.buffer,self)
        self.publisher=self.create_publisher(PlanningScene,"/planning_scene",1)
        self.state=self.create_publisher(String,"/sorting/perception/grasp_scene_state",1)
        self.create_subscription(TargetBeliefArray,"/sorting/targets",self.receive,1)
        self.create_subscription(String,"/sorting/arm/owner_state",lambda m:setattr(self,"owner",json.loads(m.data)),1)
        self.create_subscription(JointState,"/joint_states",lambda m:setattr(self,"q",dict(zip(m.name,m.position))),QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(CameraEpoch,"/sorting/episode",self.reset,QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_service(SetHandleContact,"/sorting/perception/set_handle_contact",self.set_contact)
        self.create_service(SetEstimatedGrasp,"/sorting/perception/set_estimated_grasp",self.set_grasp)
        self.create_timer(.01,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))
    def reset(self,m):
        if self.episode is not None and m.episode_id<=self.episode:return
        self.episode=m.episode_id;self.allow=False;self.relative=None;self.belief=None;self.carried_published=False;self.initial_geometry_sent=False;self.last_allow=None;self.support_contact=False;self.support_cleared=False;self.grip_time_ns=0;self.last_support_sent=None;self.support_gap=None
        if self.attached_ids:
            scene=PlanningScene();scene.is_diff=True;scene.robot_state.is_diff=True
            for name in TRAY_IDS:
                a=AttachedCollisionObject();a.link_name="panda_hand";a.object.id=name;a.object.operation=CollisionObject.REMOVE
                scene.robot_state.attached_collision_objects.append(a)
            self.publisher.publish(scene);self.attached_ids=False
    def receive(self,m):
        if m.episode_id!=self.episode:return
        targets=[b for b in m.targets if b.valid and b.camera_only]
        if targets:self.belief=targets[0];self.received=time.monotonic();self.tick(force=True)
    def target(self):
        require(self.belief is not None and self.belief.episode_id==self.episode,"no_current_camera_belief")
        require(time.monotonic()-self.received<.5,"belief_source_stale")
        return predict_belief(self.belief,self.get_clock().now().nanoseconds,self.c)[0]
    def authorize(self,request):
        require(request.episode_id==self.episode,"episode_mismatch")
        require(request.owner_epoch==self.owner.get("epoch"),"owner_epoch_mismatch")
        require(self.owner.get("owner") in {"HOLD","SERVO"} and not self.owner.get("busy"),"owner_not_ready")
    def set_contact(self,request,response):
        try:
            self.authorize(request);target=self.target();hand,stamp=hand_transform(self.buffer)
            require(self.get_clock().now().nanoseconds-stamp<100_000_000,"hand_tf_stale")
            if request.allow_handle_contact:
                error=hand[:3,3]-[target[0],target[1],self.c["fixed_handle_z_m"]+.1029]
                require(np.linalg.norm(error[:2])<.008 and .025<error[2]<.09,"enable_only_in_camera_pregrasp_corridor")
            else:require(all(self.q.get(n,0)>.039 for n in ("panda_finger_joint1","panda_finger_joint2")),"open_fingers_before_disabling")
            self.allow=bool(request.allow_handle_contact);response.success=True;response.reason="applied";self.tick(force=True)
        except Exception as e:response.success=False;response.reason=str(e)
        return response
    def set_grasp(self,request,response):
        try:
            self.authorize(request);require(self.owner.get("owner")=="SERVO","clear_lift_requires_servo_owner");require(request.held and self.allow,"grip_candidate_required")
            require(self.belief.track_id==request.track_id,"track_mismatch")
            require(all(.006<self.q.get(n,0)<.013 for n in ("panda_finger_joint1","panda_finger_joint2")),"finger_gap_does_not_match_handle")
            target=self.target();hand,stamp=hand_transform(self.buffer)
            require(self.get_clock().now().nanoseconds-stamp<100_000_000,"hand_tf_stale")
            world=np.eye(4);world[:3,3]=[target[0],target[1],self.c["fixed_handle_z_m"]-.117]
            self.relative=np.linalg.inv(hand)@world;self.attached_ids=True
            self.support_contact=True;self.support_cleared=False;self.grip_time_ns=self.get_clock().now().nanoseconds
            response.success=True;response.reason="estimated_moveit_attachment_only";self.tick()
        except Exception as e:response.success=False;response.reason=str(e)
        return response
    def tick(self,force=False):
        now=self.get_clock().now().nanoseconds;valid=False;reason="awaiting_camera";target=None
        try:
            if self.relative is not None:
                hand,stamp=hand_transform(self.buffer);require(now-stamp<100_000_000,"held_hand_tf_stale")
                world=hand@self.relative
                self.support_gap,self.support_cleared,self.support_contact=support_contact_state(
                    world,now,self.grip_time_ns,self.support_cleared,self.owner,self.c)
                target=[0.,0.,0.,0.];valid=True;reason="estimated_carried_geometry"
            else:target=self.target();valid=True;reason="camera_geometry"
            if (self.relative is None and (force or time.monotonic()-self.last_geometry_wall>=.05)) or (self.relative is not None and (not self.carried_published or self.support_contact!=self.last_support_sent)):
                if self.relative is None and self.initial_geometry_sent:
                    update=lightweight_camera_update(self.template,target,self.c,self.allow,self.allow!=self.last_allow)
                elif self.relative is not None and self.carried_published:
                    update=scene_for_camera(self.template,target,self.c,self.allow,self.relative,self.support_contact)
                    update.world.collision_objects=[];update.robot_state.attached_collision_objects=[]
                else:update=scene_for_camera(self.template,target,self.c,self.allow,self.relative,self.support_contact)
                self.publisher.publish(update);self.initial_geometry_sent=True;self.last_allow=self.allow;self.last_support_sent=self.support_contact
                self.carried_published=self.relative is not None;self.last_geometry_wall=time.monotonic()
                self.model_capture_ns=self.belief.header.stamp.sec*1_000_000_000+self.belief.header.stamp.nanosec;self.model_time_ns=now
        except Exception as e:
            valid=False;reason=str(e);self.support_contact=False
            if self.last_support_sent:
                # An invalid or stale transform must not retain contact permission.
                update=scene_for_camera(self.template,[0.,0.,0.,0.],self.c,self.allow,self.relative,False)
                update.world.collision_objects=[];update.robot_state.attached_collision_objects=[]
                self.publisher.publish(update);self.last_support_sent=False
        self.state.publish(String(data=json.dumps(dict(valid=valid,episode_id=self.episode,track_id=self.belief.track_id if self.belief else None,
            time_ns=now,capture_age_s=(now-(self.belief.header.stamp.sec*1_000_000_000+self.belief.header.stamp.nanosec))/1e9 if self.belief else None,model_capture_ns=self.model_capture_ns,model_time_ns=self.model_time_ns,allow_handle_contact=self.allow,estimated_attachment=self.relative is not None,physics_attachment=False,allow_tray_support_contact=self.support_contact,support_cleared=self.support_cleared,estimated_support_gap_m=self.support_gap,
            camera_only=True,reason=reason))))

def main():
    rclpy.init();node=CameraScene()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()
if __name__=="__main__":main()
