#!/usr/bin/env python3
"""Estimated carried/placed tray geometry with bounded destination contact."""
import json,time
from pathlib import Path
import numpy as np,yaml,rclpy
from std_msgs.msg import String
from moveit_msgs.msg import PlanningScene,AttachedCollisionObject,CollisionObject
from sorting_interfaces.srv import SetCameraPlacement
from ament_index_python.packages import get_package_share_directory
from camera_scene import CameraScene,hand_transform,support_contact_state,TRAY_IDS,lightweight_camera_update
from sort_core import placement_scene,footprint_inside_estimate,resting_pose_from_release,supported_floor_pose
from wrist_support import WristObserver
from types import SimpleNamespace
from ownership_core import require

class SortScene(CameraScene):
    def __init__(self):
        self.floor_id='';self.floor_since=0;self.placed=None;self.last_signature=None;self.release_hand=None;self.release_ns=0
        super().__init__();self.sort_config=yaml.safe_load((Path(get_package_share_directory('sorting_control'))/'config/m3d.yaml').read_text())
        self.floor_supported=False;self.tare_samples={}
        adapter=SimpleNamespace(node=self,buffer=self.buffer,s=self.sort_config,now_ns=lambda:self.get_clock().now().nanoseconds)
        self.wrist=WristObserver(adapter)
        self.create_service(SetCameraPlacement,'/sorting/perception/set_camera_placement',self.placement)
    def reset(self,m):
        if self.episode is not None and m.episode_id<=self.episode:return
        super().reset(m);self.floor_id='';self.floor_since=0;self.placed=None;self.last_signature=None;self.release_hand=None;self.release_ns=0
        self.floor_supported=False;self.tare_samples={}
        if hasattr(self,'wrist'):self.wrist.tare=None;self.wrist.loaded=None;self.wrist.frames.clear()
    def placement(self,request,response):
        try:
            require(request.episode_id==self.episode and request.owner_epoch==self.owner.get('epoch'),'Placement identity/owner mismatch')
            require(self.owner.get('owner') in ('HOLD','SERVO','PLANNED') and not self.owner.get('busy'),'Placement owner not stable')
            require(request.bin_id in self.sort_config['bins'],'Unknown destination')
            require(self.relative is not None and self.support_cleared,'A clear-lift carried model is required')
            hand,stamp=hand_transform(self.buffer);now=self.get_clock().now().nanoseconds;require(-20_000_000<=now-stamp<100_000_000,'Placement hand TF stale')
            world=hand@self.relative;require(footprint_inside_estimate(world,request.bin_id,self.sort_config),'Estimated tray footprint is outside the bin')
            if request.operation=='enable_floor':
                require(self.floor_id in ('',request.bin_id+'_floor'),'Destination cannot change while carried')
                self.floor_id=request.bin_id+'_floor';self.floor_since=now
            elif request.operation=='register_floor_support':
                require(self.owner.get('owner')=='SERVO','Wrist support registration requires Servo')
                require(self.floor_id==request.bin_id+'_floor','Destination contact was not authorized')
                _,payload=self.wrist.payload()
                require(payload<self.sort_config['supported_release']['support_fraction']*self.wrist.loaded,'Physical wrist does not show floor support')
                require(all(.006<self.q.get(n,0)<.013 for n in ('panda_finger_joint1','panda_finger_joint2')),'Support registration requires a closed grip')
                placed=supported_floor_pose(world,request.bin_id,self.sort_config)
                self.relative=np.linalg.inv(hand)@placed;self.floor_supported=True
            elif request.operation=='detach':
                require(self.floor_id==request.bin_id+'_floor','Destination contact was not authorized')
                require(all(self.q.get(n,0)>.039 for n in ('panda_finger_joint1','panda_finger_joint2')),'Open fingers before model detach')
                placed=supported_floor_pose(world,request.bin_id,self.sort_config) if getattr(self,'floor_supported',False) else resting_pose_from_release(world,request.bin_id,self.sort_config)
                remove=PlanningScene(is_diff=True);remove.robot_state.is_diff=True
                for name in TRAY_IDS:
                    obj=AttachedCollisionObject(link_name='panda_hand');obj.object.id=name;obj.object.operation=CollisionObject.REMOVE;remove.robot_state.attached_collision_objects.append(obj)
                self.publisher.publish(remove);self.placed=placed;self.relative=None;self.attached_ids=False;self.floor_id='';self.support_contact=False;self.release_hand=hand[:3,3].copy();self.release_ns=now
            else:raise RuntimeError('Unknown placement operation')
            self.last_signature=None;self.tick(force=True);response.success=True;response.reason='estimated_model_only'
        except Exception as error:response.success=False;response.reason=str(error)
        return response
    def tick(self,force=False):
        if not hasattr(self,'sort_config'):return
        now=self.get_clock().now().nanoseconds;valid=False;reason='awaiting_camera';target=[0,0,0,0];world=None
        try:
            if self.relative is not None:
                hand,stamp=hand_transform(self.buffer);require(-20_000_000<=now-stamp<100_000_000,'Held hand TF stale');world=hand@self.relative
                if self.floor_id:
                    # The selected bin has its own authorized floor. The source
                    # support plane does not bound motion at that destination.
                    self.support_contact=False
                else:self.support_gap,self.support_cleared,self.support_contact=support_contact_state(world,now,self.grip_time_ns,self.support_cleared,self.owner,self.c)
                if self.floor_id:
                    bin_id=self.floor_id[:-6]
                    require(footprint_inside_estimate(world,bin_id,self.sort_config),'Estimated carried footprint left destination')
                    require(now-self.floor_since<self.sort_config['placement_contact_timeout_s']*1e9,'Destination contact allowance expired')
                reason='estimated_carried_geometry'
            elif self.placed is not None:
                world=self.placed;reason='estimated_placed_geometry'
                if self.allow:
                    hand,stamp=hand_transform(self.buffer);require(-20_000_000<=now-stamp<100_000_000,'Retreat hand TF stale')
                    if hand[2,3]-self.release_hand[2]>.035:self.allow=False
                    else:require(now-self.release_ns<10_000_000_000,'Finger retreat allowance expired')
            else:target=self.target();reason='camera_geometry'
            signature=(self.relative is not None,self.support_contact,self.floor_id,self.placed is not None,self.allow)
            due=self.relative is None and self.placed is None and (force or time.monotonic()-self.last_geometry_wall>=.05)
            if due or signature!=self.last_signature:
                if due and self.initial_geometry_sent and signature==self.last_signature:
                    message=lightweight_camera_update(self.template,target,self.c,self.allow)
                else:message=placement_scene(self.template,target,self.c,self.allow,self.relative,self.support_contact,self.floor_id,self.placed)
                self.initial_geometry_sent=True
                self.publisher.publish(message);self.last_signature=signature;self.last_geometry_wall=time.monotonic();self.carried_published=self.relative is not None
                if self.belief:self.model_capture_ns=self.belief.header.stamp.sec*1_000_000_000+self.belief.header.stamp.nanosec
                self.model_time_ns=now
            valid=True
        except Exception as error:
            reason=str(error);self.support_contact=False
            if self.floor_id or self.last_support_sent:
                self.floor_id='';self.last_signature=None
                message=placement_scene(self.template,target,self.c,self.allow,self.relative,False)
                message.world.collision_objects=[];message.robot_state.attached_collision_objects=[];self.publisher.publish(message)
        self.last_support_sent=self.support_contact
        if hasattr(self,'wrist'):
            try:
                stamp,force=self.wrist.sample()
                if self.relative is None and not self.floor_id and self.owner.get('owner')=='HOLD' and all(self.q.get(n,0)>.039 for n in ('panda_finger_joint1','panda_finger_joint2')):
                    self.tare_samples[stamp]=force
                    while self.tare_samples and stamp-min(self.tare_samples)>250_000_000:self.tare_samples.pop(min(self.tare_samples))
                    if max(self.tare_samples)-min(self.tare_samples)>=200_000_000 and max(self.tare_samples.values())-min(self.tare_samples.values())<.1:
                        self.wrist.tare=float(np.mean(list(self.tare_samples.values())))
                elif self.relative is not None and self.support_cleared and not self.floor_id and self.wrist.tare is not None:
                    delta=force-self.wrist.tare;weight=self.sort_config['supported_release']['payload_weight_N']
                    if .8*weight<abs(delta)<1.2*weight:self.wrist.loaded=abs(delta);self.wrist.sign=1. if delta>0 else -1.
            except RuntimeError:pass # Controller operations require fresh data explicitly.
        self.state.publish(String(data=json.dumps(dict(valid=valid,reason=reason,time_ns=now,episode_id=self.episode,track_id=self.belief.track_id if self.belief else None,
            belief_capture_ns=self.belief.header.stamp.sec*1_000_000_000+self.belief.header.stamp.nanosec if self.belief else None,
            capture_age_s=(now-(self.belief.header.stamp.sec*1_000_000_000+self.belief.header.stamp.nanosec))/1e9 if self.belief else None,
            camera_only=True,estimated_attachment=self.relative is not None,physics_attachment=False,allow_handle_contact=self.allow,allow_tray_support_contact=self.support_contact,
            floor_supported=getattr(self,'floor_supported',False),support_pose_source='physical_wrist_and_known_floor' if getattr(self,'floor_supported',False) else None,
            support_cleared=self.support_cleared,estimated_support_gap_m=self.support_gap,model_capture_ns=self.model_capture_ns,model_time_ns=self.model_time_ns,
            destination_floor=self.floor_id,estimated_body_matrix=world.tolist() if world is not None else None,estimated_placed=self.placed is not None))))
def main():
    rclpy.init();node=SortScene()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
