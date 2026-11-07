#!/usr/bin/env python3
"""One-target camera filter. Select one view per capture; no truth subscription."""
import json
from pathlib import Path
import time
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock,ClockType
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from std_msgs.msg import String
from sorting_interfaces.msg import CameraEpoch,HandleObservation,TargetBelief,TargetBeliefArray
from ament_index_python.packages import get_package_share_directory
from filter_core import MotionTrack,forecast

def ns(t):return t.sec*1_000_000_000+t.nanosec
def stamp(n):
    from builtin_interfaces.msg import Time
    return Time(sec=n//1_000_000_000,nanosec=n%1_000_000_000)

class MotionFilter(Node):
    def __init__(self):
        super().__init__("camera_motion_filter")
        self.declare_parameter('filter_config','m3b_filter.yaml')
        self.declare_parameter('timing_trace',False)
        self.timing_trace=(self.create_publisher(String,'/sorting/perception/filter_timing',QoSProfile(depth=64,reliability=ReliabilityPolicy.BEST_EFFORT))
                           if self.get_parameter('timing_trace').value is True else None)
        self.accepted_trace=None
        self.c=yaml.safe_load((Path(get_package_share_directory("sorting_perception"))/"config"/self.get_parameter('filter_config').value).read_text())
        self.track=MotionTrack(self.c);self.pending={};self.held=False;self.last_wall=0.;self.last_now=0;self.rewind_episode=None;self.publish_signature=None;self.status_wall=0.
        durable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(CameraEpoch,"/sorting/episode",self.reset,durable)
        for view in ("overhead","wrist"):
            self.create_subscription(HandleObservation,f"/sorting/perception/{view}/handle_observation",self.receive,10)
        self.create_subscription(String,"/sorting/control/pickup_state",self.control_state,1)
        self.publisher=self.create_publisher(TargetBeliefArray,"/sorting/targets",1)
        self.status=self.create_publisher(String,"/sorting/perception/filter_status",1)
        self.create_timer(.005,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))
    def decorate_belief(self,belief):
        pass
    def reset(self,m):
        if self.track.reset(m.episode_id,ns(m.header.stamp)):
            self.pending.clear();self.held=False;self.last_wall=0.;self.rewind_episode=None;self.accepted_trace=None
    def control_state(self,m):
        s=json.loads(m.data)
        if s.get("episode_id")==self.track.episode and s.get("grip_candidate"):
            self.held=True;self.pending.clear()
    def receive(self,m):
        trace=(dict(stage='filter',version=1,episode_id=m.episode_id,view_id=m.view_id,capture_ns=ns(m.header.stamp),
                    observation_sequence=m.sequence,filter_receive_wall_ns=time.monotonic_ns(),
                    filter_receive_sim_ns=self.get_clock().now().nanoseconds)
               if getattr(self,'timing_trace',None) is not None and m.view_id=='overhead' else None)
        if self.rewind_episode is not None or self.held or not m.valid or m.episode_id!=self.track.episode:return
        if m.view_id!=self.c.get("tracking_view","wrist"):return
        key=ns(m.header.stamp)
        if key<=self.track.stamp_ns:return
        group=self.pending.setdefault(key,{"received":time.monotonic(),"views":{}})
        group["views"][m.view_id]=m
        if trace is not None:group['trace']=trace
        while len(self.pending)>4:self.pending.pop(min(self.pending))
    def tick(self):
        now=self.get_clock().now().nanoseconds
        if self.last_now and now<self.last_now:
            self.track.lose("clock_rewind");self.pending.clear();self.rewind_episode=self.track.episode
            self.accepted_trace=None
        self.last_now=now
        if not self.held:
            for key,group in sorted(list(self.pending.items())):
                if len(group["views"])<2 and time.monotonic()-group["received"]<self.c["view_selection_wait_wall_s"]:break
                self.pending.pop(key,None)
                m=min(group["views"].values(),key=lambda v:sum(v.planar_covariance[::3]))
                trace=group.get('trace')
                if trace is not None:trace['filter_select_wall_ns']=time.monotonic_ns()
                accepted=self.track.update(m.episode_id,key,[m.position.x,m.position.y],m.planar_covariance,m.view_id,m.quality,now)
                if accepted:
                    self.last_wall=time.monotonic();self.accepted_trace=trace
                    if trace is not None:trace['filter_update_end_wall_ns']=time.monotonic_ns()
            self.track.expire(now)
        message=TargetBeliefArray();message.header.frame_id="world";message.header.stamp=self.get_clock().now().to_msg()
        message.episode_id=int(self.track.episode or 0)
        if self.track.x is not None:
            b=TargetBelief();b.header.frame_id="world";b.header.stamp=stamp(self.track.stamp_ns)
            b.episode_id=message.episode_id;b.track_id=self.track.track_id;b.camera_only=True
            b.valid=(not self.held and self.track.ready(now) and time.monotonic()-self.last_wall<self.c["maximum_source_wall_age_s"])
            b.status="HELD" if self.held else ("TRACKED" if b.valid else "ACQUIRING")
            b.reason=self.track.reason;b.state=self.track.x.tolist();b.covariance=self.track.p.reshape(-1).tolist()
            b.last_observation_time=b.header.stamp;b.last_update_time=message.header.stamp
            b.view_id=self.track.view;b.quality=self.track.quality;b.fixed_handle_z=self.c["fixed_handle_z_m"]
            q=self.c["fixed_orientation_xyzw"];b.fixed_orientation.x,b.fixed_orientation.y,b.fixed_orientation.z,b.fixed_orientation.w=q
            lead=min(self.c["prediction_lead_s"],self.c["maximum_prediction_horizon_s"])
            x,p=forecast(self.track.x,self.track.p,lead,self.c);b.prediction_time=stamp(self.track.stamp_ns+round(lead*1e9))
            b.prediction_state=x.tolist();b.prediction_covariance=p.reshape(-1).tolist()
            b.pose_prior="supported horizontal tray; known handle height/orientation";self.decorate_belief(b);message.targets=[b]
        signature=(message.episode_id,self.track.track_id,self.track.stamp_ns, bool(message.targets and message.targets[0].valid),self.held)
        if signature!=getattr(self,"publish_signature",None):
            trace=getattr(self,'accepted_trace',None) if message.targets else None
            if trace is not None:
                trace=dict(trace,track_id=self.track.track_id,belief_header_sim_ns=ns(message.header.stamp),
                    filter_publish_sim_ns=self.get_clock().now().nanoseconds,
                    filter_publish_enter_wall_ns=time.monotonic_ns())
            self.publisher.publish(message);self.publish_signature=signature
            if trace is not None:
                trace['filter_publish_return_wall_ns']=time.monotonic_ns()
                self.timing_trace.publish(String(data=json.dumps(trace)))
                self.accepted_trace=None
        if time.monotonic()-getattr(self,"status_wall",0.)<.1:return
        self.status_wall=time.monotonic()
        self.status.publish(String(data=json.dumps(dict(episode_id=self.track.episode,track_id=self.track.track_id,
            valid=bool(message.targets and message.targets[0].valid),tracking_view=self.c.get("tracking_view","wrist"),held=self.held,accepted=self.track.accepted,
            rejected=self.track.rejected,capture_age_s=(now-self.track.stamp_ns)/1e9 if self.track.stamp_ns else None,view_selection_wait_wall_s=self.c["view_selection_wait_wall_s"],last_reason=self.track.reason,camera_only=True,classifier_ready=False))))

def main():
    rclpy.init();node=MotionFilter()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()
if __name__=="__main__":main()
