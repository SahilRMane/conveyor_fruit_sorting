#!/usr/bin/env python3
"""Localize the neutral marker from RGB, CameraInfo and capture-time encoder TF."""
import json
from pathlib import Path
import time
from functools import partial
import cv2
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock,ClockType
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from rclpy.utilities import get_rmw_implementation_identifier
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import String
from sorting_interfaces.msg import CameraEpoch,HandleObservation
from tf2_ros import Buffer,TransformListener,TransformException
import yaml
from camera_core import CaptureGuard,image_rgb,intrinsics,localize,quaternion_rotation
from camera_qos import camera_subscription_qos

def ns(stamp):return stamp.sec*1_000_000_000+stamp.nanosec

class HandleLocalizer(Node):
    def __init__(self):
        super().__init__("handle_localizer")
        share=Path(get_package_share_directory("sorting_perception"))
        self.declare_parameter("camera_config","cameras.yaml")
        self.declare_parameter("localization_config","m3a.yaml")
        self.declare_parameter("timing_trace",False)
        self.declare_parameter("process_period_s",0.01)
        self.timing_trace=(self.create_publisher(String,"/sorting/perception/localizer_timing",QoSProfile(depth=64,reliability=ReliabilityPolicy.BEST_EFFORT))
                           if self.get_parameter("timing_trace").value is True else None)
        self.image_trace={};self.info_times={}
        self.trace_rmw=get_rmw_implementation_identifier() if self.timing_trace is not None else None
        self.delivery_clock_pair=None
        self.c=yaml.safe_load((share/"config"/self.get_parameter("localization_config").value).read_text())
        self.cameras=yaml.safe_load((share/"config"/self.get_parameter("camera_config").value).read_text())
        self.guard=CaptureGuard();self.pending={};self.info={};self.sequence=0;self.epoch_message=None;self.clock_episode=None
        self.stats={v:dict(received=0,processed=0,valid=0,rejected=0,pending_replaced=0,
                          capture_gap_estimate=0,last_received_ns=0,reasons={}) for v in ("overhead","wrist")}
        self.buffer=Buffer(cache_time=Duration(seconds=20.0));self.listener=TransformListener(self.buffer,self)
        qos,metadata_qos=camera_subscription_qos()
        self.image_qos=qos;self.metadata_qos=metadata_qos
        epoch_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(CameraEpoch,"/sorting/episode",self.epoch,epoch_qos)
        self.observation_publishers={}
        for view in self.stats:
            self.observation_publishers[view]=self.create_publisher(HandleObservation,f"/sorting/perception/{view}/handle_observation",1)
            callback=(partial(self.image_with_info,view) if self.timing_trace is not None and view=='overhead'
                      else lambda m,v=view:self.image(v,m))
            self.create_subscription(Image,f"/camera/{view}/image_raw",callback,qos)
            self.create_subscription(CameraInfo,f"/camera/{view}/camera_info",lambda m,v=view:self.camera_info(v,m),metadata_qos)
        self.status=self.create_publisher(String,"/sorting/perception/camera_status",1)
        wall=Clock(clock_type=ClockType.STEADY_TIME)
        period=self.get_parameter("process_period_s").value
        if not .002<=period<=.01:raise ValueError("Localizer process period must be 2 to 10 ms")
        self.create_timer(period,self.process,clock=wall);self.create_timer(0.5,self.publish_status,clock=wall)
        self.last_clock_ns=0;cv2.setNumThreads(1)

    def epoch(self,message):
        if self.guard.reset(message.episode_id,ns(message.header.stamp)):
            self.epoch_message=message
            self.pending.clear();self.info.clear()
            self.image_trace.clear();self.info_times.clear()

    def camera_info(self,view,message):
        if getattr(self,'timing_trace',None) is not None and view=='overhead':
            times=self.info_times.setdefault(view,{})
            times.setdefault(ns(message.header.stamp),time.monotonic_ns())
            while len(times)>8:times.pop(next(iter(times)))
        cache=self.info.setdefault(view,{})
        cache[ns(message.header.stamp)]=message
        while len(cache)>8:cache.pop(next(iter(cache)))

    def image_with_info(self,view,message,info):
        # Jazzy selects message-info delivery only if two callback arguments are required.
        before=time.monotonic_ns();system=time.time_ns();after=time.monotonic_ns()
        self.image(view,message)
        trace=self.image_trace.get(view)
        if trace is None:return
        metadata=info if isinstance(info,dict) else {}
        midpoint=(before+after)//2;previous=self.delivery_clock_pair
        trace['image_arrival_wall_ns']=before
        trace['camera_delivery']=dict(rmw_implementation=self.trace_rmw,
            source_timestamp_system_ns=metadata.get('source_timestamp'),
            received_timestamp_system_ns=metadata.get('received_timestamp'),
            publication_sequence_number=metadata.get('publication_sequence_number'),
            reception_sequence_number=metadata.get('reception_sequence_number'),
            callback_entry_system_ns=system,callback_entry_monotonic_before_ns=before,
            callback_entry_monotonic_after_ns=after,
            system_monotonic_delta_ns=((system-previous[0])-(midpoint-previous[1])) if previous else None,
            offset_comparison_uncertainty_ns=((after-before)+previous[2])//2 if previous else None)
        self.delivery_clock_pair=(system,midpoint,after-before)

    def image(self,view,message):
        if getattr(self,'timing_trace',None) is not None and view=='overhead':
            self.image_trace[view]=dict(stage='localizer',version=1,view_id=view,capture_ns=ns(message.header.stamp),
                image_arrival_wall_ns=time.monotonic_ns(),image_arrival_sim_ns=self.get_clock().now().nanoseconds,
                attempts=0,metadata_waits=0,tf_waits=0,attempt_work_wall_ns=0)
        s=self.stats[view];s["received"]+=1;stamp=ns(message.header.stamp)
        if view in self.pending:s["pending_replaced"]+=1
        if s["last_received_ns"] and stamp>s["last_received_ns"]:
            s["capture_gap_estimate"]+=max(0,round((stamp-s["last_received_ns"])/1e9*self.cameras["publish_rate_hz"])-1)
        s["last_received_ns"]=max(stamp,s["last_received_ns"])
        self.pending[view]=(message,time.monotonic())

    def publish_status(self):
        value=dict(episode_id=self.guard.episode,reset_capture_barrier_ns=self.guard.barrier_ns,
                   world_frame="world",capture_time_tf=True,oracle_input=False,
                   camera_info_reliability=self.metadata_qos.reliability.name,
                   camera_info_queue_depth=self.metadata_qos.depth,image_queue_depth=self.image_qos.depth,
                   covariance_source="assumed_pixel_noise; calibration checkpoint only",views=self.stats)
        self.status.publish(String(data=json.dumps(value)))

    def process(self):
        now=self.get_clock().now().nanoseconds
        if self.last_clock_ns and now<self.last_clock_ns:
            self.pending.clear();self.info.clear();previous=self.clock_episode;self.guard=CaptureGuard()
            if getattr(self,'timing_trace',None) is not None:self.image_trace.clear();self.info_times.clear()
            # Wait for reset identity. A reset notification can arrive before
            # the clock packet; retain it only if it advances the old episode.
            m=self.epoch_message
            if m is not None and previous is not None and m.episode_id>previous and ns(m.header.stamp)<=now:
                self.guard.reset(m.episode_id,ns(m.header.stamp))
            for s in self.stats.values():s["last_received_ns"]=0
        self.last_clock_ns=now
        self.clock_episode=self.guard.episode
        for view,(image,received) in list(self.pending.items()):
            stamp=ns(image.header.stamp);frame=self.cameras["cameras"][view]["frame"]
            result=None;reason=None;started=time.monotonic();capture_accepted=False
            trace=getattr(self,'image_trace',{}).get(view)
            if trace is not None:
                attempt_start=time.monotonic_ns();trace['attempts']+=1
                trace.setdefault('first_attempt_wall_ns',attempt_start)
            try:
                self.guard.check(view,stamp,now,self.c)
                capture_accepted=True
                if image.header.frame_id!=frame:raise ValueError("unexpected_camera_frame")
                info=self.info.get(view,{}).get(stamp)
                if info is None:raise LookupError("matching_camera_info_pending")
                if trace is not None:
                    trace.setdefault('metadata_ready_wall_ns',time.monotonic_ns())
                    trace['camera_info_arrival_wall_ns']=self.info_times.get(view,{}).get(stamp)
                if trace is not None:trace['image_prepare_start_wall_ns']=time.monotonic_ns()
                rgb=image_rgb(image);k=intrinsics(info,rgb.shape,frame)
                if trace is not None:trace['image_prepare_end_wall_ns']=time.monotonic_ns()
                # Never substitute the latest transform for a capture-time lookup.
                if trace is not None:trace['tf_lookup_start_wall_ns']=time.monotonic_ns()
                transform=self.buffer.lookup_transform("world",frame,Time.from_msg(image.header.stamp))
                if trace is not None:trace['tf_ready_wall_ns']=time.monotonic_ns()
                p=transform.transform.translation;q=transform.transform.rotation
                if trace is not None:trace['localization_start_wall_ns']=time.monotonic_ns()
                result=localize(rgb,k,[p.x,p.y,p.z],quaternion_rotation([q.x,q.y,q.z,q.w]),self.c)
                if trace is not None:trace['localization_end_wall_ns']=time.monotonic_ns()
            except (TransformException,LookupError) as error:
                if trace is not None:trace['metadata_waits' if isinstance(error,LookupError) else 'tf_waits']+=1
                if time.monotonic()-received<self.c["maximum_pending_wall_s"]:continue
                reason="capture_transform_or_info_unavailable"
            except ValueError as error:reason=str(error)
            finally:
                if trace is not None:trace['attempt_work_wall_ns']+=time.monotonic_ns()-attempt_start
            self.pending.pop(view,None);self.sequence+=1
            message=HandleObservation();message.header.stamp=image.header.stamp;message.header.frame_id="world"
            message.episode_id=int(self.guard.episode or 0);message.sequence=self.sequence
            message.view_id=view;message.camera_frame=frame;message.valid=result is not None
            message.reason="localized" if result is not None else reason
            message.compute_wall_s=time.monotonic()-started
            publish_age=(self.get_clock().now().nanoseconds-stamp)/1e9
            message.covariance_source="assumed_pixel_noise; not an empirically calibrated filter"
            if result is not None:
                message.position.x,message.position.y,message.position.z=result["position"]
                message.planar_covariance=result["planar_covariance"]
                message.pixel_u=result["pixel_u"];message.pixel_v=result["pixel_v"]
                message.area_px=result["area_px"];message.quality=result["quality"]
            if capture_accepted:self.guard.commit(view,stamp)
            s=self.stats[view];s["processed"]+=1;s["last_compute_wall_s"]=message.compute_wall_s
            s["maximum_compute_wall_s"]=max(s.get("maximum_compute_wall_s",0.),message.compute_wall_s)
            s["last_publish_capture_age_s"]=publish_age
            if result is None:
                s["rejected"]+=1;s["reasons"][reason]=s["reasons"].get(reason,0)+1
            else:s["valid"]+=1
            if trace is not None:
                trace.update(episode_id=message.episode_id,observation_sequence=message.sequence,valid=message.valid,
                    observation_publish_sim_ns=self.get_clock().now().nanoseconds,
                    observation_publish_enter_wall_ns=time.monotonic_ns())
            self.observation_publishers[view].publish(message)
            if trace is not None:
                trace['observation_publish_return_wall_ns']=time.monotonic_ns()
                self.image_trace.pop(view,None)
                self.timing_trace.publish(String(data=json.dumps(trace)))

def main():
    rclpy.init();node=HandleLocalizer()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()

if __name__=="__main__":main()
