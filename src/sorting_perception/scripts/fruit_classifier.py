#!/usr/bin/env python3
"""Classify a capture-time RGB crop. This node receives no simulator labels."""
import copy,json,time
from pathlib import Path
import cv2,yaml
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock,ClockType
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from tf2_ros import Buffer,TransformListener,TransformException
from sensor_msgs.msg import Image,CameraInfo
from std_msgs.msg import String
from sorting_interfaces.msg import CameraEpoch,HandleObservation,ClassObservation,TargetBeliefArray
from ament_index_python.packages import get_package_share_directory
from classifier_core import Classifier,fruit_crop
from classifier_state import EvidenceStore,ns
from camera_core import CaptureGuard,image_rgb,intrinsics,quaternion_rotation
from camera_qos import camera_subscription_qos

class FruitClassifier(Node):
    def __init__(self):
        super().__init__('fruit_classifier');share=Path(get_package_share_directory('sorting_perception'))
        self.c=yaml.safe_load((share/'config/m3c.yaml').read_text());self.cameras=yaml.safe_load((share/'config'/self.c['camera_config']).read_text())
        self.declare_parameter('model_directory',str(share/'models/m3c_baseline'));self.model=Classifier(self.get_parameter('model_directory').value,self.c)
        self.guard=CaptureGuard();self.store=EvidenceStore(self.c);self.pending=None;self.info={};self.observations={};self.target_message=None;self.last_now=0;self.last_episode=None
        self.stats=dict(images_received=0,images_replaced=0,classified=0,rejected=0,accepted=0,last_reason='startup',maximum_inference_wall_s=0.)
        self.buffer=Buffer(cache_time=Duration(seconds=20.));self.listener=TransformListener(self.buffer,self)
        image_qos,info_qos=camera_subscription_qos();durable=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(CameraEpoch,'/sorting/episode',self.reset,durable)
        self.create_subscription(Image,'/camera/wrist/image_raw',self.image,image_qos)
        self.create_subscription(CameraInfo,'/camera/wrist/camera_info',self.metadata,info_qos)
        self.create_subscription(HandleObservation,'/sorting/perception/wrist/handle_observation',self.observation,10)
        self.create_subscription(TargetBeliefArray,'/sorting/targets',self.targets,1)
        self.publisher=self.create_publisher(ClassObservation,'/sorting/perception/class_observation',1)
        self.target_pub=self.create_publisher(TargetBeliefArray,'/sorting/classified_targets',1)
        self.state=self.create_publisher(String,'/sorting/perception/classifier_state',1)
        wall=Clock(clock_type=ClockType.STEADY_TIME);self.create_timer(.005,self.process,clock=wall);self.create_timer(.2,self.status,clock=wall);cv2.setNumThreads(1)
    def reset(self,m):
        if self.guard.reset(m.episode_id,ns(m.header.stamp)):
            self.store.reset(m.episode_id,ns(m.header.stamp));self.pending=None;self.info.clear();self.observations.clear();self.target_message=None;self.last_episode=m.episode_id
    def image(self,m):
        self.stats['images_received']+=1
        if self.pending is not None:self.stats['images_replaced']+=1
        self.pending=(m,time.monotonic())
    def metadata(self,m):
        self.info[ns(m.header.stamp)]=m
        while len(self.info)>8:self.info.pop(next(iter(self.info)))
    def observation(self,m):
        if m.valid and m.episode_id==self.guard.episode:self.observations[ns(m.header.stamp)]=m
        while len(self.observations)>8:self.observations.pop(next(iter(self.observations)))
    def targets(self,m):self.target_message=m;self.publish_targets()
    def publish_targets(self):
        if self.target_message is not None:self.target_pub.publish(self.store.enrich(self.target_message,self.get_clock().now().nanoseconds))
    def status(self):
        self.state.publish(String(data=json.dumps(dict(checkpoint='M3C',classifier_ready=True,model_sha256=self.model.sha,temperature=self.model.temperature,
            episode_id=self.guard.episode,camera_only=True,oracle_input=False,command_outputs=False,scope='single prepositioned tray; fixed wrist RGB',**self.stats))))
    def process(self):
        now=self.get_clock().now().nanoseconds
        if self.last_now and now<self.last_now:
            self.pending=None;self.info.clear();self.observations.clear();self.target_message=None;self.guard=CaptureGuard();self.store=EvidenceStore(self.c)
        self.last_now=now
        if self.pending is None:return
        image,received=self.pending;capture=ns(image.header.stamp);frame=self.cameras['cameras']['wrist']['frame'];result=None;reason='pending';bbox=[0]*4;elapsed=0.
        try:
            self.guard.check('wrist',capture,now,dict(maximum_future_stamp_s=.02,maximum_image_age_s=self.c['maximum_capture_age_s']))
            if image.header.frame_id!=frame:raise ValueError('unexpected_camera_frame')
            info=self.info.get(capture);observation=self.observations.get(capture)
            if info is None or observation is None:raise LookupError('exact_info_or_handle_observation_pending')
            rgb=image_rgb(image);k=intrinsics(info,rgb.shape,frame)
            transform=self.buffer.lookup_transform('world',frame,Time.from_msg(image.header.stamp));p=transform.transform.translation;q=transform.transform.rotation
            crop,bbox=fruit_crop(rgb,k,[p.x,p.y,p.z],quaternion_rotation([q.x,q.y,q.z,q.w]),[observation.position.x,observation.position.y,observation.position.z],self.c)
            started=time.monotonic();result=self.model.infer(crop);elapsed=time.monotonic()-started
            if elapsed>=self.c['maximum_inference_wall_s']:raise ValueError('inference_time_exceeded')
            if (self.get_clock().now().nanoseconds-capture)/1e9>=self.c['maximum_capture_age_s']:raise ValueError('classification_capture_became_stale')
            reason='accepted' if result['accepted'] else 'low_confidence'
        except (TransformException,LookupError) as error:
            if time.monotonic()-received<self.c['maximum_pending_wall_s']:return
            result=None;reason=str(error)
        except ValueError as error:result=None;reason=str(error)
        self.pending=None;self.guard.commit('wrist',capture);m=ClassObservation();m.header=copy.deepcopy(image.header);m.episode_id=int(self.guard.episode or 0);m.view_id='wrist'
        m.valid=result is not None;m.reason=reason;m.class_ids=self.c['class_ids'];m.temperature=float(self.model.temperature);m.model_sha256=self.model.sha;m.crop_xyxy=bbox;m.inference_wall_s=elapsed
        if result is not None:
            m.probabilities=result['probabilities'];m.confidence=result['confidence'];m.entropy=result['entropy'];m.accepted=result['accepted'];self.store.add(m)
            self.stats['classified']+=1;self.stats['accepted']+=int(m.accepted)
        else:self.stats['rejected']+=1
        self.stats['last_reason']=reason;self.stats['maximum_inference_wall_s']=max(self.stats['maximum_inference_wall_s'],elapsed)
        self.publisher.publish(m);self.publish_targets()
def main():
    rclpy.init();node=FruitClassifier()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
