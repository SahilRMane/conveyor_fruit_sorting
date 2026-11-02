"""Physical wrist load and bounded floor-support proof; no simulator truth input."""
import copy,math,time
import numpy as np
from rclpy.time import Time
from geometry_msgs.msg import WrenchStamped
from rclpy.qos import QoSProfile,ReliabilityPolicy
from tf2_ros import TransformException
from ownership_core import ns,require
from prediction_core import rotation

def coarse_lower_transition(position,goal,elapsed,duration,payload,loaded,tolerance):
    """A load transient away from the approach point cannot end coarse lowering."""
    position=np.asarray(position,dtype=float);goal=np.asarray(goal,dtype=float)
    require(position.shape==(3,) and goal.shape==(3,) and np.isfinite(position).all() and np.isfinite(goal).all(),
            'Invalid coarse lowering pose')
    require(all(math.isfinite(v) for v in (elapsed,duration,payload,loaded,tolerance)) and
            elapsed>=0 and duration>0 and loaded>0 and 0<tolerance<=.0015,'Invalid coarse lowering transition settings')
    distance=float(np.linalg.norm(position-goal));near=distance<tolerance
    low_load=payload<.8*loaded;finished=elapsed>duration
    ready=near and (low_load or finished)
    return dict(ready=bool(ready),reason=('near_goal_low_wrist_load' if low_load else 'coarse_goal_reached') if ready else 'continue_coarse_lower',
        position_m=position.tolist(),goal_m=goal.tolist(),distance_to_goal_m=distance,tolerance_m=tolerance,
        elapsed_s=elapsed,duration_s=duration,payload_N=payload,loaded_N=loaded,
        near_goal=bool(near),low_wrist_load=bool(low_load),trajectory_time_complete=bool(finished))

class SupportWindow:
    def __init__(self,c):
        require(all(math.isfinite(c[k]) for k in ('support_fraction','stable_span_s','payload_weight_N','approach_speed_m_s','maximum_wrench_age_s','force_velocity_gain_m_per_N_s')),'Invalid wrist support settings')
        require(0<c['support_fraction']<=.25 and c['stable_span_s']>=.02 and c['minimum_samples']>=3,'Wrist support proof is too weak')
        require(0<c['approach_speed_m_s']<=.0003 and 0<c['maximum_wrench_age_s']<=.05,'Wrist approach or age limit is too large')
        require(0<c['force_velocity_gain_m_per_N_s']<=.00005,'Wrist force gain is too large')
        self.c=c;self.first=None;self.last=0;self.count=0
    def observe(self,stamp,payload,loaded):
        require(math.isfinite(payload) and math.isfinite(loaded) and loaded>0,'Invalid wrist load')
        good=payload<self.c['support_fraction']*loaded
        if not good:self.first=None;self.count=0
        if stamp>self.last:
            if good:
                if self.first is None:self.first=stamp
                self.count+=1
            self.last=stamp
        return bool(good and self.first is not None and self.count>=self.c['minimum_samples'] and
                    stamp-self.first>=round(self.c['stable_span_s']*1e9))

class WristObserver:
    def __init__(self,controller):
        self.controller=controller;self.frames={};self.tare=None;self.sign=1.;self.loaded=None;self.last_diagnostic=None
        controller.node.create_subscription(WrenchStamped,'/panda_hand_wrench_broadcaster/wrench',self.receive,
            QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
    def receive(self,m):
        stamp=ns(m.header.stamp);self.frames[stamp]=(copy.deepcopy(m),time.monotonic())
        while len(self.frames)>16:self.frames.pop(min(self.frames))
    def sample(self):
        c=self.controller;settings=c.s['supported_release'];now=c.now_ns()
        wall=time.monotonic();rejected=dict(capture_age=0,receive_age=0,capture_time_tf=0);tf_error=None
        for stamp,(m,received) in sorted(self.frames.items(),reverse=True):
            if not -.02<=(now-stamp)/1e9<settings['maximum_wrench_age_s']:
                rejected['capture_age']+=1;continue
            if wall-received>=settings['maximum_wrench_wall_age_s']:
                rejected['receive_age']+=1;continue
            require(m.header.frame_id=='panda_hand','Unexpected wrist wrench frame')
            values=[m.wrench.force.x,m.wrench.force.y,m.wrench.force.z,m.wrench.torque.x,m.wrench.torque.y,m.wrench.torque.z]
            require(all(math.isfinite(v) for v in values),'Invalid wrist wrench values')
            try:tf=c.buffer.lookup_transform('world','panda_hand',Time.from_msg(m.header.stamp))
            except TransformException as error:
                rejected['capture_time_tf']+=1
                if tf_error is None:tf_error=str(error)
                continue
            q=tf.transform.rotation;world=rotation([q.x,q.y,q.z,q.w])@np.asarray(values[:3])
            self.last_diagnostic=dict(status='PASS',stamp_ns=stamp,age_s=(now-stamp)/1e9,force_world_N=world.tolist(),tare_world_z_N=self.tare,loaded_payload_N=self.loaded)
            return stamp,float(world[2])
        previous=self.last_diagnostic
        last_valid=previous if previous and previous.get('status')=='PASS' else (previous or {}).get('last_valid_sample')
        newest=max(self.frames) if self.frames else None
        latest_tf_ns=None;latest_tf_error=None
        try:latest_tf_ns=ns(c.buffer.lookup_transform('world','panda_hand',Time()).header.stamp)
        except TransformException as error:latest_tf_error=str(error)
        self.last_diagnostic=dict(status='UNAVAILABLE',time_ns=now,buffered_samples=len(self.frames),
            newest_wrench_stamp_ns=newest,newest_wrench_age_s=(now-newest)/1e9 if newest is not None else None,
            newest_receive_age_wall_s=wall-self.frames[newest][1] if newest is not None else None,
            maximum_wrench_age_s=settings['maximum_wrench_age_s'],maximum_wrench_wall_age_s=settings['maximum_wrench_wall_age_s'],
            rejected=rejected,capture_time_tf_error=tf_error,latest_tf_stamp_ns=latest_tf_ns,
            latest_tf_age_s=(now-latest_tf_ns)/1e9 if latest_tf_ns is not None else None,
            latest_tf_error=latest_tf_error,last_valid_sample=copy.deepcopy(last_valid))
        raise RuntimeError('Fresh capture-time wrist wrench unavailable')
    def calibrate(self):
        c=self.controller;settings=c.s['supported_release'];samples={}
        def ready():
            try:stamp,force=self.sample()
            except RuntimeError as error:
                if 'unavailable' in str(error):return False
                raise
            samples[stamp]=force
            while samples and stamp-min(samples)>round(settings['tare_span_s']*1e9)+50_000_000:samples.pop(min(samples))
            if max(samples)-min(samples)<round(settings['tare_span_s']*1e9):return False
            require(max(samples.values())-min(samples.values())<settings['tare_maximum_spread_N'],'Wrist tare is not stable')
            self.tare=float(np.mean(list(samples.values())));return True
        require(c.belt_speed==0. and all(c.q.get(n,0)>.039 for n in ('panda_finger_joint1','panda_finger_joint2')),'Wrist tare requires open fingers and stopped belt')
        c.until(ready,5.);c.result['wrist_tare_world_z_N']=self.tare
    def measure_loaded(self):
        require(self.tare is not None,'Wrist tare is missing')
        _,force=self.sample();delta=force-self.tare;weight=self.controller.s['supported_release']['payload_weight_N']
        require(.8*weight<abs(delta)<1.2*weight,'Wrist payload load outside calibration bound')
        self.sign=1. if delta>0 else -1.;self.loaded=abs(delta)
        return self.loaded
    def payload(self):
        require(self.tare is not None and self.loaded is not None,'Wrist load calibration is missing')
        stamp,force=self.sample();return stamp,self.sign*(force-self.tare)
