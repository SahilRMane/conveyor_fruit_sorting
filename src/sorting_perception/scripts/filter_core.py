"""Capture-time constant-velocity filter. No ROS or simulator imports."""
import math
import numpy as np

def require(ok,reason):
    if not ok:raise ValueError(reason)

def matrices(dt,acceleration_sigma):
    require(math.isfinite(dt) and dt>=0,"invalid_dt")
    f=np.eye(4);f[0,2]=f[1,3]=dt
    g=np.array([[dt*dt/2,0],[0,dt*dt/2],[dt,0],[0,dt]])
    return f,(acceleration_sigma**2)*(g@g.T)

def forecast(state,covariance,dt,c):
    f,q=matrices(dt,c["acceleration_sigma_m_s2"])
    p=f@np.asarray(covariance)@f.T+q
    return f@np.asarray(state),.5*(p+p.T)

class MotionTrack:
    def __init__(self,c):
        self.c=c;self.episode=None;self.track_id=0;self.barrier_ns=0;self.x=None;self.p=None
        self.stamp_ns=0;self.first_ns=0;self.samples=0;self.view="";self.quality=0.;self.reason="no_observation"
        self.rejected=0;self.accepted=0
    def reset(self,episode,barrier):
        if self.episode is not None and episode<=self.episode:return False
        self.episode=int(episode);self.barrier_ns=int(barrier);self.lose("reset");return True
    def lose(self,reason):
        self.x=self.p=None;self.stamp_ns=self.first_ns=0;self.samples=0;self.reason=reason
    def update(self,episode,stamp_ns,position,covariance,view,quality,now_ns):
        try:
            require(episode==self.episode,"episode_mismatch")
            require(stamp_ns>self.barrier_ns,"capture_before_reset")
            require(-self.c["maximum_future_stamp_s"]<=(now_ns-stamp_ns)/1e9<self.c["maximum_capture_age_s"],"capture_age")
            require(stamp_ns>self.stamp_ns,"out_of_order_or_duplicate")
            z=np.asarray(position,dtype=float);r=np.asarray(covariance,dtype=float).reshape(2,2)
            require(z.shape==(2,) and np.all(np.isfinite(z)) and np.all(np.isfinite(r)) and
                    np.allclose(r,r.T) and np.linalg.eigvalsh(r).min()>=0,"invalid_measurement")
            require(self.c["workspace_x_m"][0]<=z[0]<=self.c["workspace_x_m"][1] and
                    self.c["workspace_y_m"][0]<=z[1]<=self.c["workspace_y_m"][1],"outside_workspace")
            r=r+(self.c["measurement_sigma_floor_m"]**2)*np.eye(2)
            if self.x is not None and (stamp_ns-self.stamp_ns)/1e9>self.c["track_timeout_s"]:self.lose("reacquiring")
            if self.x is None:
                self.track_id+=1;self.x=np.r_[z,0.,0.];self.p=np.zeros((4,4));self.p[:2,:2]=r
                self.p[2:,2:]=np.eye(2)*self.c["initial_velocity_sigma_m_s"]**2;self.first_ns=stamp_ns;self.samples=1
            else:
                x,p=forecast(self.x,self.p,(stamp_ns-self.stamp_ns)/1e9,self.c)
                h=np.c_[np.eye(2),np.zeros((2,2))];innovation=z-h@x;s=h@p@h.T+r
                require(float(innovation@np.linalg.solve(s,innovation))<self.c["innovation_gate_chi2"],"innovation_gate")
                k=np.linalg.solve(s,(p@h.T).T).T;candidate=x+k@innovation
                require(np.linalg.norm(candidate[2:])<self.c["maximum_target_speed_m_s"],"target_speed")
                a=np.eye(4)-k@h;p=a@p@a.T+k@r@k.T
                # Keep a documented velocity uncertainty floor. This also
                # admits the commanded scenario's initial speed change.
                p[2:,2:]+=np.diag(np.maximum(0,self.c["velocity_sigma_floor_m_s"]**2-np.diag(p[2:,2:])))
                self.x=candidate;self.p=.5*(p+p.T);self.samples+=1
            self.stamp_ns=int(stamp_ns);self.view=view;self.quality=float(quality);self.reason="accepted";self.accepted+=1
            return True
        except ValueError as error:self.reason=str(error);self.rejected+=1;return False
    def ready(self,now_ns):
        return (self.x is not None and self.samples>=self.c["minimum_initial_samples"] and
                (self.stamp_ns-self.first_ns)/1e9>=self.c["minimum_initial_span_s"] and
                0<=(now_ns-self.stamp_ns)/1e9<self.c["maximum_capture_age_s"])
    def expire(self,now_ns):
        if self.x is not None and (now_ns-self.stamp_ns)/1e9>self.c["track_timeout_s"]:self.lose("track_timeout")

def belief_prediction(message,now_ns,c):
    stamp=message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
    dt=(now_ns-stamp)/1e9
    require(message.valid and message.camera_only and message.header.frame_id=="world","invalid_belief")
    require(-.02<=dt<c["maximum_capture_age_s"],"belief_capture_stale")
    x,p=forecast(np.asarray(message.state),np.asarray(message.covariance).reshape(4,4),max(dt,0),c)
    return x,p,stamp
