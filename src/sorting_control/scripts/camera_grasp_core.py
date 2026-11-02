"""Camera closure and encoder grip-candidate proofs; no ROS or truth imports."""
import numpy as np

def predict_belief(message,now_ns,c):
    stamp=message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec;dt=(now_ns-stamp)/1e9
    if not (message.valid and message.camera_only and message.header.frame_id=="world" and -.02<=dt<c["maximum_capture_age_s"]):
        raise ValueError("Camera belief is invalid, stale or future")
    dt=max(0.,dt);f=np.eye(4);f[0,2]=f[1,3]=dt
    g=np.array([[dt*dt/2,0],[0,dt*dt/2],[dt,0],[0,dt]])
    p=np.asarray(message.covariance).reshape(4,4);state=np.asarray(message.state)
    if not (np.all(np.isfinite(state)) and np.all(np.isfinite(p)) and np.allclose(p,p.T) and np.linalg.eigvalsh(p).min()>=-1e-12):raise ValueError("Invalid belief state or covariance")
    p=f@p@f.T+(c["acceleration_sigma_m_s2"]**2)*(g@g.T)
    return f@np.asarray(message.state),.5*(p+p.T),stamp

class CameraClosureWindow:
    def __init__(self,c):self.c=c;self.first=None;self.last=0;self.count=0;self.captures=set();self.reasons=[]
    def observe(self,s):
        c=self.c;e=np.asarray(s["position_error_m"]);v=np.asarray(s["relative_velocity_m_s"])
        p=np.asarray(s["predicted_covariance"]).reshape(4,4);reasons=[]
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(p)):reasons.append("nonfinite")
        if not all(np.isfinite(s[key]) for key in ("orientation_error_rad","capture_age_s","tf_age_s","remaining_s")):reasons.append("nonfinite")
        if not reasons and (not np.allclose(p,p.T) or np.linalg.eigvalsh(p).min()<-1e-12):reasons.append("invalid_covariance")
        if reasons:
            self.first=None;self.count=0;self.captures.clear();self.reasons=reasons;return False
        if np.linalg.norm(e[:2])>=c["close_position_xy_m"] or abs(e[2])>=c["close_position_z_m"]:reasons.append("position")
        if np.linalg.norm(v)>=c["close_relative_speed_m_s"]:reasons.append("relative_velocity")
        if s["orientation_error_rad"]>=c["close_orientation_rad"]:reasons.append("orientation")
        if not -.02<=s["capture_age_s"]<c["maximum_capture_age_s"]:reasons.append("capture_age")
        if not -.02<=s["tf_age_s"]<c["maximum_tf_age_s"]:reasons.append("tf_age")
        if np.sqrt(max(np.linalg.eigvalsh(p[:2,:2])))>=c["maximum_position_sigma_m"]:reasons.append("position_uncertainty")
        if np.sqrt(max(np.linalg.eigvalsh(p[2:,2:])))>=c["maximum_velocity_sigma_m_s"]:reasons.append("velocity_uncertainty")
        if s["remaining_s"]<=c["deadline_reserve_s"]:reasons.append("deadline")
        if not s["descent_complete"]:reasons.append("descent")
        stamp=s["tf_stamp_ns"];self.reasons=reasons
        if reasons or stamp<self.last:self.first=None;self.count=0;self.captures.clear()
        if not reasons and stamp>self.last:
            if self.first is None:self.first=stamp
            self.count+=1;self.captures.add(s["capture_ns"])
        self.last=max(self.last,stamp)
        return (self.first is not None and self.count>=c["close_minimum_samples"] and
                len(self.captures)>=c["close_minimum_camera_captures"] and (stamp-self.first)/1e9>=c["close_stable_span_s"])

class FingerWindow:
    def __init__(self,c):self.c=c;self.first=None;self.last=0;self.count=0
    def observe(self,stamp,positions,velocities):
        good=(all(self.c["finger_gap_min_m"]<p<self.c["finger_gap_max_m"] for p in positions) and
              max(abs(v) for v in velocities)<self.c["finger_stopped_velocity_m_s"])
        if not good:self.first=None;self.count=0
        if good and stamp>self.last:
            if self.first is None:self.first=stamp
            self.count+=1
        self.last=max(self.last,stamp)
        return self.first is not None and self.count>=self.c["finger_minimum_samples"] and (stamp-self.first)/1e9>=self.c["finger_stable_span_s"]


class ClosureProof:
    """Consume one coherent controller snapshot once; expire its authorization."""
    def __init__(self,c):
        self.c=c;self.window=CameraClosureWindow(c);self.last_time=0;self.sample=None;self.ready=False
    def observe(self,s):
        stamp=s["time_ns"]
        if stamp<=self.last_time:return self.ready
        self.last_time=stamp;self.sample=s;self.ready=self.window.observe(s)
        return self.ready
    def authorized(self,now_ns):
        if not (self.ready and self.sample is not None):return False
        dt=(now_ns-self.last_time)/1e9;s=self.sample;c=self.c
        if not (0<=dt<c.get("maximum_closure_snapshot_age_s",.025) and
                -.02<=(now_ns-s["capture_ns"])/1e9<c["maximum_capture_age_s"] and
                s["remaining_s"]-dt>c["deadline_reserve_s"]):return False
        f=np.eye(4);f[0,2]=f[1,3]=dt
        g=np.array([[dt*dt/2,0],[0,dt*dt/2],[dt,0],[0,dt]])
        p=f@np.asarray(s["predicted_covariance"]).reshape(4,4)@f.T+c["acceleration_sigma_m_s2"]**2*(g@g.T)
        error=np.asarray(s["position_error_m"]);relative=np.asarray(s["relative_velocity_m_s"])
        return bool(np.sqrt(max(np.linalg.eigvalsh(p[:2,:2])))<c["maximum_position_sigma_m"] and
            np.sqrt(max(np.linalg.eigvalsh(p[2:,2:])))<c["maximum_velocity_sigma_m_s"] and
            np.linalg.norm(error[:2])+np.linalg.norm(relative[:2])*dt<c["close_position_xy_m"] and
            abs(error[2])+abs(relative[2])*dt<c["close_position_z_m"] and
            s["orientation_error_rad"]+c["maximum_servo_angular_speed_rad_s"]*dt<c["close_orientation_rad"])
