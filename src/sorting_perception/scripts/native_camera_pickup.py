"""Native RGB-controlled physical trial; no ROS participant or live Servo."""
import ctypes as ct
import json
from pathlib import Path
import numpy as np
import cv2
import yaml
from types import SimpleNamespace as NS
from camera_core import localize,quaternion_rotation
from filter_core import MotionTrack,forecast
from camera_grasp_core import CameraClosureWindow,ClosureProof,FingerWindow

class NativePlant:
    def __init__(self,library,scene):
        self.lib=ct.CDLL(str(library));self.lib.nc_create.argtypes=[ct.c_char_p];self.lib.nc_create.restype=ct.c_void_p
        self.lib.nc_error.restype=ct.c_char_p;self.lib.nc_truth.argtypes=[ct.c_void_p];self.lib.nc_truth.restype=ct.c_char_p
        double=ct.POINTER(ct.c_double)
        for name,types in {"nc_destroy":[ct.c_void_p],"nc_shape":[ct.c_void_p,ct.c_char_p,ct.POINTER(ct.c_int)],"nc_feedback":[ct.c_void_p,double],"nc_pregrasp":[ct.c_void_p,double],
            "nc_settings":[ct.c_void_p,double],"nc_control":[ct.c_void_p,double,double,ct.c_double],"nc_step":[ct.c_void_p,ct.c_double,ct.c_double],
            "nc_wrist_wrench":[ct.c_void_p,double],"nc_floor_peaks":[ct.c_void_p,double],"nc_gripper":[ct.c_void_p,ct.c_double],"nc_open":[ct.c_void_p],"nc_resume":[ct.c_void_p],"nc_close":[ct.c_void_p],"nc_hold_reference":[ct.c_void_p],"nc_hold":[ct.c_void_p],"nc_render":[ct.c_void_p,ct.c_char_p,ct.POINTER(ct.c_ubyte),double]}.items():
            getattr(self.lib,name).argtypes=types
        self.p=self.lib.nc_create(str(scene).encode())
        if not self.p:raise RuntimeError(self.lib.nc_error().decode())
    def call(self,name,*args):
        if getattr(self.lib,name)(self.p,*args)<0:raise RuntimeError(self.lib.nc_error().decode())
    def array(self,value):return np.ascontiguousarray(value,dtype=np.float64).ctypes.data_as(ct.POINTER(ct.c_double))
    def feedback(self):
        a=np.zeros(28);self.call("nc_feedback",a.ctypes.data_as(ct.POINTER(ct.c_double)));return a
    def truth(self):return json.loads(self.lib.nc_truth(self.p))
    def floor_peaks(self):
        """Grader-only maximum since the last read, sampled each physics step."""
        values=np.zeros(2);self.call('nc_floor_peaks',values.ctypes.data_as(ct.POINTER(ct.c_double)))
        return values
    def wrist_wrench(self):
        values=np.zeros(7);self.call('nc_wrist_wrench',values.ctypes.data_as(ct.POINTER(ct.c_double)))
        return values
    def render(self,view):
        shape=np.zeros(2,np.int32);self.call("nc_shape",view.encode(),shape.ctypes.data_as(ct.POINTER(ct.c_int)))
        w,h=map(int,shape);assert 0<w<=4096 and 0<h<=4096
        rgb=np.zeros((h,w,3),np.uint8);a=np.zeros(16)
        self.call("nc_render",view.encode(),rgb.ctypes.data_as(ct.POINTER(ct.c_ubyte)),a.ctypes.data_as(ct.POINTER(ct.c_double)))
        assert list(a[1:3])==[w,h];f=a[2]/2/np.tan(np.deg2rad(a[3])/2)
        k=np.array([[f,0,a[1]/2],[0,f,a[2]/2],[0,0,1.]])
        return rgb,k,a[4:7],a[7:].reshape(3,3),round(a[0]*1e9)
    def control(self,p,v,speed):self.call("nc_control",self.array(p),self.array(v),float(speed))
    def destroy(self):self.lib.nc_destroy(self.p)

def run(library,scene,perception,control,folder,before_motion=None,after_pickup=None,preserve_hold_reference=False,wrist_support=False):
    """Policy methods receive images/encoder FK; truth enters only evaluation."""
    c=yaml.safe_load((control/"config/m3b.yaml").read_text());fc=yaml.safe_load((perception/"config/m3b_filter.yaml").read_text())
    servo=yaml.safe_load((control/"config/servo.yaml").read_text())["moveit_servo"]
    lc=yaml.safe_load((perception/"config/m3b_localization.yaml").read_text());plant=NativePlant(library,scene);track=MotionTrack(fc);track.reset(0,0)
    camera_config=yaml.safe_load((perception/"config/cameras_m3b.yaml").read_text())
    active_views=[view for view,cam in camera_config["cameras"].items() if cam.get("publish_policy","streaming")=="streaming"]
    period=1./camera_config["publish_rate_hz"];delivery_delay=.060;queue=[]
    carried_samples=[];carried_reference=None
    next_capture=0.;observations=[];misses=[];trace=[];errors=[];grasp=None;max_slip=0.;max_rise=0.;max_relative_speed=0.;phase="acquire";closure_trace=[]
    plant.call("nc_settings",plant.array([c["position_gain_per_s"],c["orientation_gain_per_s"],c["maximum_servo_linear_speed_m_s"],c["maximum_servo_angular_speed_rad_s"],c["maximum_joint_speed_rad_s"],servo["lower_singularity_threshold"],servo["hard_stop_singularity_threshold"]]))
    report=dict(status="FAIL",mode="native RGB camera-controlled physics",live_ROS_Servo="NOT_RUN",oracle_controller=False,forced_attachment=False,active_rgb_views=active_views)
    try:
        def capture(force=False):
            nonlocal next_capture
            t=plant.feedback()[0]
            due=[item for item in queue if item[0]<=t+1e-9]
            queue[:]=[item for item in queue if item[0]>t+1e-9]
            for _,view,o,stamp,h in due:
                accepted=track.update(0,stamp,o["position"][:2],o["planar_covariance"],view,o["quality"],round(t*1e9))
                observations.append(dict(phase=phase,capture_ns=stamp,view=view,accepted=accepted,reason=track.reason,**o))
                # Evaluator-only truth was saved at capture time, never fed to the filter.
                if accepted:errors.append(float(np.linalg.norm(track.x[:2]-h)))
            if not force and t+1e-9<next_capture:return
            next_capture=t+period if force else next_capture+period;choices=[]
            for view in active_views:
                rgb,k,p,r,stamp=plant.render(view)
                cv2.imwrite(str(folder/("latest-"+view+".png")),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                try:
                    observation=localize(rgb,k,p,r,lc);choices.append((view,observation,stamp))
                    if phase in {"acquire","closure","close"}:cv2.imwrite(str(folder/(phase+"-"+view+".png")),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                except ValueError as e:misses.append(dict(phase=phase,view=view,time=t,reason=str(e)))
            choices=[choice for choice in choices if choice[0]==fc.get("tracking_view","wrist")]
            if choices:
                view,o,stamp=min(choices,key=lambda v:sum(v[1]["planar_covariance"][::3]))
                truth=plant.truth();q=truth["tray_quat_wxyz"];h=np.asarray(truth["tray_xyz"])+quaternion_rotation([q[1],q[2],q[3],q[0]])@np.array([0,0,.117])
                # The policy receives the captured RGB observation after a measured-budget delay.
                # Truth is carried only for scoring at the identical capture timestamp.
                queue.append((t+delivery_delay,view,o,stamp,h[:2].copy()))
            track.expire(round(t*1e9))
        def predicted():
            t=plant.feedback()[0];dt=t-track.stamp_ns/1e9
            if not track.ready(round(t*1e9)):raise RuntimeError("Native camera track not fresh/ready: "+track.reason)
            return forecast(track.x,track.p,max(dt,0),fc)
        def evaluate():
            nonlocal max_slip,max_rise
            s=plant.truth();trace.append(dict(phase=phase,**s))
            assert s["forbidden_contacts"]==0,"Forbidden physical contact"
            assert max(s["left_force_N"],s["right_force_N"])<15.,"Finger force bound"
            assert plant.feedback()[27]<c["maximum_joint_speed_rad_s"],"Measured native joint speed bound"
            if grasp is not None:
                assert min(s["left_force_N"],s["right_force_N"])>.2,"Physical grip contact lost"
                q=s["hand_quat_wxyz"];r=quaternion_rotation([q[1],q[2],q[3],q[0]])
                relative=r.T@(np.asarray(s["tray_xyz"])-s["hand_xyz"]);slip=float(np.linalg.norm(relative-grasp["relative"]))
                max_slip=max(max_slip,slip);assert slip<.003,"Hand-frame slip bound"
                q1=np.asarray(s["tray_quat_wxyz"]);q0=np.asarray(grasp["state"]["tray_quat_wxyz"])
                assert abs(np.dot(q0,q1))>np.cos(c["maximum_tilt_rad"]/2),"Tray tilt bound"
                rise=s["tray_xyz"][2]-grasp["state"]["tray_xyz"][2];max_rise=max(max_rise,rise)
                if rise>.005:assert s["support_force_N"]<.1 and abs(s["external_force_x_N"])<1e-8,"Lifted traction/support"
            return s
        for _ in range(12):capture(True);plant.call("nc_step",period,0.)
        capture()
        x,_=predicted();pre=[x[0],x[1],c["fixed_handle_z_m"]+.1629]
        phase="pregrasp";plant.call("nc_pregrasp",plant.array(pre))
        track.lose("native pregrasp time gap");queue.clear()
        for _ in range(12):capture(True);plant.call("nc_step",period,0.)
        capture()
        if wrist_support:plant.wrist_tare_world_z_N=float(plant.wrist_wrench()[3])
        if before_motion is not None:before_motion(plant)
        phase="track";start=plant.feedback()[0];window=ClosureProof(c);ready=False;hand_history=[]
        closure=None
        for _ in range(800):
            capture();x,p=predicted();fb=plant.feedback();t=fb[0]-start;u=np.clip(t/c["descend_duration_s"],0,1);blend=u*u*(3-2*u);ds=6*u*(1-u)/c["descend_duration_s"]
            target=np.array([x[0],x[1],c["fixed_handle_z_m"]+.1029+.06*(1-blend)])
            ff=np.array([x[2],x[3],-.06*ds]);plant.control(target,ff,c["belt_speed_m_s"]);evaluate()
            fb=plant.feedback();hand_history.append((fb[0],fb[1:4].copy()));hand_history=hand_history[-9:]
            if len(hand_history)<9:continue
            velocity=(fb[1:4]-hand_history[0][1])/(fb[0]-hand_history[0][0]);x,p=predicted()
            q=fb[4:8];r=quaternion_rotation([q[1],q[2],q[3],q[0]]);desired=np.array([[0,1,0],[1,0,0],[0,0,-1]])
            orientation=np.arccos(np.clip((np.trace(desired.T@r)-1)/2,-1,1))
            closure=dict(time_ns=round(fb[0]*1e9),position_error_m=target-fb[1:4],relative_velocity_m_s=velocity-[x[2],x[3],0],predicted_covariance=p,
                orientation_error_rad=orientation,capture_age_s=fb[0]-track.stamp_ns/1e9,tf_age_s=0.,remaining_s=5.,descent_complete=t>=c["descend_duration_s"],
                tf_stamp_ns=round(fb[0]*1e9),capture_ns=track.stamp_ns)
            good=window.observe(closure);closure_trace.append(dict(time=fb[0],reasons=list(window.window.reasons),position_error_m=closure["position_error_m"].tolist(),relative_speed=float(np.linalg.norm(closure["relative_velocity_m_s"])),orientation=orientation,position_sigma=float(np.sqrt(max(np.linalg.eigvalsh(p[:2,:2])))),velocity_sigma=float(np.sqrt(max(np.linalg.eigvalsh(p[2:,2:]))))))
            if good:ready=True;break
        assert ready,"Native camera closure did not stabilize: "+str(window.window.reasons)
        phase="closure";capture(True);truth=plant.truth();max_relative_speed=float(np.linalg.norm(np.asarray(truth["hand_linear_velocity_m_s"])-truth["tray_linear_velocity_m_s"]))
        assert max_relative_speed<c["close_relative_speed_m_s"],"True native closure relative velocity bound"
        q=truth["tray_quat_wxyz"];handle=np.asarray(truth["tray_xyz"])+quaternion_rotation([q[1],q[2],q[3],q[0]])@np.array([0,0,.117])
        closure_error=np.asarray(truth["hand_xyz"])-(handle+[0,0,.1029])
        assert np.linalg.norm(closure_error[:2])<c["close_position_xy_m"] and abs(closure_error[2])<c["close_position_z_m"],"True native closure position bound"
        assert max(errors)<.002,"Native supported filter error exceeds2mm"
        phase="close";plant.call("nc_close");finger=FingerWindow(c);candidate=False
        for _ in range(250):
            capture();x,_=predicted();plant.control([x[0],x[1],c["fixed_handle_z_m"]+.1029],[x[2],x[3],0.],c["belt_speed_m_s"]);evaluate()
            fb=plant.feedback()
            if finger.observe(round(fb[0]*1e9),fb[15:17],fb[24:26]):candidate=True;break
        assert candidate,"Encoder grip candidate did not stabilize"
        s=plant.truth();assert min(s["left_force_N"],s["right_force_N"])>.2,"Native candidate has no bilateral contact"
        q=s["hand_quat_wxyz"];relative=quaternion_rotation([q[1],q[2],q[3],q[0]]).T@(np.asarray(s["tray_xyz"])-s["hand_xyz"])
        grasp=dict(state=s,relative=relative);fb=plant.feedback();p0=fb[1:4].copy();camera_state=predicted()[0];vx=camera_state[2];begin=fb[0];brake=None
        carried_reference=dict(camera_root_xyz=[float(camera_state[0]),float(camera_state[1]),c["fixed_handle_z_m"]-.117],
            hand_xyz=fb[1:4].tolist(),hand_quat_wxyz=fb[4:8].tolist())
        def save_carried():
            fb=plant.feedback();carried_samples.append(dict(time_sim_s=float(fb[0]),phase=phase,joint_positions=fb[8:17].tolist(),
                hand_xyz=fb[1:4].tolist(),hand_quat_wxyz=fb[4:8].tolist()))
        save_carried()
        phase="lift"
        for index in range(360):
            if index%10==0:save_carried()
            fb=plant.feedback();t=fb[0]-begin;u=np.clip(t/c["lift_duration_s"],0,1);blend=u*u*(3-2*u)
            xx=p0[0]+vx*t;vv=vx
            if fb[3]-p0[2]>.005 and brake is None:brake=(t,xx)
            if brake is not None:
                b=np.clip((t-brake[0])/.5,0,1);xx=brake[1]+vx*.5*(b-b*b+b*b*b/3);vv=vx*(1-b)**2
            plant.control([xx,p0[1],p0[2]+c["lift_height_m"]*blend],[vv,0.,c["lift_height_m"]*6*u*(1-u)/c["lift_duration_s"]],c["belt_speed_m_s"]);evaluate()
        phase="hold";save_carried();fixed=plant.feedback()[1:4]
        for _ in range(60):plant.control(fixed,[0.,0.,0.],0.);evaluate()
        save_carried()
        assert plant.truth()["tray_xyz"][2]-grasp["state"]["tray_xyz"][2]>=.045,"Native held tray rise below45mm"
        phase="final_hold";plant.call("nc_hold_reference" if preserve_hold_reference else "nc_hold")
        for _ in range(60):plant.call("nc_step",.01,0.);evaluate()
        assert plant.truth()["tray_xyz"][2]-grasp["state"]["tray_xyz"][2]>.03,"Native final hold below30mm"
        report.update(status="PASS",true_closure_position_error_m=closure_error.tolist(),maximum_hand_frame_slip_m=max_slip,maximum_rise_m=max_rise,true_closure_relative_speed_m_s=max_relative_speed,
            maximum_supported_filter_error_xy_m=max(errors),maximum_joint_speed_rad_s=plant.feedback()[27],camera_captures=len(observations),configured_camera_rate_hz=camera_config["publish_rate_hz"],simulated_capture_delivery_delay_s=delivery_delay,
            rejected_captures=track.rejected,controller_inputs=["RGB","camera calibration","encoder kinematics"],
            scope="Native bounded kinematics and physics; live ROS planning/Servo/reset remain required")
        if wrist_support:report["controller_inputs"].append("physical wrist force")
        if after_pickup is not None:report["native_sort"]=after_pickup(plant,carried_reference,grasp["state"])
    except Exception as e:report.update(status="FAIL",error=str(e),phase=phase)
    finally:
        plant.destroy()
        for name,value in (("native-camera-pickup.json",report),("native-camera-observations.json",observations),("native-camera-misses.json",misses),("native-camera-physical-trace.json",trace),("native-closure-checks.json",closure_trace),("native-carried-kinematics.json",dict(reference=carried_reference,samples=carried_samples,source="camera root estimate and encoder FK; no tray truth"))):
            (folder/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    if report["status"]!="PASS":raise RuntimeError(report["error"])
    return report
