"""Offline RGB + encoder controlled sort. Simulator truth is only a grader."""
import json
from pathlib import Path
import cv2,numpy as np,yaml
from native_camera_pickup import run,NativePlant
from camera_core import localize,quaternion_rotation
from classifier_core import Classifier,fruit_crop
from sort_core import choose_route,footprint_inside_estimate,resting_pose_from_release,supported_floor_pose
from wrist_support import SupportWindow,coarse_lower_transition
from lift_support_fixtures import hand_matrix

def check_floor_force(sample,peaks,index):
    """Grade every physics-step peak as well as the current force sample."""
    current=np.asarray(sample['bin_floor_force_N']);peaks=np.asarray(peaks)
    assert current.shape==(2,) and peaks.shape==(2,) and np.isfinite(current).all() and np.isfinite(peaks).all() and np.all(current>=0) and np.all(peaks>=0),'Invalid native bin force record'
    assert max(current[1-index],peaks[1-index])<.1,'Unselected bin touched'
    assert max(current[index],peaks[index])<2*.52*9.81,'Bin floor impact exceeds bound'

def classify(plant,perception,model,folder):
    lc=yaml.safe_load((perception/'config/m3b_localization.yaml').read_text())
    rgb,k,p,r,capture=plant.render('wrist');observation=localize(rgb,k,p,r,lc)
    crop,bbox=fruit_crop(rgb,k,p,r,observation['position'],model.c);value=model.infer(crop)
    cv2.imwrite(str(folder/'class-crop.png'),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
    return dict(value,capture_ns=capture,crop_xyxy=bbox,model_sha256=model.sha)

def physical_sort(plant,reference,grasp,c,route,folder):
    root=np.eye(4);root[:3,3]=reference['camera_root_xyz'];relative=np.linalg.inv(hand_matrix(reference))@root
    traces=[];samples=[];max_slip=0.;released=False;phase='transfer_retract'
    q=grasp['hand_quat_wxyz'];r=quaternion_rotation([q[1],q[2],q[3],q[0]])
    initial_relative=r.T@(np.asarray(grasp['tray_xyz'])-grasp['hand_xyz'])
    def body():
        fb=plant.feedback();return hand_matrix(dict(hand_xyz=fb[1:4],hand_quat_wxyz=fb[4:8]))@relative
    def score():
        nonlocal max_slip
        fb=plant.feedback();s=plant.truth()
        peaks=plant.floor_peaks()
        s['bin_floor_step_peak_N']=peaks.tolist();traces.append(dict(phase=phase,**s))
        if len(traces)%5==0:samples.append(dict(time_sim_s=float(fb[0]),phase=phase,joint_positions=fb[8:17].tolist(),hand_xyz=fb[1:4].tolist(),hand_quat_wxyz=fb[4:8].tolist()))
        assert not s['forbidden_contacts'] and sum(s['bin_wall_force_N'])<.1,'Forbidden contact during native sort'
        assert max(s['left_force_N'],s['right_force_N'])<15.,'Finger force exceeds bound'
        assert abs(s['external_force_x_N'])<1e-8,'Conveyor traction during transfer'
        assert abs(np.dot(s['tray_quat_wxyz'],grasp['tray_quat_wxyz']))>np.cos(c['maximum_tilt_rad']/2),'Tilt exceeds bound'
        assert -.003<s['tray_xyz'][2]-grasp['tray_xyz'][2]<c['maximum_transfer_rise_m'],'Rise exceeds transfer bound'
        index=c['bins'][route['bin_id']]['index'];check_floor_force(s,peaks,index)
        if not released:
            q=s['hand_quat_wxyz'];r=quaternion_rotation([q[1],q[2],q[3],q[0]])
            slip=float(np.linalg.norm(r.T@(np.asarray(s['tray_xyz'])-s['hand_xyz'])-initial_relative));max_slip=max(max_slip,slip)
            assert slip<.003 and min(s['left_force_N'],s['right_force_N'])>.2,'Grip/slip condition failed'
        if phase=='transfer':assert s['tray_xyz'][2]-grasp['tray_xyz'][2]>.085 and s['support_force_N']+sum(s['bin_floor_force_N'])<.1,'Airborne transfer not clear'
        return s
    def move(goal,duration,contact=None):
        p0=plant.feedback()[1:4].copy();delta=np.asarray(goal)-p0;begin=plant.feedback()[0]
        for _ in range(int((duration+6.)/.01)):
            t=plant.feedback()[0]-begin;u=np.clip(t/duration,0,1);blend=u*u*(3-2*u);ds=6*u*(1-u)/duration
            plant.control(p0+delta*blend,delta*ds,0.);score()
            if contact is not None and contact(goal,t):return
            if t>duration+.3 and np.linalg.norm(plant.feedback()[1:4]-goal)<.0015:return
        raise RuntimeError('Native endpoint did not settle: '+phase)
    try:
        plant.call('nc_resume');p=plant.feedback()[1:4];world=body();move(p+[c['retract_root_x_m']-world[0,3],-world[1,3],0],6.)
        phase='transfer_raise';p=plant.feedback()[1:4];world=body();move(p+[0,0,c['transfer_root_z_m']-world[2,3]],3.)
        phase='transfer';destination=np.asarray(c['bins'][route['bin_id']]['center']);p=plant.feedback()[1:4];world=body()
        hover=p+[destination[0]-world[0,3],destination[1]-world[1,3],c['transfer_root_z_m']-world[2,3]]
        move(hover,max(6.,float(np.linalg.norm(hover-p))/.03))
        assert footprint_inside_estimate(body(),route['bin_id'],c),'Estimated footprint outside destination'
        release_destination=destination.copy()
        if c.get('use_wrist_support',True):
            settings=c['supported_release'];tare=plant.wrist_tare_world_z_N
            loaded=plant.wrist_wrench()[3]-tare;assert .8*settings['payload_weight_N']<abs(loaded)<1.2*settings['payload_weight_N'],'Wrist payload calibration failed'
            sign=float(np.sign(loaded));loaded=abs(loaded)
            release_destination[2]=settings['approach_start_root_z_m']
            def contact(goal,elapsed):
                return coarse_lower_transition(plant.feedback()[1:4],goal,elapsed,c['lower_duration_s'],
                    sign*(plant.wrist_wrench()[3]-tare),loaded,c['placement_position_tolerance_m'])['ready']
        else:release_destination[2]=c['release_root_z_m'];contact=None
        phase='lower';p=plant.feedback()[1:4];move(p+release_destination-body()[:3,3],c['lower_duration_s'],contact)
        if c.get('use_wrist_support',True):
            start=plant.feedback()[1:4].copy();begin=plant.feedback()[0];proof=SupportWindow(settings)
            def supported_control():
                current=plant.feedback()[1:4].copy();force=sign*(plant.wrist_wrench()[3]-tare)
                speed=-float(np.clip(settings['force_velocity_gain_m_per_N_s']*force,-settings['approach_speed_m_s'],settings['approach_speed_m_s']))
                plant.control([start[0],start[1],current[2]],[0.,0.,speed],0.)
            for _ in range(round(settings['maximum_approach_sim_s']/.01)):
                distance=abs(plant.feedback()[3]-start[2])
                assert distance<settings['maximum_approach_distance_m'],'Wrist support approach exceeded bound'
                supported_control();score();force=sign*(plant.wrist_wrench()[3]-tare)
                traces[-1]['wrist_payload_N']=float(force)
                if proof.observe(round(plant.feedback()[0]*1e9),force,loaded):break
            else:raise RuntimeError('Wrist support approach timed out')
            fixed=plant.feedback()[1:4].copy()
            for _ in range(10):supported_control();score()
            placed=supported_floor_pose(body(),route['bin_id'],c)
            # A measured support event supplies the known floor-height prior.
            relative=np.linalg.inv(hand_matrix(dict(hand_xyz=plant.feedback()[1:4],hand_quat_wxyz=plant.feedback()[4:8])))@placed
            phase='release';released=True
            for position in settings['supported_unload_positions_m']:
                plant.call('nc_gripper',float(position))
                for _ in range(round((c['release_unload_settle_s']+c.get('native_release_action_wait_s',1.75))/.01)):
                    supported_control();score()
            assert sign*(plant.wrist_wrench()[3]-tare)<settings['maximum_open_payload_fraction']*loaded,'Wrist payload did not unload before opening'
        else:
            placed=resting_pose_from_release(body(),route['bin_id'],c);assert np.linalg.norm(body()[:3,3]-release_destination)<.002,'Estimated placement centre error'
            phase='release';released=True;fixed=plant.feedback()[1:4].copy()
            for position in c['release_unload_positions_m']:
                plant.call('nc_gripper',float(position))
                dwell=c['release_unload_settle_s']+c.get('native_release_action_wait_s',1.75)
                for _ in range(round(dwell/.01)):plant.control(fixed,[0,0,0],0.);score()
        for gap in c['release_finger_positions_m']:
            plant.call('nc_gripper',float(gap))
            dwell=.15+(c.get('native_release_action_wait_s',0.) if gap==c['release_finger_positions_m'][0] else 0.)
            for _ in range(round(dwell/.01)):
                if c.get('use_wrist_support',True):supported_control()
                else:plant.control(fixed,[0,0,0],0.)
                score()
        for _ in range(30):
            if c.get('use_wrist_support',True):supported_control()
            else:plant.control(fixed,[0,0,0],0.)
            score()
        assert min(plant.feedback()[15:17])>.039,'Native gripper not open'
        phase='retreat';move(plant.feedback()[1:4]+[0,0,.1],3.)
        phase='settled';plant.call('nc_hold')
        for _ in range(200):plant.call('nc_step',.01,0.);score()
        settled=[s for s in traces if s['phase']=='settled'];index=c['bins'][route['bin_id']]['index']
        for s in settled[-150:]:
            assert np.linalg.norm(np.asarray(s['tray_xyz'])-destination)<.003,'Physical placement centre error'
            assert abs(s['bin_floor_force_N'][index]-.52*9.81)<.5,'Incorrect bin support'
            assert s['left_force_N']+s['right_force_N']<.1,'Fingers still touch tray'
            assert max(abs(v) for v in s['tray_linear_velocity_m_s'])<.002 and max(abs(v) for v in s['tray_angular_velocity_rad_s'])<.02,'Placed tray not settled'
        report=dict(status='PASS',action='PICK_PLACE',bin_id=route['bin_id'],maximum_hand_frame_slip_m=max_slip,final_state=settled[-1],placement_samples=len(settled),forced_attachment=False)
        return report
    finally:
        (folder/'sort-physical-trace.json').write_text(json.dumps(traces,indent=2)+'\n')
        (folder/'sort-kinematics.json').write_text(json.dumps(dict(reference=reference,relative=relative.tolist(),placed_matrix=body().tolist() if not released else placed.tolist(),bin_id=route['bin_id'],samples=samples),indent=2)+'\n')

def run_fixture(library,scene,perception,control,folder,expected_class,covered=False):
    folder.mkdir(parents=True,exist_ok=True);c=yaml.safe_load((control/'config/m3d.yaml').read_text());mc=yaml.safe_load((perception/'config/m3c.yaml').read_text());model=Classifier(perception/'models/m3c_baseline',mc);decision={}
    def before(plant):
        nonlocal decision
        value=classify(plant,perception,model,folder);decision=choose_route(mc['class_ids'],value['probabilities'],c)
        assert decision['action']=='PICK_PLACE' and decision['class_id']==expected_class,'Native class route incorrect'
        (folder/'class.json').write_text(json.dumps(dict(decision=decision,evidence=value),indent=2)+'\n')
    if covered:
        plant=NativePlant(library,scene)
        try:
            value=classify(plant,perception,model,folder);decision=choose_route(mc['class_ids'],value['probabilities'],c)
            assert decision['action']=='PASS','Covered scene was accepted'
            # No arm or gripper command is issued on this branch.
            initial=plant.truth();plant.call('nc_step',c['pass_transport_s'],.03);plant.call('nc_step',.3,0.);final=plant.truth()
            assert np.linalg.norm(np.asarray(final['tray_xyz'])[1:]-np.asarray(initial['tray_xyz'])[1:])<.003 and min(final['finger_positions_m'])>.039 and .02<final['tray_xyz'][0]-initial['tray_xyz'][0]<.08,'PASS did not transport supported tray with open fingers'
            return dict(status='PASS',action='PASS',evidence=value,physical_grasp='NOT_RUN',conveyor_displacement_m=final['tray_xyz'][0]-initial['tray_xyz'][0])
        finally:plant.destroy()
    report=run(library,scene,perception,control,folder,preserve_hold_reference=True,wrist_support=True,before_motion=before,after_pickup=lambda plant,reference,grasp:physical_sort(plant,reference,grasp,c,decision,folder))
    report['class_decision']=decision;return report
