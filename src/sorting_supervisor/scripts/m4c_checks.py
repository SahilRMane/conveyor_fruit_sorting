"""Offline RGB-domain and constant-speed filter checks; labels stay in this evaluator."""
import cv2
import numpy as np
import yaml
from m4b_common import classifier_config,model_directory
from m4a_common import share
from classifier_core import Classifier,fruit_crop,require
from camera_core import localize
from filter_core import MotionTrack
from m4c_core import remaining,required_time
from m4c_timing import timing_budget


def replay_capture_delivery(captures, limit_s, delivery_s, callback_s):
    """Check capture age before queued callbacks, with timestamps left unchanged."""
    capture_ns=sorted(captures);delay=round((delivery_s+callback_s)*1e9)
    arrivals=[(t+delay,t) for t in capture_ns];last=None;index=0;maximum=0.;failures=0
    for now in range(arrivals[0][0],arrivals[-1][0]+2_000_001,2_000_000):
        # A clock callback can run before a ready belief callback.
        if last is not None:
            age=(now-last)/1e9;maximum=max(maximum,age)
            if age>=limit_s:failures+=1
        while index<len(arrivals) and arrivals[index][0]<=now:
            last=arrivals[index][1];index+=1
    return dict(maximum_capture_age_sim_s=maximum,stale_checks=failures,
                delivery_budget_s=delivery_s,callback_budget_s=callback_s,
                scope='Budgeted callback-order replay; not a live latency measurement')


def localization(row, raw, c):
    rgb=cv2.cvtColor(cv2.imread(str(raw/row['image'])),cv2.COLOR_BGR2RGB)
    f=row['height']/2/np.tan(np.deg2rad(row['fovy_deg'])/2)
    k=np.array([[f,0,row['width']/2],[0,f,row['height']/2],[0,0,1.]])
    rotation=np.asarray(row['camera_optical_rotation']).reshape(3,3)
    loc=localize(rgb,k,row['camera_xyz'],rotation,c)
    error=float(np.linalg.norm(np.asarray(loc['position'])[:2]-row['evaluator_handle_xyz'][:2]))
    return rgb,k,rotation,loc,error


def check_native(raw, c, filter_config, costs=None):
    report=yaml.safe_load((raw/'probe.yaml').read_text())
    require(abs(report['belt_speed_m_s']-c['belt_speed_m_s'])<1e-12,'Native probe used another conveyor speed')
    lc=yaml.safe_load((share('sorting_perception')/'config/m3b_localization.yaml').read_text())
    cc=classifier_config();model=Classifier(model_directory(),cc);domain=[]
    for row in report['domain_frames']:
        lo,hi=c['classification_x_ranges'][row['pose']]
        if not lo-1e-9<=row['x_m']<=hi+1e-9:continue
        rgb,k,r,loc,error=localization(row,raw,lc)
        require(error<c['maximum_localization_error_m'],'Moving wrist domain localization exceeds 2 mm')
        crop,_=fruit_crop(rgb,k,row['camera_xyz'],r,loc['position'],cc);score=model.infer(crop)
        visible=row['condition']=='clear' or (row['condition']=='partial' and row['pose']=='inspect_left')
        require(score['accepted']==visible,'Frozen classifier confidence failed in moving domain')
        if visible:require(int(np.argmax(score['probabilities']))==row['label'],'Frozen classifier accepted a wrong class')
        domain.append(dict(pose=row['pose'],x_m=row['x_m'],condition=row['condition'],label=row['label'],
                           confidence=score['confidence'],accepted=score['accepted'],localization_error_m=error))
    require(len(domain)>=36,'Too few moving-domain classifier probes')
    track=MotionTrack(filter_config);track.reset(0,0);grades=[];failures=[]
    for row in report['moving_frames']:
        at=int(row['capture_ns'])
        try:
            _,_,_,loc,error=localization(row,raw,lc)
            delivered=at+round(c['delivery_budget_s']*1e9)
            accepted=track.update(0,at,loc['position'][:2],loc['planar_covariance'],'overhead',loc['quality'],delivered)
            require(accepted,'Filter rejected native overhead capture: '+track.reason)
            require(error<c['maximum_localization_error_m'],'Native overhead localization exceeds 2 mm')
            if track.ready(delivered):
                filter_error=float(np.linalg.norm(track.x[:2]-row['evaluator_handle_xyz'][:2]))
                velocity_error=float(abs(track.x[2]-row['vx_m_s']))
                require(filter_error<c['maximum_localization_error_m'],'Native filtered location exceeds 2 mm')
                require(velocity_error<c['maximum_velocity_error_m_s'],'Native velocity error exceeds bound')
                grades.append(dict(capture_ns=at,localization_error_m=filter_error,velocity_error_m_s=velocity_error,
                                   velocity_sigma_m_s=float(np.sqrt(track.p[2,2])),pose=row['pose_transition']))
        except ValueError as error:
            failures.append(dict(capture_ns=at,reason=str(error)));track.expire(at)
    require(len(grades)>=300,'Too few native moving overhead grades')
    gaps=np.diff([g['capture_ns'] for g in grades])/1e9
    require(max(gaps)<c['maximum_tracking_gap_s'],'Native overhead continuity gap exceeds bound')
    require(report['pair_render_p95_wall_s']<c['maximum_pair_render_p95_wall_s'],'Combined RGB render exceeds the two-camera budget')
    replay=replay_capture_delivery([g['capture_ns'] for g in grades],c['maximum_capture_age_s'],
                                   c['delivery_budget_s'],c['callback_budget_s'])
    require(replay['stale_checks']==0,'Budgeted camera delivery crosses the active age limit')
    budget=timing_budget(c)
    captures=[g['capture_ns'] for g in grades]
    # Drop isolated updates, preserving capture stamps. Add the complete jitter allocation.
    retained=[t for i,t in enumerate(captures) if i%31!=15]
    missing=replay_capture_delivery(retained,c['maximum_capture_age_s'],
        c['delivery_budget_s']+c['capture_gap_jitter_s'],c['callback_budget_s'])
    require(missing['stale_checks']==0,'One missing capture crosses the active age limit')
    require(max(np.diff(retained))/1e9<=budget['maximum_capture_gap_s'],
            'One missing native capture exceeds the gap allocation')
    deadline_checks={}
    if costs is not None:
        speed=c['belt_speed_m_s'];age=.10
        x=c['assess_trigger_x_m']+(costs['initial_assess']+c['capture_budget_s'])*speed
        b=dict(valid=True,camera_only=True,view_id='overhead',frame_id='world',capture_ns=1_000_000_000,
               state=[x-speed*age,0.,speed,0.],covariance=track.p.reshape(-1).tolist())
        horizon=remaining(b,1_100_000_000,c,filter_config)
        needed=required_time('INSPECT','assess',costs,c)
        require(horizon['remaining_s']>needed,'Measured costs and overhead uncertainty leave no inspection slack')
        b['state'][0]=c['late_x_m']+2*speed-speed*age
        late=remaining(b,1_100_000_000,c,filter_config)
        require(late['remaining_s']<=required_time('INITIAL','assess',costs,c),'Late-arrival deadline probe did not reject inspection')
        deadline_checks=dict(nominal_inspection_slack_s=horizon['remaining_s']-needed,
                             worst_initial_capture_x_m=x,late_remaining_s=late['remaining_s'],capture_age_s=age)
    return dict(status='PASS',scope='Development-domain RGB and constant-speed dynamics; no live ROS',
                belt_speed_m_s=c['belt_speed_m_s'],
                model_sha256=model.sha,classifier_domain_samples=len(domain),classifier_domain=domain,
                moving_samples=len(grades),rejected_frames=failures,track_id=track.track_id,
                maximum_overhead_error_m=max(g['localization_error_m'] for g in grades),
                maximum_velocity_error_m_s=max(g['velocity_error_m_s'] for g in grades),
                maximum_tracking_gap_sim_s=float(max(gaps)),
                velocity_sigma_range_m_s=[min(g['velocity_sigma_m_s'] for g in grades),max(g['velocity_sigma_m_s'] for g in grades)],
                pair_render_p95_wall_s=report['pair_render_p95_wall_s'],benchmark=report['benchmark'],
                deadline_checks=deadline_checks,capture_delivery_replay=replay,
                missing_capture_replay=missing,timing_budget=budget)
