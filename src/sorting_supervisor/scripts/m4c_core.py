"""Camera-only track binding and deadline proposals. No ROS or simulator input."""
import math
import numpy as np
import m4a_common  # Resolve the existing installed perception libraries.
from filter_core import forecast
from inspection_core import time_to_exit
from m4b_core import SingleViewPolicy
from classifier_core import require


class DeadlinePass(ValueError):
    """Decline this target and cancel any active arm command."""


def remaining(snapshot, now_ns, c, filter_config, *, tracking_view="overhead"):
    require(snapshot['valid'] and snapshot['camera_only'] and snapshot['view_id']==tracking_view
            and snapshot['frame_id']=='world', 'Invalid overhead belief or frame')
    age = (now_ns-snapshot['capture_ns'])/1e9
    require(0 <= age < c['maximum_capture_age_s'],
            f"Overhead capture is stale or future: age_s={age:.6f}, limit_s={c['maximum_capture_age_s']:.6f}")
    state=np.asarray(snapshot['state'],dtype=float);cov=np.asarray(snapshot['covariance'],dtype=float).reshape(4,4)
    require(state.shape==(4,) and np.isfinite(state).all() and np.isfinite(cov).all()
            and np.allclose(cov,cov.T) and np.linalg.eigvalsh(cov).min()>=-1e-12, 'Invalid overhead state or covariance')
    x,p=forecast(state,cov,age,filter_config)
    sigma_x, sigma_v = math.sqrt(max(0.,p[0,0])), math.sqrt(max(0.,p[2,2]))
    require(x[2] >= -c['uncertainty_sigma']*sigma_v, 'Reverse motion is outside M4C')
    horizon=time_to_exit(float(x[0]),max(0.,float(x[2])),sigma_x,sigma_v,c)
    return dict(remaining_s=horizon, predicted_x_m=float(x[0]), velocity_m_s=float(x[2]),
                position_sigma_m=sigma_x,velocity_sigma_m_s=sigma_v,capture_ns=snapshot['capture_ns'])


def pick_reserve(c):
    return c['approach_allocation_s']+c['close_s']+c['margin_s']


def required_time(action, pose, costs, c):
    reserve=pick_reserve(c)
    if action=='INITIAL': return costs['initial_assess']+c['capture_budget_s']+reserve
    if action=='INSPECT': return costs['inspect_left']+c['capture_budget_s']+costs['return_assess']+reserve
    if action=='PICK_READY': return reserve+(costs['return_assess'] if pose=='inspect_left' else 0.)
    return 0.


def bind_capture(sample, history, episode, track_id, c, filter_config):
    require(sample['episode_id']==episode, 'Wrist evidence belongs to another episode')
    eligible=[b for b in history if b['episode_id']==episode and b['track_id']==track_id
              and b['valid'] and b['camera_only'] and b['view_id']=='overhead' and b['frame_id']=='world'
              and b['capture_ns']<=sample['capture_ns']]
    require(bool(eligible), 'No earlier overhead capture binds this wrist image')
    b=max(eligible,key=lambda value:value['capture_ns']);dt=(sample['capture_ns']-b['capture_ns'])/1e9
    require(dt<c['maximum_capture_age_s'], 'Capture-time overhead pair is too old')
    x,p=forecast(b['state'],np.asarray(b['covariance']).reshape(4,4),dt,filter_config)
    error=float(np.linalg.norm(x[:2]-sample['handle_xyz'][:2]))
    require(error<c['maximum_capture_binding_error_m'], 'Wrist image does not match the overhead track')
    return dict(track_id=track_id,overhead_capture_ns=b['capture_ns'],wrist_capture_ns=sample['capture_ns'],
                binding_error_m=error,predicted_position_sigma_m=float(np.sqrt(max(p[0,0],p[1,1]))))


class MovingPolicy(SingleViewPolicy):
    def __init__(self, episode, track_id, begin_ns, config, costs):
        super().__init__(episode,begin_ns,config)
        self.track_id,self.costs=track_id,costs

    def observe(self, *, track_id, horizon, **kwargs):
        require(track_id==self.track_id, 'Overhead track identity changed')
        require(math.isfinite(horizon) and horizon>=0, 'Invalid exit horizon')
        result=super().observe(**kwargs)
        needed=required_time(result['action'],kwargs['pose'],self.costs,self.config)
        if result['action'] in ('INSPECT','PICK_READY') and horizon<=needed:
            result.update(action='PASS',reason='insufficient_conservative_slack')
            for key in ('selected_view','class_id','proposed_bin'): result.pop(key,None)
        result.update(track_id=track_id,remaining_s=horizon,estimated_service_time_s=needed,
                      slack_s=horizon-needed,physical_pickup_timing_validated=False)
        self.state=result['action']
        self.history[-1].update(result,to_state=self.state)
        return result
