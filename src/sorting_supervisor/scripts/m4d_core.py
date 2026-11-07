"""An immutable class route carried only on one continuously observed track."""
from copy import deepcopy
import math
import json
from types import MappingProxyType


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


BRIDGE_PREFIX='M4D_VIEW_BRIDGE:'


def associated_snapshot(belief):
    from m4c_runtime import snapshot
    row=snapshot(belief)
    if belief.view_id=='wrist':
        require(belief.reason.startswith(BRIDGE_PREFIX),'Wrist target has no capture-time association proof')
        row['view_bridge']=json.loads(belief.reason[len(BRIDGE_PREFIX):])
    return row


class RouteLease:
    def __init__(self, decision, model_sha, maximum_age_s):
        require(decision['action'] == 'PICK_READY', 'Only PICK_READY can create a class route')
        require(math.isfinite(maximum_age_s) and 0 < maximum_age_s <= 30, 'Invalid class route lifetime')
        value = deepcopy(decision)
        value['probabilities'] = tuple(value['probabilities'])
        self.route = MappingProxyType(value)
        self.model_sha = model_sha
        self.begin = decision['decision_time_ns']
        self.last_capture = None
        self.expires = self.begin+round(maximum_age_s*1e9)
        self.invalid = None
        self.view = 'overhead'

    def check(self, target, now_ns, config):
        require(self.invalid is None, self.invalid)
        try:
            require(self.begin <= now_ns < self.expires, 'Class route expired or clock moved backwards')
            require(target['episode_id'] == self.route['episode_id']
                    and target['track_id'] == self.route['track_id'], 'Class route identity changed')
            require(target['valid'] and target['camera_only'] and target['frame_id'] == 'world',
                    'Class route lost its valid camera track')
            if target['view_id'] == 'wrist':
                bridge=target.get('view_bridge',{})
                require(bridge.get('episode_id')==self.route['episode_id']
                        and bridge.get('track_id')==self.route['track_id']
                        and bridge.get('from_view')=='overhead' and bridge.get('to_view')=='wrist',
                        'Camera view association identity differs')
                error=bridge.get('binding_error_m',float('inf'))
                require(math.isfinite(error) and 0<=error<config['maximum_capture_binding_error_m'],
                        'Camera view association exceeds its spatial bound')
                require(self.begin<=bridge['capture_ns']<=target['capture_ns']
                        and 0<=bridge['capture_ns']-bridge['overhead_capture_ns']<round(config['maximum_capture_age_s']*1e9),
                        'Camera view association capture is invalid')
                self.view='wrist'
            require(target['view_id']==self.view,'Camera view changed without a valid association')
            require(0 <= now_ns-target['capture_ns'] < round(config['maximum_capture_age_s']*1e9),
                    'Class route has stale or future target feedback')
            if self.last_capture is not None:
                require(target['capture_ns'] >= self.last_capture, 'Class route capture moved backwards')
                require(target['capture_ns']-self.last_capture < round(config['maximum_tracking_gap_s']*1e9),
                        'Class route tracking gap exceeded its bound')
            self.last_capture = target['capture_ns']
            return dict(action='PICK_PLACE', class_id=self.route['class_id'], bin_id=self.route['proposed_bin'],
                        confidence=self.route['confidence'], probabilities=list(self.route['probabilities']),
                        episode_id=self.route['episode_id'], track_id=self.route['track_id'],
                        capture_ns=self.route['evidence_capture_ns'], model_sha256=self.model_sha,
                        reason='inspected_class_on_continuous_camera_track',
                        class_age_sim_s=(now_ns-self.route['evidence_capture_ns'])/1e9,
                        tracking_view=self.view,camera_view_association=deepcopy(target.get('view_bridge')),
                        class_carried_on_continuous_track=True, route_expires_ns=self.expires)
        except (ValueError, KeyError) as error:
            self.invalid = str(error)
            raise
