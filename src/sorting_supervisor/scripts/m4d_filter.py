#!/usr/bin/env python3
"""Keep one track while changing from overhead inspection to wrist grasp tracking."""
import json
import numpy as np
import rclpy
from m4a_common import share
from m4c_common import config
from motion_filter import MotionFilter,ns
from filter_core import forecast
from m4d_core import BRIDGE_PREFIX


class PickupViewFilter(MotionFilter):
    def __init__(self):
        self.requested_wrist=False;self.bridge=None
        self.integration=config()
        super().__init__()

    def reset(self,message):
        previous=self.track.episode
        super().reset(message)
        if previous!=self.track.episode:
            self.requested_wrist=False;self.bridge=None;self.c['tracking_view']='overhead'

    def control_state(self,message):
        super().control_state(message)
        row=json.loads(message.data)
        if row.get('episode_id')==self.track.episode and row.get('active') and row.get('phase')=='moving_pregrasp':
            self.requested_wrist=True

    def receive(self,message):
        if (self.requested_wrist and self.bridge is None and message.view_id=='wrist' and message.valid
                and message.header.frame_id=='world' and message.episode_id==self.track.episode
                and self.track.x is not None and not self.held):
            capture=ns(message.header.stamp);dt=(capture-self.track.stamp_ns)/1e9
            if 0<=dt<self.integration['maximum_capture_age_s']:
                predicted,_=forecast(self.track.x,self.track.p,dt,self.c)
                error=float(np.linalg.norm(predicted[:2]-[message.position.x,message.position.y]))
                if np.isfinite(error) and error<self.integration['maximum_capture_binding_error_m']:
                    self.bridge=dict(episode_id=self.track.episode,track_id=self.track.track_id,
                        from_view='overhead',to_view='wrist',overhead_capture_ns=self.track.stamp_ns,
                        capture_ns=capture,binding_error_m=error)
                    self.c['tracking_view']='wrist';self.pending.clear()
        super().receive(message)

    def decorate_belief(self,belief):
        if belief.view_id=='wrist' and self.bridge is not None:
            belief.reason=BRIDGE_PREFIX+json.dumps(self.bridge,separators=(',',':'))


def main():
    rclpy.init();node=PickupViewFilter()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
