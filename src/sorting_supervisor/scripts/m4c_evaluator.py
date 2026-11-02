"""Evaluator-only camera truth and physical transport checks. Never controls a decision."""
import json
import time
import numpy as np
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from sorting_interfaces.msg import CameraTruth,TargetBeliefArray
from test_m4a import Evaluator
from m4a_runtime import ns
from classifier_core import require
from m4c_runtime import snapshot


class MovingEvaluator(Evaluator):
    def __init__(self, config):
        Node.__init__(self,'m4c_evaluator',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.c=config;self.truth={};self.overhead_truth={};self.contact=None;self.failure=None
        self.window_start_ns=None;self.window_end_ns=None;self.pending={};self.grades=[];self.transport=[]
        self.previous_truth=None;self.last_capture=0
        self.latest_transport_ns=None;self.transport_failure=None
        self.create_subscription(CameraTruth,'/sorting/ground_truth/camera_sample',self.receive_truth,200)
        self.create_subscription(String,'/sorting/contact_state',self.receive_contact,1)
        self.create_subscription(String,'/sorting/ground_truth/conveyor_state',self.receive_transport,1)
        self.create_subscription(TargetBeliefArray,'/sorting/m4c/targets',self.receive_targets,10)
        self.create_timer(.02,self.grade_pending)

    def receive_truth(self, m):
        cache=self.truth if m.view_id=='wrist' else self.overhead_truth
        cache[(m.episode_id,ns(m.header.stamp))]=m
        while len(cache)>10000:cache.pop(next(iter(cache)))

    def receive_contact(self, m):
        self.contact=json.loads(m.data);c=self.contact
        if c['forbidden_contacts'] or c['left_force_N']+c['right_force_N']>=.1:
            self.failure='Moving inspection made physical contact'
        if min(c['finger_positions_m'])<.039 or c['support_force_N']<.5:
            self.failure='Moving inspection lost open fingers or tray support'

    def receive_transport(self, m):
        s=json.loads(m.data);at=round(s['sim_time']*1e9)
        self.latest_transport_ns=max(at,self.latest_transport_ns if self.latest_transport_ns is not None else at)
        if self.window_start_ns is None or at<self.window_start_ns+1_000_000_000:return
        if self.window_end_ns is not None and at>=self.window_end_ns:return
        reason=None
        if not s['command_fresh'] or abs(s['requested_speed_m_s']-self.c['belt_speed_m_s'])>1e-9:
            reason='Conveyor command stopped during the moving inspection'
        if abs(s['force_x_N'])>8.000001 or abs(s['tray_vx_m_s']-self.c['belt_speed_m_s'])>.0015:
            reason=reason or 'Physical conveyor force or speed exceeded its bound'
        if reason:
            self.failure=self.failure or reason
            if self.transport_failure is None:
                self.transport_failure=dict(reason=reason,sample=s,sample_ns=at,
                    received_clock_ns=self.get_clock().now().nanoseconds,
                    window_start_ns=self.window_start_ns,window_end_ns=self.window_end_ns)
        if not self.transport or s['sim_time']-self.transport[-1]['sim_time']>=.05:self.transport.append(s)

    def receive_targets(self, message):
        if self.window_start_ns is None:return
        for b in message.targets:
            capture=ns(b.header.stamp)
            if (b.valid and b.view_id=='overhead' and capture>=self.window_start_ns and capture>self.last_capture
                    and (self.window_end_ns is None or capture<self.window_end_ns)):
                self.pending[capture]=(snapshot(b),time.monotonic());self.last_capture=capture

    def grade_pending(self):
        for capture,(b,wall) in list(self.pending.items()):
            truth=next((self.overhead_truth[(b['episode_id'],capture+delta)] for delta in (0,-1,1,-2,2)
                        if (b['episode_id'],capture+delta) in self.overhead_truth),None)
            if truth is None:
                if time.monotonic()-wall>2.:
                    self.failure='Evaluator overhead capture-time pair timed out';self.pending.pop(capture)
                continue
            p=truth.handle_position;xy=np.array([p.x,p.y]);error=float(np.linalg.norm(np.asarray(b['state'])[:2]-xy))
            velocity_error=None
            if self.previous_truth is not None:
                previous_at,previous_xy=self.previous_truth;dt=(capture-previous_at)/1e9
                if dt>0:velocity_error=float(np.linalg.norm(np.asarray(b['state'])[2:]-(xy-previous_xy)/dt))
            self.previous_truth=(capture,xy)
            age=(self.get_clock().now().nanoseconds-capture)/1e9
            self.grades.append(dict(capture_ns=capture,track_id=b['track_id'],localization_error_m=error,
                                    velocity_error_m_s=velocity_error,capture_age_sim_s=age))
            self.pending.pop(capture)
            if error>=self.c['maximum_localization_error_m']:self.failure='Overhead localization exceeds 2 mm'
            if velocity_error is not None and velocity_error>=self.c['maximum_velocity_error_m_s']:
                self.failure='Overhead velocity estimate exceeded its bound'

    def finish(self, events):
        require(self.failure is None,self.failure)
        require(len(self.grades)>=10 and len(self.transport)>=5,'Too few moving tracking or transport samples')
        require(len({g['track_id'] for g in self.grades})==1,'Evaluator detected track reassignment')
        captures=[g['capture_ns'] for g in self.grades];gaps=np.diff(captures)/1e9
        require(not len(gaps) or max(gaps)<self.c['maximum_tracking_gap_s'],'Overhead tracking gap exceeded 0.3 s')
        transitions=[];begin=None
        for event in events:
            if event['state']=='PLAN':begin=event
            if event['state']=='SETTLED' and begin is not None:
                first,last=begin['enter_ns'],event['enter_ns'];inside=[t for t in captures if first<=t<=last]
                require(bool(inside),'No overhead feedback during wrist motion')
                gap=max(np.diff([first]+inside+[last]))/1e9
                require(gap<self.c['maximum_tracking_gap_s'],'Overhead continuity failed during wrist motion')
                transitions.append(dict(pose=event['pose'],samples=len(inside),maximum_gap_sim_s=float(gap)))
                begin=None
        return dict(status='PASS',overhead_samples=len(self.grades),transport_samples=len(self.transport),
                    maximum_tracking_gap_sim_s=float(max(gaps)) if len(gaps) else 0.,wrist_transitions=transitions,
                    maximum_overhead_error_m=max(g['localization_error_m'] for g in self.grades),
                    maximum_velocity_error_m_s=max(g['velocity_error_m_s'] or 0. for g in self.grades),
                    minimum_physical_speed_m_s=min(s['tray_vx_m_s'] for s in self.transport))
