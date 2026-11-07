"""Inspection and physical pickup share one belt publisher and one class route."""
from m4c_runtime import MovingInspectionExecutor, FeedbackExecutor
from m4c_core import MovingPolicy, bind_capture, DeadlinePass
from m4d_core import RouteLease
from m4d_handoff import check_handoff
from filter_core import forecast
from m4c_core import remaining
from classifier_core import require
from m4c_timing import CaptureTiming
from m4a_runtime import ns
from m4d_core import associated_snapshot
import time
import cv2


class IntegratedExecutor(FeedbackExecutor):
    def __init__(self, callback_budget_s):
        super().__init__(callback_budget_s)
        self.inspector=None
        self.physical=None

    def spin_once(self, timeout_sec=None):
        super().spin_once(timeout_sec=timeout_sec)
        control=self.inspector
        if self.physical is not None and (self.physical.grip_candidate or not self.physical.target_required):
            control.capture_monitoring=False
        if control is not None and control.timing is not None and control.capture_monitoring:
            control.timing.sample(control.now(),enforce=False,
                context=dict(phase='physical_pickup' if control.handed_off else 'inspection',
                             state=self.physical.phase if self.physical is not None else None),
                batch=self.callback_diagnostic.get('current_batch'))
            if control.handed_off:
                require(control.timing.failure is None,
                        'Camera delivery allocation failed before physical grip: '+str(control.timing.failure))
            if control.pickup_timing is not None:
                timing=control.pickup_timing
                timing.sample(control.now(),enforce=True,context=dict(phase='wrist_pickup'),
                              batch=self.callback_diagnostic.get('current_batch'))
                if timing.frames>=5 and timing.last_capture-timing.begin_ns>=200_000_000:
                    timing.status='PASS'


class IntegratedInspector(MovingInspectionExecutor):
    def __init__(self, *args):
        self.handed_off = False
        self.capture_monitoring = True
        self.route_lease = None
        self.pickup_timing = None
        self.deferred_images = []
        super().__init__(*args)

    def save_capture_image(self, path, rgb):
        # shortcut: two evidence views per pilot; revise this bound for additional views.
        require(len(self.deferred_images)<2, 'Capture evidence queue exceeded two images')
        self.deferred_images.append((path, rgb.copy()))

    def flush_capture_images(self):
        require(not self.capture_monitoring and not self.refinement_active and self.active_goal is None,
                'Stop camera control before writing capture evidence')
        records=[]
        while self.deferred_images:
            path, rgb=self.deferred_images[0]
            begin=time.monotonic()
            require(cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)),
                    'Capture evidence image write failed: '+str(path))
            records.append(dict(path=str(path),write_wall_s=time.monotonic()-begin))
            self.deferred_images.pop(0)
        return dict(status='PASS',scope='Image encoding and disk writes after control stops',images=records)

    def receive_joints(self, message):
        if not self.handed_off:
            super().receive_joints(message)

    def send_refinement(self):
        if not self.handed_off:
            super().send_refinement()

    def receive_targets(self, message):
        if not self.capture_monitoring:
            return
        if self.route_lease is not None:
            expected = self.route_lease.route
            targets = [b for b in message.targets if b.view_id in ('overhead','wrist')]
            if message.episode_id != expected['episode_id'] or not targets or any(
                    not b.valid or b.track_id != expected['track_id'] for b in targets):
                self.route_lease.invalid = 'Continuous overhead identity was lost after inspection'
        wrist=[b for b in message.targets if b.view_id=='wrist']
        if wrist:
            require(self.handed_off,'Wrist tracking started before the physical handoff')
            b=associated_snapshot(wrist[0])
            if self.pickup_timing is None:
                previous=self.timing.last_capture
                self.pickup_timing=CaptureTiming(self.mc,previous)
                self.pickup_timing.last_capture=previous
                self.pickup_timing.identity=(b['episode_id'],b['track_id'])
            self.pickup_timing.note(b,ns(message.header.stamp),self.now(),context=dict(phase='wrist_pickup'))
            if self.pickup_timing.pending:self.pickup_timing.pending[-1]['view_id']='wrist'
            return
        super().receive_targets(message)


def wrist_timing_report(control):
    if control.pickup_timing is None:return None
    report=control.pickup_timing.report()
    for key in ('first_violation','first_delivery_violation','worst_delivery_sample','worst_capture_gap_sample'):
        if report.get(key):
            report[key].pop('upstream_trace',None)
            report[key]['view_id']='wrist'
    report['upstream_diagnostic']=dict(scope='Detailed upstream timing trace is overhead-only')
    report['scope']='Observed wrist capture delivery, source stamps preserved; 50 ms delivery and 85 ms gap limits'
    return report


def inspect(control, output, integration):
    c = control.mc
    control.prime_timing()
    control.start_transport()
    start_ns = control.now()
    samples = []
    policy = None
    handoff = None
    try:
        control.wait_sim(2.)
        control.until(lambda: control.target()['predicted_x_m'] >= c['assess_trigger_x_m'],
                      timeout=c['entry_wait_wall_s'], sim_timeout=c['entry_wait_sim_s'])
        policy = MovingPolicy(control.episode, control.bound_track, control.now(), c, control.costs)
        for pose in ('assess', 'inspect_left'):
            if pose == 'inspect_left':
                policy.start_inspection(control.now())
            name = 'initial_assess' if pose == 'assess' else 'inspect_left'
            control.begin_operation(name, capture=True, returning=pose == 'inspect_left')
            control.move(pose)
            captured = control.capture(pose, output)
            for sample in captured:
                sample['class_evidence'] = sample.pop('classifier_diagnostic_only')
                sample['track_binding'] = bind_capture(sample, list(control.history), control.episode,
                    control.bound_track, c, control.filter_config)
            samples += captured
            evidence = captured[-1]
            lo, hi = c['classification_x_ranges'][pose]
            require(lo <= evidence['handle_xyz'][0] <= hi, 'Class capture is outside the tested moving domain')
            control.operation = None
            decision = policy.observe(track_id=control.bound_track, horizon=control.target()['remaining_s'],
                episode=control.episode, pose=pose, capture_ns=evidence['capture_ns'], now_ns=control.now(),
                probabilities=evidence['class_evidence']['probabilities'])
            decision['decision_time_ns'] = control.now()
            control.publish_decision(decision)
            if decision['action'] == 'PICK_READY':
                target = max((b for b in control.history if b['capture_ns'] <= evidence['capture_ns']),
                             key=lambda b: b['capture_ns'])
                d = dict(decision, frame_id='world', model_sha256=control.model.sha,
                    capture_ns=decision['evidence_capture_ns'], decision_ns=decision['decision_time_ns'],
                    deadline_ns=control.now()+round(decision['remaining_s']*1e9))
                wrist = dict(episode_id=evidence['episode_id'], capture_ns=evidence['capture_ns'],
                             frame_id='world', view_id='wrist', valid=True, position=evidence['handle_xyz'])
                handoff = check_handoff(d, target, wrist, control.owner,
                    dict(fresh=control.feedback_snapshot()['fresh'], fingers_open=all(control.q[n]>=.039
                         for n in ('panda_finger_joint1','panda_finger_joint2'))),
                    now_ns=control.now(), episode=control.episode, config=c,
                    filter_config=control.filter_config, model_sha=control.model.sha,
                    forecast=forecast, remaining=remaining)
                control.route_lease = RouteLease(decision, control.model.sha, integration['maximum_route_age_sim_s'])
            if decision['action'] != 'INSPECT':
                break
        if pose == 'inspect_left':
            control.begin_operation('return_assess', reserve=decision['action'] == 'PICK_READY')
            control.move('assess')
            control.operation = None
    except DeadlinePass as error:
        decision = dict(action='PASS', reason=str(error), episode_id=control.episode,
                        track_id=control.bound_track, decision_time_ns=control.now())
        control.route_lease = None
    if control.route_lease is not None:
        control.route_lease.check(control.overhead, control.now(), c)
        require(control.target()['remaining_s'] > c['approach_allocation_s']+c['close_s']+c['margin_s'],
                'Pickup slack disappeared during the return from inspection')
    return dict(decision=decision, samples=samples, policy=policy.history if policy else [],
                inspections=policy.inspections if policy else 0, handoff=handoff, transport_start_ns=start_ns)
