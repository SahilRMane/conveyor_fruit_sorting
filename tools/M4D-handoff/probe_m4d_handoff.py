#!/usr/bin/env python3
"""Run the installed moving inspection and audit its live pickup handoff."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from handoff_core import check_handoff, require
from ros_paths import prepare, write_json
from verify_m4c_gate import checked_gate, digest


def stop_child(child):
    """Give the inspection runner time to stop its belt and retain HOLD."""
    if child.poll() is not None:
        return False
    os.killpg(child.pid, signal.SIGINT)
    escalated = False
    for timeout, sig in ((20, signal.SIGTERM), (10, signal.SIGKILL)):
        try:
            child.wait(timeout=timeout)
            return escalated
        except subprocess.TimeoutExpired:
            escalated = True
            os.killpg(child.pid, sig)
    child.wait(timeout=5)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-dir', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--expected-class', choices=('red_fruit', 'green_fruit'), required=True)
    parser.add_argument('--expected-path', choices=('direct', 'inspect', 'unresolved', 'deadline'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare()
    gate = checked_gate(args.gate, args.preflight_dir)
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        parser.error('Use a new output directory to preserve previous evidence')
    out.mkdir(parents=True)
    source_hashes = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}

    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from sorting_interfaces.msg import Decision, CameraEpoch, HandleObservation, TargetBeliefArray
    from m4c_common import config
    from m4c_core import remaining
    from filter_core import forecast
    from m4c_runtime import FeedbackExecutor, snapshot
    from m4a_runtime import ns
    from ownership_core import NAMES
    import yaml

    c = config()
    fc = yaml.safe_load((args.preflight_dir / 'filter.yaml').read_text())
    rclpy.init()

    class Observer(Node):
        def __init__(self):
            super().__init__('m4d_handoff_audit', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.episode = None
            self.owner, self.joints = {}, None
            self.owner_wall = self.joint_wall = 0.
            self.targets, self.wrist = deque(maxlen=256), {}
            self.pending, self.decisions, self.checks, self.rejections = [], [], [], []
            self.clock_last = 0
            self.failure = None
            durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(CameraEpoch, '/sorting/episode', self.receive_epoch, durable)
            self.create_subscription(String, '/sorting/arm/owner_state', self.receive_owner, 10)
            self.create_subscription(JointState, '/joint_states', self.receive_joints,
                                     QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.create_subscription(TargetBeliefArray, '/sorting/m4c/targets', self.receive_targets, 10)
            self.create_subscription(HandleObservation, '/sorting/perception/wrist/handle_observation', self.receive_wrist, 10)
            self.create_subscription(Decision, '/sorting/m4c/decisions', self.receive_decision, 10)

        def receive_epoch(self, message):
            if self.episode is not None and self.episode != message.episode_id:
                self.failure = 'Episode changed during the handoff probe'
            self.episode = message.episode_id

        def receive_owner(self, message):
            self.owner = json.loads(message.data)
            self.owner_wall = time.monotonic()

        def receive_joints(self, message):
            if self.joints is None or ns(message.header.stamp) > ns(self.joints.header.stamp):
                self.joints, self.joint_wall = message, time.monotonic()

        def receive_targets(self, message):
            for belief in message.targets:
                if belief.view_id == 'overhead':
                    self.targets.append(snapshot(belief))

        def receive_wrist(self, message):
            self.wrist[(message.episode_id, ns(message.header.stamp))] = dict(
                episode_id=message.episode_id, capture_ns=ns(message.header.stamp),
                frame_id=message.header.frame_id, view_id=message.view_id, valid=message.valid,
                position=[message.position.x, message.position.y, message.position.z])
            while len(self.wrist) > 256:
                self.wrist.pop(next(iter(self.wrist)))

        def receive_decision(self, message):
            row = dict(action=message.action, episode_id=message.episode_id, track_id=message.track_id,
                       frame_id=message.header.frame_id, pose=message.selected_view,
                       decision_ns=ns(message.header.stamp), capture_ns=ns(message.evidence_capture_time),
                       deadline_ns=ns(message.deadline), class_id=message.class_id, proposed_bin=message.proposed_bin,
                       probabilities=list(message.class_probabilities), confidence=message.confidence,
                       estimated_service_time_s=message.estimated_service_time_s, model_sha256=message.model_sha256,
                       receipt_wall_s=time.monotonic())
            self.decisions.append(row)
            print('Observed policy action: ' + row['action'], flush=True)
            if message.action == 'PICK_READY':
                self.pending.append(row)

        def feedback(self, now):
            if self.joints is None:
                return dict(fresh=False, fingers_open=False)
            q = dict(zip(self.joints.name, self.joints.position))
            v = dict(zip(self.joints.name, self.joints.velocity))
            fingers = ('panda_finger_joint1', 'panda_finger_joint2')
            fields = all(n in q and n in v and math.isfinite(q[n]) and math.isfinite(v[n]) for n in NAMES)
            open_hand = all(n in q and math.isfinite(q[n]) and q[n] >= .039 for n in fingers)
            age_s = (now - ns(self.joints.header.stamp)) / 1e9
            fresh = (fields and -.02 <= age_s < c['maximum_capture_age_s']
                     and 0 <= time.monotonic()-self.joint_wall < c['maximum_source_wall_age_s'])
            return dict(fresh=fresh, fingers_open=open_hand, joint_age_sim_s=age_s)

        def check_pending(self):
            now = self.get_clock().now().nanoseconds
            if self.clock_last and now < self.clock_last:
                self.failure = 'Simulation clock moved backwards during the handoff probe'
            self.clock_last = now
            for decision in list(self.pending):
                identity = (decision['episode_id'], decision['capture_ns'])
                wrist = self.wrist.get(identity)
                candidates = [t for t in self.targets if t['episode_id'] == decision['episode_id']
                              and t['track_id'] == decision['track_id'] and t['capture_ns'] <= decision['capture_ns']]
                target = max(candidates, key=lambda t: t['capture_ns']) if candidates else None
                # Allow bounded cross-topic arrival without changing source timestamps.
                ahead = (decision['decision_ns'] > now or (self.joints is not None
                         and ns(self.joints.header.stamp) > now+20_000_000))
                if (wrist is None or target is None or ahead) and time.monotonic()-decision['receipt_wall_s'] < .05:
                    continue
                try:
                    require(self.failure is None, self.failure)
                    require(wrist is not None and target is not None, 'Exact handoff capture pair did not arrive within 50 ms')
                    require(time.monotonic()-self.owner_wall < c['maximum_source_wall_age_s'], 'Ownership feedback stopped')
                    # Reject identity replacement even if an older matching sample remains cached.
                    require(self.targets and self.targets[-1]['episode_id'] == decision['episode_id']
                            and self.targets[-1]['track_id'] == decision['track_id']
                            and self.targets[-1]['valid'], 'Current track changed or became invalid')
                    check = check_handoff(decision, target, wrist, self.owner, self.feedback(now),
                        now_ns=now, episode=self.episode, config=c, filter_config=fc,
                        model_sha=gate['model_sha256'], forecast=forecast, remaining=remaining)
                    self.checks.append(check)
                    print('Live handoff: PASS (' + check['class_id'] + ' -> ' + check['proposed_bin'] + ')', flush=True)
                except Exception as error:
                    self.rejections.append(dict(decision=decision, checked_ns=now, error=str(error)))
                self.pending.remove(decision)

    observer = Observer()
    executor = FeedbackExecutor(c['callback_budget_s'])
    executor.add_node(observer)
    child = None
    begin = time.monotonic()
    report = dict(checkpoint='M4D.1', status='FAIL', acceptance_passed=False, full_M4_gate_passed=False,
                  physical_grasp='NOT_RUN', physical_pickup_permitted=False,
                  scope='Live receipt-time handoff audit; physical execution is not enabled',
                  output_directory=str(out), gate_sha256=digest(args.gate),
                  installed_manifest=gate['installed_manifest'])
    try:
        ready_deadline = time.monotonic()+60
        while time.monotonic() < ready_deadline:
            executor.spin_once(timeout_sec=.01)
            if observer.episode is not None and observer.owner.get('owner') == 'HOLD' and observer.owner.get('ready'):
                break
        else:
            raise TimeoutError('Ready HOLD did not arrive within 60 seconds; check Terminal 1')
        command = ['ros2', 'run', 'sorting_supervisor', 'test_m4c.py',
                   '--preflight-dir', str(args.preflight_dir.resolve()),
                   '--expected-class', args.expected_class, '--expected-path', args.expected_path,
                   '--output-dir', str(out/'inspection')]
        with (out/'inspection.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print('Inspection started. Detailed output: ' + str(out/'inspection.log'), flush=True)
            deadline = time.monotonic()+300
            next_progress = time.monotonic()+20
            while child.poll() is None:
                executor.spin_once(timeout_sec=.01)
                observer.check_pending()
                require(time.monotonic() < deadline, 'Inspection child exceeded the 300 second wall limit')
                if time.monotonic() >= next_progress:
                    print('Inspection continues. Arm owner: ' + str(observer.owner.get('owner')), flush=True)
                    next_progress = time.monotonic()+20
            drain_deadline = time.monotonic()+.1
            while time.monotonic() < drain_deadline:
                executor.spin_once(timeout_sec=.005)
                observer.check_pending()
        report['inspection_exit_code'] = child.returncode
        require(child.returncode == 0, 'Installed inspection failed; read inspection/summary.json and inspection.log')
        summary = json.loads((out/'inspection/summary.json').read_text())
        timing = summary['timing_diagnostic']['observed']
        require(summary.get('status') == 'PASS' and summary.get('acceptance_passed') is True
                and summary.get('final_hold_status') == 'PASS' and summary.get('final_owner') == 'HOLD'
                and summary.get('overlap_violations') == 0, 'Inspection or final HOLD failed')
        require(summary.get('installed_manifest') == gate['installed_manifest'], 'Inspection source differs from the gate')
        require(timing.get('status') == 'PASS' and not timing.get('violation_counts'), 'Live inspection timing failed')
        require(observer.failure is None and not observer.pending and not observer.rejections,
                observer.failure or 'A live handoff check failed or remained pending')
        terminal = [d for d in observer.decisions if d['action'] in ('PICK_READY', 'PASS')]
        require(len(terminal) == 1, 'Expected exactly one live terminal decision')
        require(terminal[0]['action'] == summary['decision']['action']
                and terminal[0]['decision_ns'] == summary['decision']['decision_time_ns'], 'Live and saved decisions differ')
        pick = summary['decision']['action'] == 'PICK_READY'
        require(len(observer.checks) == (1 if pick else 0), 'Handoff checks do not match the decision')
        require(source_hashes == {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                'Handoff audit files changed during execution')
        report.update(status='PASS', acceptance_passed=True, handoff_status='PASS' if pick else 'NOT_REQUESTED',
                      policy_path=summary['policy_path'], final_hold_status='PASS', final_owner='HOLD',
                      overlap_violations=0, timing_status=timing['status'],
                      maximum_delivery_age_s=timing['maximum_delivery_age_s'])
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = str(error) or 'Interrupted'
    finally:
        if child is not None and child.poll() is None:
            try:
                report['stop_terminal_1_required'] = stop_child(child)
            except (ProcessLookupError, subprocess.TimeoutExpired) as error:
                report['cleanup_error'] = str(error)
                report['stop_terminal_1_required'] = True
        report.update(elapsed_wall_s=time.monotonic()-begin, handoff_checks=observer.checks,
                      rejected_handoffs=observer.rejections, live_decisions=observer.decisions,
                      observer_subscriptions=[s.topic_name for s in observer.subscriptions],
                      actuator_publishers=[], callback_batch_diagnostic=executor.callback_diagnostic,
                      audit_source_hashes=source_hashes)
        write_json(out/'summary.json', report)
        executor.remove_node(observer)
        observer.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ('installed_manifest', 'callback_batch_diagnostic', 'live_decisions')}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
