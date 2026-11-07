"""Use the inspected class with the existing physical pickup and placement checks."""
import copy
import math
import time
from pathlib import Path
import numpy as np
from ament_index_python.packages import get_package_share_directory
from camera_sort import SortController
from camera_pickup import Controller
from m4c_runtime import snapshot
from m4d_core import associated_snapshot
from m4c_core import remaining
from prediction_core import angle_error
from grasp_core import smooth
from ownership_core import require, ns
from sorting_interfaces.msg import ArmServoCommand
from m4a_runtime import stamp


class InspectedSortController(SortController):
    def __init__(self, node, executor, inspector, lease, moving_config, filter_config, config, episode_deadline, empty_tare):
        self.executor = executor
        self.inspector = inspector
        self.enabled = False
        self.lease = lease
        self.mc, self.fc_track, self.integration = moving_config, filter_config, config
        self.episode_wall_deadline = episode_deadline
        self.pickup_begin_ns = None
        self.empty_tare = empty_tare
        self.continuity_log = []
        self.wrench_wait=None
        self.wrench_wait_total=0.
        super().__init__(node)
        self.c['belt_speed_m_s'] = moving_config['belt_speed_m_s']
        self.c['acceleration_sigma_m_s2'] = filter_config['acceleration_sigma_m_s2']
        self.belt_speed = 0.

    def enable_pickup(self, lease, empty_tare):
        require(not self.enabled and not self.active and not self.input_enabled,'Physical pickup is already active')
        require(self.model_sha == lease.model_sha, 'Inspected classifier model differs from the installed model')
        self.lease=lease;self.empty_tare=empty_tare
        self.belt_speed=float(self.inspector.belt_speed)
        self.enabled=True
        self.send_belt()

    def command_publisher(self, node):
        return self.inspector.servo_publisher

    def transform_support(self, node):
        return self.inspector.buffer, None

    def belt_publisher(self, node):
        return self.inspector.belt

    def send_belt(self):
        if not self.enabled:
            return
        self.inspector.belt_speed = float(self.belt_speed)
        super().send_belt()

    def classifier_directory(self):
        return Path(get_package_share_directory('sorting_supervisor'))/'models/m4b_baseline'

    def start(self, request, response):
        response.accepted = False
        response.reason = 'Only the integrated inspection supervisor can start this controller'
        return response

    def receive_belief(self, message):
        if self.active and self.target_required and self.epoch and message.episode_id == self.epoch.episode_id:
            rows = [b for b in message.targets if b.track_id == self.lease.route['track_id']]
            if not rows or not rows[0].valid or rows[0].view_id not in ('overhead','wrist'):
                self.lease.invalid = 'Continuous overhead track was lost before physical grip'
        super().receive_belief(message)

    def current_route(self):
        require(self.lease is not None, 'No inspected class route is enabled')
        require(self.belief is not None, 'Awaiting the current overhead target')
        route = self.lease.check(associated_snapshot(self.belief), self.now_ns(), self.mc)
        if self.epoch is not None:
            require(route['episode_id'] == self.epoch.episode_id, 'Episode changed before physical pickup')
        return route

    def target(self, at_ns=None):
        self.current_route()
        return super().target(at_ns)

    def select_class(self, required_class=None):
        route = self.current_route()
        if required_class is not None:
            require(route['class_id'] == required_class, 'Inspected class route changed')
        return route

    def before_tracking(self):
        self.until(lambda:self.belief is not None and self.belief.view_id=='wrist'
                   and self.inspector.pickup_timing is not None and self.inspector.pickup_timing.status=='PASS',2.)
        require(math.isfinite(self.empty_tare), 'Stationary empty-wrist calibration is missing')
        samples = {}
        settings = self.s['supported_release']
        def empty_wrist_verified():
            require(self.owner.get('owner')=='HOLD' and not self.owner.get('busy')
                    and all(self.q.get(n,0)>.039 for n in ('panda_finger_joint1','panda_finger_joint2'))
                    and not self.scene.get('estimated_attachment'), 'Empty-wrist check requires open fingers and HOLD')
            try:stamp,force=self.wrist.sample()
            except RuntimeError as error:
                if 'unavailable' in str(error):return False
                raise
            if abs(force-self.empty_tare)>settings['tare_maximum_spread_N']:
                samples.clear();return False
            samples[stamp]=force
            return max(samples)-min(samples)>=round(settings['tare_span_s']*1e9)
        self.until(empty_wrist_verified,2.)
        self.wrist.tare=self.empty_tare
        self.result['wrist_tare_world_z_N']=self.empty_tare
        self.result['wrist_tare_source']='stationary_empty_hand_before_inspection'
        self.result['moving_empty_wrist_verification']='PASS'
        self.route=self.select_class(self.route['class_id'])
        self.result['class_decision']=copy.deepcopy(self.route)
        self.publish_decision(self.route)

    def check_execution(self):
        require(time.monotonic() < self.episode_wall_deadline, 'Integrated episode wall time exceeded its bound')
        if self.grip_candidate or not self.target_required:
            self.inspector.capture_monitoring = False
        if self.active and not self.cleanup and self.target_required and not self.grip_candidate:
            require(self.inspector.timing.failure is None,
                    'Camera delivery allocation failed before physical grip: '+str(self.inspector.timing.failure))
            route = self.current_route()
            target = snapshot(self.belief)
            horizon = remaining(target, self.now_ns(), self.mc, self.fc_track,tracking_view=target['view_id'])['remaining_s']
            require(horizon > self.mc['close_s']+self.mc['margin_s'], 'Current pickup deadline is too close')
            if self.pickup_begin_ns is not None:
                elapsed = (self.now_ns()-self.pickup_begin_ns)/1e9
                budget = self.mc['approach_allocation_s']+self.mc['close_s']
                require(elapsed < budget, 'Integrated moving approach exceeded its allocated time')
            deadline = self.now_ns()+round(horizon*1e9)
            if self.input_enabled:
                self.input_deadline = min(self.input_deadline, deadline)
                if self.motion_lease:
                    self.motion_lease = min(self.motion_lease, deadline)
            if not self.continuity_log or self.now_ns()-self.continuity_log[-1]['time_ns'] >= 40_000_000:
                self.continuity_log.append(dict(time_ns=self.now_ns(), capture_ns=target['capture_ns'],
                    class_age_sim_s=route['class_age_sim_s'], track_id=target['track_id'],
                    view_id=target['view_id'],remaining_s=horizon))

    def until(self, predicate, timeout=30.):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=.01)
            if self.active and not self.cleanup:
                require(not self.abort, 'Integrated pickup aborted')
                require(self.owner.get('owner') != 'FAULT', 'Arm gate fault: '+self.owner.get('reason', ''))
                if self.input_enabled:
                    require(self.owner.get('owner') == 'SERVO' and not self.owner.get('busy'), 'Servo ownership changed')
                self.check_execution()
            try:
                ready=predicate()
                self.finish_wrench_wait()
                if ready:return
            except RuntimeError as error:
                if not self.defer_wrench_tf(error):raise
        raise TimeoutError('Integrated pickup timeout in '+self.phase)

    def send(self):
        if not self.input_enabled:return super().send()
        self.check_execution()
        try:
            if self.wrench_wait is not None:
                self.wrist.sample()
                self.finish_wrench_wait()
            super().send()
            self.finish_wrench_wait()
        except RuntimeError as error:
            if not self.defer_wrench_tf(error):raise

    def finish_wrench_wait(self):
        if self.wrench_wait is None:return
        diagnostic=self.wrist.last_diagnostic or {}
        if diagnostic.get('status')!='PASS':return
        if not -.02<=(self.now_ns()-diagnostic['stamp_ns'])/1e9<self.s['supported_release']['maximum_wrench_age_s']:return
        elapsed=time.monotonic()-self.wrench_wait['begin_wall_s'];limit=self.integration['wrench_tf_wait_s']
        require(0<=elapsed<limit and 0<=self.now_ns()-self.wrench_wait['begin_ns']<round(limit*1e9)
                and self.wrench_wait_total+elapsed<self.integration['maximum_wrench_tf_wait_total_s'],
                'Capture-time wrist transform wait exceeded its bound')
        self.wrench_wait_total+=elapsed
        self.wrench_wait.update(status='RESUMED',end_ns=self.now_ns(),elapsed_wall_s=elapsed)
        self.wrench_wait=None

    def defer_wrench_tf(self,error):
        if str(error)!='Fresh capture-time wrist wrench unavailable' or self.phase not in ('lower','release'):
            return False
        d=self.wrist.last_diagnostic or {}
        settings=self.s['supported_release']
        pending=(d.get('status')=='UNAVAILABLE' and d.get('rejected',{}).get('capture_time_tf',0)>0
            and d.get('newest_wrench_stamp_ns',0)>(d.get('latest_tf_stamp_ns') or 0)
            and d.get('newest_wrench_age_s') is not None
            and -.02<=d['newest_wrench_age_s']<settings['maximum_wrench_age_s']
            and d.get('newest_receive_age_wall_s') is not None
            and d['newest_receive_age_wall_s']<settings['maximum_wrench_wall_age_s']
            and d.get('latest_tf_age_s') is not None and -.02<=d['latest_tf_age_s']<self.c['maximum_tf_age_s'])
        if not pending:return False
        now=self.now_ns();wall=time.monotonic()
        if self.wrench_wait is None:
            self.wrench_wait=dict(status='WAITING',begin_ns=now,begin_wall_s=wall,phase=self.phase,
                                  wrench_stamp_ns=d['newest_wrench_stamp_ns'],latest_tf_stamp_ns=d['latest_tf_stamp_ns'])
            self.result.setdefault('wrench_tf_waits',[]).append(self.wrench_wait)
        elapsed=wall-self.wrench_wait['begin_wall_s']
        limit=self.integration['wrench_tf_wait_s']
        require(0<=elapsed<limit and 0<=now-self.wrench_wait['begin_ns']<round(limit*1e9)
                and self.wrench_wait_total+elapsed<self.integration['maximum_wrench_tf_wait_total_s'],
                'Capture-time wrist transform wait exceeded its bound')
        if self.input_enabled:
            require(self.owner.get('owner')=='SERVO' and not self.owner.get('busy')
                    and self.owner.get('epoch')==self.input_epoch,'Wrist wait lost Servo ownership')
            message=ArmServoCommand(epoch=int(self.input_epoch),command_type=ArmServoCommand.TWIST,
                deadline=stamp(min(self.input_deadline,now+round(limit*1e9))))
            message.twist.header.frame_id='world';message.twist.header.stamp=stamp(now)
            self.publisher.publish(message)
        return True

    def approach_bin_with_wrist(self,goal,orientation):
        self.until(lambda:bool(self.wrist.sample()),timeout=.2)
        super().approach_bin_with_wrist(goal,orientation)

    def prepare_pregrasp(self, folder, result):
        self.pickup_begin_ns = self.now_ns()
        self.current_route()
        position, orientation = self.hand_pose()
        assess = np.array([.55, 0., .62])
        require(np.linalg.norm(position-assess) < self.integration['maximum_assess_entry_error_m'],
                'Physical pickup must start at the completed planned assess pose')
        desired = self.settings['grasp_quaternion_xyzw']
        require(angle_error(orientation, desired) < .015, 'Assess orientation is outside the pickup corridor')
        height = self.fc['fixed_handle_z_m']+.1029+.06
        self.set_phase('moving_pregrasp')
        self.acquire_mode('SERVO', seconds=25., kind=1)
        begin = self.now_ns()
        duration = self.integration['moving_pregrasp_duration_s']
        stable = None

        def reference(now):
            state, _, _ = self.target(now)
            blend, speed = smooth((now-begin)/1e9, duration)
            return [state[0], state[1], position[2]+(height-position[2])*blend], \
                   [state[2], state[3], (height-position[2])*speed], desired

        def settled():
            nonlocal stable
            target, _, _ = self.target()
            actual, attitude = self.hand_pose()
            good = (self.now_ns()-begin >= round(duration*1e9)
                    and np.linalg.norm(actual[:2]-target[:2]) < .002
                    and abs(actual[2]-height) < .002 and angle_error(attitude, desired) < .015)
            stable = (stable if stable is not None else self.now_ns()) if good else None
            return stable is not None and self.now_ns()-stable >= round(self.integration['moving_pregrasp_stable_s']*1e9)

        self.command_function = reference
        self.input_enabled = True
        self.until(settled, self.integration['moving_pregrasp_timeout_wall_s'])
        self.input_enabled = False
        self.acquire_mode('HOLD')
        current, _, _ = self.target()
        result['moving_pregrasp'] = dict(status='PASS', start_ns=begin, end_ns=self.now_ns(),
            belt_speed_m_s=self.belt_speed, mode='planned_assess_then_moving_camera_servo')
        result['checks']['camera_pregrasp'] = 'PASS'
        return [float(current[0]), float(current[1]), height], desired

    def diagnostic(self):
        result = super().diagnostic()
        result.update(class_route_invalid=self.lease.invalid, class_route_expires_ns=self.lease.expires,
                      continuity=copy.deepcopy(self.continuity_log[-5:]))
        return result
