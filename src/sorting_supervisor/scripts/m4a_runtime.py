"""Inspection executor. Inputs: RGB, calibration, episode identity, encoders and ownership.

This node has no evaluator-truth or oracle subscription. M4A moves through a
fixed development sweep; deadline proposals do not command a grasp.
"""
from collections import Counter
import json
import math
import time
import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time as TimeMessage
from moveit_msgs.srv import GetMotionPlan
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Float64, String
from sorting_interfaces.msg import CameraEpoch, ArmServoCommand
from sorting_interfaces.srv import SetArmOwner
from sorting_interfaces.action import ExecuteArmTrajectory
from tf2_ros import Buffer, TransformListener, TransformException
import yaml
from m4a_common import config, share
from camera_core import image_rgb, intrinsics, localize, quaternion_rotation
from camera_qos import camera_subscription_qos
from classifier_core import Classifier, fruit_crop, require
from m2b_runtime import plan_request, normalize_plan
from ownership_core import NAMES
from prediction_core import angle_error, tracking_twist


def ns(stamp):
    return stamp.sec*1_000_000_000+stamp.nanosec


def stamp(value):
    return TimeMessage(sec=value//1_000_000_000, nanosec=value%1_000_000_000)


class InspectionExecutor(Node):
    def __init__(self, preflight_directory):
        super().__init__('m4a_inspection_executor', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.c = config()
        self.preflight_directory = preflight_directory
        p = share('sorting_perception')
        self.lc = yaml.safe_load((p/'config/m3b_localization.yaml').read_text())
        self.cc = yaml.safe_load((p/'config/m3c.yaml').read_text())
        self.settings = yaml.safe_load((share('sorting_control')/'config/m2b.yaml').read_text())
        self.model = Classifier(p/'models/m3c_baseline', self.cc)
        self.owner, self.q, self.v = {}, {}, {}
        self.episode = None
        self.initial_episode = None
        self.epoch_barrier = 0
        self.q_received = 0.
        self.q_stamp = 0
        self.q_sequence = 0
        self.q_ignored = 0
        self.feedback_armed = False
        self.feedback_diagnostic = {}
        self.clock_last = 0
        self.failure = None
        self.info = {}
        self.pending = None
        self.rejections = Counter()
        self.active_goal = None
        self.refinement_active = False
        self.refinement_goal = None
        self.events = []
        self.buffer = Buffer(cache_time=Duration(seconds=20.))
        self.listener = TransformListener(self.buffer, self)
        camera_qos, metadata_qos = camera_subscription_qos()
        durable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(CameraEpoch, '/sorting/episode', self.receive_epoch, durable)
        self.create_subscription(Image, '/camera/wrist/image_raw', self.receive_image, camera_qos)
        self.create_subscription(CameraInfo, '/camera/wrist/camera_info', self.receive_info, metadata_qos)
        self.create_subscription(JointState, '/joint_states', self.receive_joints,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(String, '/sorting/arm/owner_state', self.receive_owner, 10)
        self.owner_client = self.create_client(SetArmOwner, '/sorting/arm/set_owner')
        self.planner = self.create_client(GetMotionPlan, '/m2b_planner/plan_kinematic_path')
        self.model_parameters = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        self.action = ActionClient(self, ExecuteArmTrajectory, '/sorting/arm/execute_trajectory')
        self.servo_publisher = self.create_publisher(ArmServoCommand, '/sorting/arm/servo_command', 1)
        self.belt = self.create_publisher(Float64, '/sorting/conveyor_command', 1)
        self.belt_speed = 0.0
        self.create_timer(.1, lambda: self.belt.publish(Float64(data=float(self.belt_speed))),
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.create_timer(.02, self.send_refinement, clock=Clock(clock_type=ClockType.STEADY_TIME))
        cv2.setNumThreads(1)

    def now(self):
        return self.get_clock().now().nanoseconds

    def receive_epoch(self, m):
        if self.episode is not None and m.episode_id != self.episode:
            self.failure = 'Episode changed during inspection; restart this test'
            self.pending = None
            self.info.clear()
        self.episode, self.epoch_barrier = m.episode_id, ns(m.header.stamp)

    def receive_joints(self, m):
        if ns(m.header.stamp) <= self.q_stamp:
            self.q_ignored += 1
            return
        self.q = dict(zip(m.name, m.position))
        self.v = dict(zip(m.name, m.velocity))
        self.q_received, self.q_stamp = time.monotonic(), ns(m.header.stamp)
        self.q_sequence += 1
        if all(n in self.q and n in self.v for n in NAMES):
            if not all(math.isfinite(self.q[n]) and math.isfinite(self.v[n]) for n in NAMES):
                self.failure = 'Non-finite joint feedback'
            if any(abs(self.v[n]) > self.c['maximum_joint_speed_rad_s'] for n in NAMES):
                self.failure = 'Inspection joint speed exceeded its bound'
        for n in ('panda_finger_joint1', 'panda_finger_joint2'):
            if n in self.q and self.q[n] < .039:
                self.failure = 'Inspection fingers must stay open'

    def receive_owner(self, m):
        self.owner = json.loads(m.data)
        if self.owner.get('owner') == 'FAULT' or self.owner.get('stats', {}).get('overlap_violations', 0):
            self.failure = 'Ownership fault: '+self.owner.get('reason', '')

    def receive_info(self, m):
        self.info[ns(m.header.stamp)] = m
        while len(self.info) > 64:
            self.info.pop(next(iter(self.info)))

    def receive_image(self, m):
        self.pending = (m, time.monotonic())

    def feedback_snapshot(self):
        now = self.now()
        wall_age = time.monotonic()-self.q_received if self.q_received else None
        sim_age = now-self.q_stamp if self.q_stamp else None
        positions = list(NAMES)+['panda_finger_joint1', 'panda_finger_joint2']
        missing_positions = [n for n in positions if n not in self.q]
        missing_velocities = [n for n in NAMES if n not in self.v]
        nonfinite = [n for n in positions if n in self.q and not math.isfinite(self.q[n])]
        nonfinite += [n+'/velocity' for n in NAMES if n in self.v and not math.isfinite(self.v[n])]
        violations = []
        if now <= 0: violations.append('simulation_clock_not_started')
        if missing_positions or missing_velocities: violations.append('missing_joint_fields')
        if nonfinite: violations.append('nonfinite_joint_fields')
        if wall_age is None or wall_age >= .5: violations.append('wall_feedback_timeout')
        if sim_age is None: violations.append('joint_stamp_not_started')
        elif sim_age >= 150_000_000: violations.append('joint_stamp_behind_clock')
        elif sim_age < -20_000_000: violations.append('joint_stamp_ahead_of_clock')
        return dict(fresh=not violations, clock_ns=now, joint_stamp_ns=self.q_stamp,
                    feedback_sequence=self.q_sequence, ignored_joint_samples=self.q_ignored,
                    feedback_age_wall_s=wall_age,
                    feedback_age_sim_s=sim_age/1e9 if sim_age is not None else None,
                    missing_positions=missing_positions, missing_velocities=missing_velocities,
                    nonfinite_fields=nonfinite, violations=violations)

    def prime_feedback(self):
        """Stay in HOLD until advancing feedback is coherent for 0.25 wall seconds."""
        stable_since, first_sequence = None, self.q_sequence
        self.events.append(dict(state='STARTUP_FEEDBACK', enter_ns=self.now()))

        def ready():
            nonlocal stable_since, first_sequence
            require(self.owner.get('owner') == 'HOLD' and not self.owner.get('busy'),
                    'Startup feedback check requires exclusive HOLD')
            self.feedback_diagnostic = self.feedback_snapshot()
            if not self.feedback_diagnostic['fresh']:
                stable_since, first_sequence = None, self.q_sequence
                return False
            if stable_since is None:
                stable_since, first_sequence = time.monotonic(), self.q_sequence
            return time.monotonic()-stable_since >= .25 and self.q_sequence-first_sequence >= 2

        try:
            self.until(ready, timeout=10.)
        except TimeoutError as error:
            raise TimeoutError('Startup joint feedback did not become coherent within 10 s: '+
                               json.dumps(self.feedback_diagnostic)) from error
        self.feedback_armed = True
        self.events.append(dict(state='FEEDBACK_READY', enter_ns=self.now(),
                                feedback=dict(self.feedback_diagnostic)))

    def until(self, predicate, *, timeout=15., sim_timeout=None, check=True):
        end, begin = time.monotonic()+timeout, self.now()
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=.01)
            if check:
                require(self.failure is None, self.failure)
                now = self.now()
                require(not self.clock_last or now >= self.clock_last, 'Simulation clock moved backwards')
                self.clock_last = now
                self.feedback_diagnostic = self.feedback_snapshot()
                if self.feedback_armed:
                    require(self.feedback_diagnostic['fresh'],
                            'Joint feedback became stale: '+json.dumps(self.feedback_diagnostic))
                if sim_timeout is not None:
                    require(now-begin < int(sim_timeout*1e9), 'Simulation state timeout')
            if predicate():
                return
        raise TimeoutError('M4A wall timeout; see events and capture rejections')

    def future(self, value, timeout=15., check=True):
        self.until(value.done, timeout=timeout, check=check)
        return value.result()

    def wait_sim(self, seconds):
        begin = self.now()
        self.until(lambda: self.now()-begin >= int(seconds*1e9), timeout=10.)

    def acquire(self, owner, seconds=35., command_type=0):
        req = SetArmOwner.Request(owner=owner, servo_command_type=command_type,
                                  deadline=stamp(self.now()+int(seconds*1e9)) if owner != 'HOLD' else TimeMessage())
        result = self.future(self.owner_client.call_async(req))
        require(result.success and result.owner == owner, result.reason)
        self.until(lambda: self.owner.get('owner') == owner and not self.owner.get('busy')
                   and self.owner.get('epoch') == result.epoch)
        return result.epoch, req.deadline

    def startup(self):
        self.until(lambda: self.now() > 0 and self.episode is not None and self.owner.get('ready')
                   and self.owner.get('owner') == 'HOLD' and len(self.q) >= 9
                   and self.owner_client.service_is_ready() and self.planner.service_is_ready()
                   and self.model_parameters.service_is_ready()
                   and self.action.server_is_ready(), timeout=60.)
        count = self.owner['stats']['planned_accepted']
        require(count == 0, f'Restart Terminal 1 before each M4A test: this launch already accepted {count} planned trajectories; read the earlier run for the original failure')
        self.initial_episode = self.episode
        values = self.future(self.model_parameters.call_async(
            GetParameters.Request(names=['robot_description', 'use_sim_time']))).values
        require(len(values) == 2 and values[0].type == ParameterType.PARAMETER_STRING
                and values[1].type == ParameterType.PARAMETER_BOOL and values[1].bool_value,
                'Missing robot model or simulation clock setting')
        import xml.etree.ElementTree as ET
        robot = ET.fromstring(values[0].string_value)
        path = robot.find(".//ros2_control/hardware/param[@name='mujoco_model']")
        require(path is not None and path.text in {str(self.preflight_directory/f'scene-{label}.xml')
                                                   for label in ('red', 'green')},
                'The running scene does not match this M4A preflight; restart Terminal 1')
        self.prime_feedback()
        self.wait_sim(.5)

    def hand_pose(self):
        transform = self.buffer.lookup_transform('world', 'panda_hand', Time())
        require(-20_000_000 <= self.now()-ns(transform.header.stamp) < 100_000_000, 'Hand transform is stale or future')
        p, q = transform.transform.translation, transform.transform.rotation
        return np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])

    def pose_error(self, name):
        p, q = self.hand_pose()
        return (float(np.linalg.norm(p-self.c['poses'][name])),
                angle_error(self.c['hand_quaternion_xyzw'], q))

    def send_refinement(self):
        if not self.refinement_active or self.failure is not None:
            return
        try:
            require(self.owner.get('owner') == 'SERVO' and not self.owner.get('busy')
                    and self.owner.get('epoch') == self.refinement_epoch, 'Inspection Servo ownership changed')
            require(self.now() < ns(self.refinement_deadline), 'Inspection Servo lease expired')
            p, q = self.hand_pose()
            linear, angular = tracking_twist(self.c['poses'][self.refinement_goal], [0., 0., 0.],
                                             self.c['hand_quaternion_xyzw'], p, q, self.c['refinement'])
            message = ArmServoCommand(epoch=int(self.refinement_epoch), deadline=self.refinement_deadline,
                                      command_type=ArmServoCommand.TWIST)
            message.twist.header.stamp = self.get_clock().now().to_msg()
            message.twist.header.frame_id = 'world'
            message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z = map(float, linear)
            message.twist.twist.angular.x, message.twist.twist.angular.y, message.twist.twist.angular.z = map(float, angular)
            self.servo_publisher.publish(message)
        except Exception as error:
            self.refinement_active = False
            self.failure = 'Inspection refinement: '+str(error)

    def refine(self, name):
        c = self.c['refinement']
        error, angle = self.pose_error(name)
        self.events.append(dict(state='PLANNED_ENDPOINT', pose=name, enter_ns=self.now(),
                                position_error_m=error, orientation_error_rad=angle))
        require(error < c['maximum_entry_position_error_m'] and angle < c['maximum_entry_orientation_error_rad'],
                f'Inspection plan ended outside the correction region: {error:.6f} m, {angle:.6f} rad')
        self.refinement_epoch, self.refinement_deadline = self.acquire(
            'SERVO', seconds=c['maximum_sim_s']+3., command_type=ArmServoCommand.TWIST)
        self.refinement_goal = name
        self.refinement_active = True
        self.events.append(dict(state='REFINE', pose=name, enter_ns=self.now(),
                                epoch=int(self.refinement_epoch)))
        stable_since = None

        def settled():
            nonlocal stable_since
            error, angle = self.pose_error(name)
            require(error < c['maximum_entry_position_error_m'] and angle < c['maximum_entry_orientation_error_rad'],
                    'Inspection refinement left its bounded correction region')
            if error < c['position_tolerance_m'] and angle < c['orientation_tolerance_rad']:
                if stable_since is None:
                    stable_since = self.now()
                return self.now()-stable_since >= int(c['stable_sim_s']*1e9)
            stable_since = None
            return False

        try:
            self.until(settled, timeout=c['maximum_wall_s'], sim_timeout=c['maximum_sim_s'])
        finally:
            self.refinement_active = False
        self.acquire('HOLD')
        self.wait_sim(self.c['settle_sim_s'])

    def move(self, name):
        require(self.feedback_armed, 'Joint feedback readiness has not been confirmed')
        self.events.append(dict(state='PLAN', pose=name, enter_ns=self.now(), wall=time.monotonic()))
        position = self.c['poses'][name]
        request = plan_request(self.q, position, self.c['hand_quaternion_xyzw'], self.settings)
        seed = dict(self.q)
        response = self.future(self.planner.call_async(request), timeout=12.)
        result = response.motion_plan_response
        require(result.error_code.val == 1, f'Inspection plan failed: {result.error_code.val}')
        trajectory = normalize_plan(result.trajectory.joint_trajectory)
        require(len(trajectory.points) >= 2, 'Inspection plan is empty')
        duration = ns(trajectory.points[-1].time_from_start)/1e9
        require(0 < duration < self.c['maximum_motion_sim_s'], 'Inspection plan exceeds duration bound')
        require(max(abs(self.q[n]-seed[n]) for n in NAMES) < .025, 'Robot moved away from planning seed')
        epoch, deadline = self.acquire('PLANNED', duration+5.)
        goal = ExecuteArmTrajectory.Goal(epoch=epoch, deadline=deadline, trajectory=trajectory)
        handle = self.future(self.action.send_goal_async(goal))
        require(handle.accepted, 'Command gate rejected inspection trajectory')
        self.active_goal = handle
        self.events.append(dict(state='MOVE', pose=name, enter_ns=self.now(), planned_duration_s=duration))
        response = self.future(handle.get_result_async(), timeout=self.c['maximum_motion_wall_s'])
        self.active_goal = None
        require(response.status == GoalStatus.STATUS_SUCCEEDED and response.result.success, response.result.reason)
        self.acquire('HOLD')
        self.wait_sim(self.c['settle_sim_s'])
        self.refine(name)
        error, angle = self.pose_error(name)
        require(error < self.c['maximum_position_error_m'] and angle < self.c['maximum_orientation_error_rad'],
                f'Physical inspection endpoint error: {error:.6f} m, {angle:.6f} rad')
        self.events.append(dict(state='SETTLED', pose=name, enter_ns=self.now(), position_error_m=error,
                                orientation_error_rad=angle))

    def capture(self, name, output):
        barrier, samples, last = self.now(), [], 0
        self.events.append(dict(state='CAPTURE', pose=name, enter_ns=barrier))

        def process():
            nonlocal last
            if self.pending is None:
                return False
            image, received = self.pending
            capture, now = ns(image.header.stamp), self.now()
            if capture <= max(barrier, last, self.epoch_barrier) or not 0 <= now-capture < int(self.c['maximum_capture_age_s']*1e9):
                self.rejections['previous_state_or_stale_capture'] += 1
                self.pending = None
                return False
            try:
                info = self.info.get(capture)
                if info is None:
                    raise LookupError('matching_camera_info_pending')
                require(image.header.frame_id == 'wrist_camera_optical_frame', 'Unexpected camera frame')
                rgb = image_rgb(image)
                k = intrinsics(info, rgb.shape, image.header.frame_id)
                tf = self.buffer.lookup_transform('world', image.header.frame_id, Time.from_msg(image.header.stamp))
                p, q = tf.transform.translation, tf.transform.rotation
                translation, xyzw = [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]
                rotation = quaternion_rotation(xyzw)
                loc = localize(rgb, k, translation, rotation, self.lc)
                crop, box = fruit_crop(rgb, k, translation, rotation, loc['position'], self.cc)
                age = (self.now()-capture)/1e9
                require(0 <= age < self.c['maximum_capture_age_s'], 'Capture expired during processing')
                result = dict(pose=name, episode_id=self.episode, capture_ns=capture, age_sim_s=age,
                              handle_xyz=loc['position'], camera_xyz=translation, camera_quaternion_xyzw=xyzw,
                              crop_xyxy=box, classifier_diagnostic_only=self.model.infer(crop))
            except (LookupError, TransformException) as error:
                if time.monotonic()-received < self.c['maximum_metadata_wait_wall_s']:
                    return False
                self.rejections[str(error)] += 1
                self.pending = None
                return False
            except ValueError as error:
                self.rejections[str(error)] += 1
                self.pending = None
                return False
            last, self.pending = capture, None
            if not samples:
                cv2.imwrite(str(output/f'{name}-{capture}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            samples.append(result)
            return len(samples) >= self.c['samples_per_pose']

        self.until(process, timeout=self.c['maximum_pose_wait_wall_s'])
        self.events.append(dict(state='CAPTURE_COMPLETE', pose=name, enter_ns=self.now(), frames=len(samples)))
        return samples

    def stop(self):
        # HOLD changes the owner epoch, so a late goal cannot execute after cleanup.
        self.belt_speed = 0.0
        self.refinement_active = False
        if self.active_goal is not None:
            self.active_goal.cancel_goal_async()
        end = time.monotonic()+12.
        while True:
            left = end-time.monotonic()
            require(left > 0, 'Could not confirm HOLD within the cleanup bound')
            self.until(lambda: self.owner and not self.owner.get('busy'), timeout=left, check=False)
            if self.owner.get('owner') == 'FAULT':
                raise RuntimeError('Ownership FAULT retained; restart the launch')
            if self.owner.get('owner') == 'HOLD':
                break
            req = SetArmOwner.Request(owner='HOLD')
            result = self.future(self.owner_client.call_async(req), timeout=max(.01, end-time.monotonic()), check=False)
            if not result.success and result.reason == 'An owner transition is already running':
                continue
            require(result.success, result.reason)
            self.until(lambda: self.owner.get('owner') == 'HOLD' and not self.owner.get('busy')
                       and self.owner.get('epoch') == result.epoch,
                       timeout=max(.01, end-time.monotonic()), check=False)
            break
        self.belt.publish(Float64(data=0.0))
