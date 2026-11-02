#!/usr/bin/env python3
"""Route one authorised arm source through the existing trajectory controller."""
import copy
import json
import math
import time
from pathlib import Path
import yaml

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.action.graph import get_action_client_names_and_types_by_node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointJog, JointTrajectoryControllerState
from hold_reference import select_hold
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import TwistStamped
from moveit_msgs.msg import ServoStatus
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import ServoCommandType, GetPlanningScene
from ament_index_python.packages import get_package_share_directory
from mujoco_ros2_control_msgs.srv import SetPause
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sorting_interfaces.action import ExecuteArmTrajectory
from sorting_interfaces.msg import ArmServoCommand
from sorting_interfaces.srv import SetArmOwner
from ownership_core import NAMES, Ownership, StoppedFeedbackWindow, ns, positions_valid, require, validate_trajectory

RAW_ACTION = "/panda_arm_controller/follow_joint_trajectory"
RAW_TOPIC = "/panda_arm_controller/joint_trajectory"
# This feedback is current state. Do not retain a queue of old states.
FEEDBACK_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class CommandGate(Node):
    def __init__(self):
        super().__init__("arm_command_gate", parameter_overrides=[Parameter("use_sim_time", value=True)])
        defaults = {"feedback_timeout_wall_s": 0.75, "servo_input_timeout_wall_s": 0.4,
                    "servo_output_timeout_wall_s": 0.5, "command_age_s": 0.15,
                    "transition_timeout_wall_s": 5.0, "stopped_velocity_rad_s": 0.01,
                    "maximum_lease_s": 60.0, "maximum_servo_joint_speed_rad_s": 0.10,
                    "maximum_servo_linear_speed_m_s": 0.05, "maximum_servo_angular_speed_rad_s": 0.10,
                    "maximum_first_servo_position_error_rad": 0.05,
                    "preserve_hold_reference": False, "maximum_hold_reference_delta_rad": .05,
                    "maximum_controller_reference_age_s": .15}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        # An explicit service prevents a second planning-scene server from
        # being selected during startup. Empty retains M2A auto-discovery.
        self.declare_parameter("planning_scene_service", "")
        self.group = ReentrantCallbackGroup()
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.core = Ownership(self, self.p("maximum_lease_s"))
        self.q, self.v = {}, {}
        self.feedback_wall = 0.0
        self.feedback_sequence = 0
        self.feedback_stamp_ns = 0
        self.feedback_ignored_samples = 0
        self.maximum_feedback_gap_wall_s = 0.0
        self.watchdog_wall = 0.0
        self.maximum_watchdog_gap_wall_s = 0.0
        self.last_stop = None
        self.controller_reference = None
        self.hold_reference_diagnostic = None
        self.resetting = False
        self.last_sim_ns = 0
        self.input_wall = self.output_wall = self.activation_wall = 0.0
        self.first_output = True
        self.servo_status = None
        self.servo_status_wall = 0.0
        self.plan_slot = False
        self.plan_deadline_ns = self.input_deadline_ns = 0
        self.backend_send = self.backend_goal = self.backend_result = None
        self.graph_pending = False
        self.next_graph_check = 0.0
        self.controller_snapshot = []
        self.scene_verified = False
        self.controller_streaming_verified = False
        self.controller_parameters = {}
        self.startup_stage = "created"
        self.last_timeout = None
        self.handoff_check = None
        self.last_handoff_feedback_gap = None
        self.failure_feedback = None
        self.service_paths = {"pause": "/sorting_servo/pause_servo",
                              "command_type": "/sorting_servo/switch_command_type",
                              "planning_scene": "/get_planning_scene",
                              "controller_parameters": "/panda_arm_controller/get_parameters"}
        self.events = self.create_publisher(String, "/sorting/arm/events", 100)
        self.state_pub = self.create_publisher(String, "/sorting/arm/owner_state", 10)
        self.raw_pub = self.create_publisher(JointTrajectory, RAW_TOPIC, 1)
        self.jog_pub = self.create_publisher(JointJog, "/sorting/arm/servo_joint_input", 1)
        self.twist_pub = self.create_publisher(TwistStamped, "/sorting/arm/servo_twist_input", 1)
        self.backend = ActionClient(self, FollowJointTrajectory, RAW_ACTION, callback_group=self.group)
        self.pause_client = self.create_client(SetBool, "/sorting_servo/pause_servo", callback_group=self.group)
        self.type_client = self.create_client(ServoCommandType, "/sorting_servo/switch_command_type", callback_group=self.group)
        self.controllers = self.create_client(ListControllers, "/controller_manager/list_controllers", callback_group=self.group)
        self.scene_client = self.create_client(GetPlanningScene, "/get_planning_scene", callback_group=self.group)
        self.controller_parameter_client = self.create_client(
            GetParameters, self.service_paths["controller_parameters"], callback_group=self.group)
        self.create_subscription(JointTrajectoryControllerState, "/panda_arm_controller/controller_state", self.receive_controller_reference, FEEDBACK_QOS, callback_group=self.group)
        self.create_subscription(JointState, "/joint_states", self.joints, FEEDBACK_QOS, callback_group=self.group)
        self.create_subscription(ArmServoCommand, "/sorting/arm/servo_command", self.servo_input, 1, callback_group=self.group)
        self.create_subscription(JointTrajectory, "/sorting/arm/servo_output", self.servo_output, 1, callback_group=self.group)
        self.create_subscription(ServoStatus, "/sorting/arm/servo_status", self.status, 10, callback_group=self.group)
        self.create_service(SetArmOwner, "/sorting/arm/set_owner", self.set_owner, callback_group=self.group)
        self.create_service(SetBool, "/sorting/arm/reset_guard", self.reset_guard, callback_group=self.group)
        self.server = ActionServer(self, ExecuteArmTrajectory, "/sorting/arm/execute_trajectory",
                                   self.execute, goal_callback=self.goal, cancel_callback=self.cancel,
                                   callback_group=self.group)
        self.create_timer(0.02, self.watchdog, clock=self.wall_clock, callback_group=self.group)
        self.create_timer(0.1, self.publish_state, clock=self.wall_clock, callback_group=self.group)

    def p(self, name):
        return self.get_parameter(name).value

    def now_ns(self):
        return self.get_clock().now().nanoseconds

    def event(self, event, **details):
        if event == "owner_transition_failed":
            # Capture before failure_stop pauses physics and ages the stream.
            self.failure_feedback = self.feedback_snapshot()
            details["feedback_at_failure"] = self.failure_feedback
        message = String()
        message.data = json.dumps({"event": event, "sim_time_ns": self.now_ns(),
                                   "wall_time": time.monotonic(), **details}, allow_nan=False)
        self.events.publish(message)
        if event == "owner_transition_failed":
            self.get_logger().error(message.data)
        elif event == "stop_requested":
            self.get_logger().warning(message.data)
        elif event in {"owner_activated", "startup_services_resolved", "collision_scene_verified"}:
            self.get_logger().info(message.data)

    def publish_state(self):
        with self.core.lock:
            state = self.core.snapshot()
            state.update(sim_time_ns=self.now_ns(), servo_status=self.servo_status,
                         planned_goal_active=self.plan_slot, controllers=self.controller_snapshot,
                         collision_scene_verified=self.scene_verified,
                         controller_streaming_verified=self.controller_streaming_verified,
                         controller_parameters=self.controller_parameters.copy(),
                         startup_stage=self.startup_stage, readiness=self.readiness_snapshot(),
                         preserve_hold_reference=self.p('preserve_hold_reference'), hold_reference=self.hold_reference_diagnostic,
                         last_timeout=self.last_timeout, service_paths=self.service_paths.copy(),
                         handoff_check=self.handoff_check, last_handoff_feedback_gap=self.last_handoff_feedback_gap,
                         feedback_at_failure=self.failure_feedback, last_stop=self.last_stop,
                         feedback_health=self.feedback_snapshot(), callback_executor="SingleThreadedExecutor",
                         resetting=self.resetting)
        message = String(); message.data = json.dumps(state, allow_nan=False)
        self.state_pub.publish(message)

    def joints(self, message):
        with self.core.lock:
            stamp = ns(message.header.stamp)
            if stamp <= self.feedback_stamp_ns:
                # A duplicate or delayed sample must not replace current state
                # or refresh the wall watchdog. Reset clears the stamp below.
                self.feedback_ignored_samples += 1
                return
            wall = time.monotonic()
            if self.feedback_wall:
                self.maximum_feedback_gap_wall_s = max(self.maximum_feedback_gap_wall_s, wall-self.feedback_wall)
            self.q = dict(zip(message.name, message.position))
            self.v = dict(zip(message.name, message.velocity))
            self.feedback_wall = wall
            self.feedback_stamp_ns = stamp
            self.feedback_sequence += 1

    def feedback_fresh(self):
        return self.feedback_snapshot()["fresh"]

    def feedback_snapshot(self):
        now = self.now_ns()
        wall_age = time.monotonic()-self.feedback_wall
        sim_age = now-self.feedback_stamp_ns
        missing_q = [n for n in NAMES if n not in self.q]
        missing_v = [n for n in NAMES if n not in self.v]
        nonfinite = [n for n in NAMES if n in self.q and n in self.v
                     and not (math.isfinite(self.q[n]) and math.isfinite(self.v[n]))]
        violations = []
        if missing_q or missing_v: violations.append("missing_joint_fields")
        if nonfinite: violations.append("nonfinite_joint_fields")
        if wall_age >= self.p("feedback_timeout_wall_s"): violations.append("wall_feedback_timeout")
        if sim_age >= 250_000_000: violations.append("joint_stamp_behind_clock")
        if sim_age < -50_000_000: violations.append("joint_stamp_ahead_of_clock")
        return {"sequence": self.feedback_sequence, "sim_time_ns": now,
                "joint_stamp_ns": self.feedback_stamp_ns,
                "age_wall_s": wall_age, "age_sim_s": sim_age/1e9,
                "wall_timeout_s": self.p("feedback_timeout_wall_s"),
                "maximum_sim_age_s": 0.25, "future_tolerance_s": 0.05,
                "fresh": not violations, "violations": violations,
                "missing_positions": missing_q, "missing_velocities": missing_v,
                "nonfinite_joints": nonfinite,
                "ignored_samples": self.feedback_ignored_samples,
                "maximum_feedback_gap_wall_s": self.maximum_feedback_gap_wall_s,
                "maximum_watchdog_gap_wall_s": self.maximum_watchdog_gap_wall_s}

    def readiness_snapshot(self):
        return {"use_sim_time": bool(self.get_parameter("use_sim_time").value),
                "simulation_clock_started": self.now_ns() > 0,
                "joint_feedback_fresh": self.feedback_fresh(),
                "missing_positions": [n for n in NAMES if n not in self.q],
                "missing_velocities": [n for n in NAMES if n not in self.v],
                "joint_feedback_sequence": self.feedback_sequence,
                "joint_stamp_ns": self.feedback_stamp_ns,
                "feedback_age_wall_s": time.monotonic()-self.feedback_wall,
                "feedback_age_sim_s": (self.now_ns()-self.feedback_stamp_ns)/1e9,
                "arm_action_ready": self.backend.server_is_ready(),
                "controller_service_ready": self.controllers.service_is_ready(),
                "controller_parameter_service_ready": self.controller_parameter_client.service_is_ready(),
                "controller_streaming_verified": self.controller_streaming_verified,
                "pause_service_ready": self.pause_client.service_is_ready(),
                "command_type_service_ready": self.type_client.service_is_ready(),
                "planning_scene_service_ready": self.scene_client.service_is_ready()}

    def resolve_servo_services(self):
        available = self.get_service_names_and_types()
        specifications = [("pause", "pause_client", SetBool, "std_srvs/srv/SetBool", "/pause_servo"),
                          ("command_type", "type_client", ServoCommandType, "moveit_msgs/srv/ServoCommandType", "/switch_command_type"),
                          ("planning_scene", "scene_client", GetPlanningScene, "moveit_msgs/srv/GetPlanningScene", None)]
        for key, attr, kind, type_name, suffix in specifications:
            candidates = [name for name, types in available if type_name in types and (suffix is None or name.endswith(suffix))]
            explicit = self.p("planning_scene_service") if key == "planning_scene" else ""
            if isinstance(explicit, str) and explicit:
                candidates = [name for name in candidates if name == explicit]
            preferred = self.service_paths[key]
            if preferred in candidates:
                selected = preferred
            elif len(candidates) == 1:
                selected = candidates[0]
            elif not candidates:
                return False
            else:
                raise RuntimeError(f"Ambiguous {type_name} services: {candidates}")
            if selected != preferred:
                old = getattr(self, attr)
                setattr(self, attr, self.create_client(kind, selected, callback_group=self.group))
                self.destroy_client(old)
                self.service_paths[key] = selected
        return True

    async def wait_until(self, predicate, timeout=None, label="condition"):
        timeout = self.p("transition_timeout_wall_s") if timeout is None else timeout
        future = Future(executor=self.executor)
        end = time.monotonic()+timeout
        def poll():
            if future.done():
                return
            try:
                if predicate():
                    future.set_result(True)
                elif time.monotonic() >= end:
                    self.last_timeout = {"operation": label, "stage": self.startup_stage,
                                         "timeout_wall_s": timeout, "readiness": self.readiness_snapshot()}
                    future.set_exception(TimeoutError(f"Ownership timeout: {label} ({timeout:g} wall seconds)"))
            except Exception as error:
                future.set_exception(error)
        timer = self.create_timer(0.01, poll, clock=self.wall_clock, callback_group=self.group)
        try:
            await future
        finally:
            self.destroy_timer(timer)

    async def wait_future(self, future, label="response"):
        await self.wait_until(future.done, label=label)
        return future.result()

    async def call(self, client, request):
        await self.wait_until(client.service_is_ready, label="discover service "+client.srv_name)
        return await self.wait_future(client.call_async(request), label="response from "+client.srv_name)

    async def wait_ready(self):
        timeout = 30.0 if not self.core.ready else self.p("transition_timeout_wall_s")
        self.startup_stage = "simulation_feedback"
        await self.wait_until(lambda: self.now_ns() > 0 and self.feedback_fresh(), timeout,
                              label="simulation clock and fresh joint feedback")
        self.startup_stage = "arm_controller"
        await self.wait_until(lambda: self.backend.server_is_ready() and self.controllers.service_is_ready(), timeout,
                              label="arm action and controller-manager service")
        self.startup_stage = "controller_streaming_parameters"
        await self.wait_until(self.controller_parameter_client.service_is_ready, timeout,
                              label="arm controller parameter service")
        await self.verify_controller_streaming()
        if self.p('preserve_hold_reference'):
            self.startup_stage = 'controller_reference'
            if not self.core.ready and not self.controller_reference_ready():
                # Some controller versions populate their reference after the
                # first trajectory. Bootstrap only with an open, unloaded hand.
                require(all(self.q.get(n,0.)>.039 for n in ('panda_finger_joint1','panda_finger_joint2')),
                        'Open fingers before bootstrapping controller reference')
                self.publish_hold(preserve_override=False)
            await self.wait_until(self.controller_reference_ready, timeout, label='fresh arm-controller position reference')
        self.startup_stage = "servo_services"
        await self.wait_until(self.resolve_servo_services, timeout, label="Servo and planning-scene service names")
        await self.wait_until(lambda: self.pause_client.service_is_ready() and self.type_client.service_is_ready()
                              and self.scene_client.service_is_ready(), timeout, label="Servo service discovery")
        self.event("startup_services_resolved", service_paths=self.service_paths.copy())
        self.startup_stage = "controller_graph"
        await self.check_graph()
        if not self.scene_verified:
            self.startup_stage = "collision_scene"
            expected = {b["id"] for b in yaml.safe_load((Path(get_package_share_directory("sorting_control"))/"config/fixtures.yaml").read_text())["boxes"]}
            end = time.monotonic()+10.0
            while True:
                request = GetPlanningScene.Request()
                request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
                response = await self.call(self.scene_client, request)
                received = {o.id for o in response.scene.world.collision_objects}
                if expected.issubset(received):
                    self.scene_verified = True
                    self.event("collision_scene_verified", fixture_boxes=len(expected))
                    break
                require(time.monotonic() < end, "Servo did not receive the collision fixtures")
                retry = time.monotonic()+0.1
                await self.wait_until(lambda: time.monotonic() >= retry)
        self.startup_stage = "ready"

    async def verify_controller_streaming(self):
        # Read the loaded controller, not only the YAML file on disk.
        self.controller_streaming_verified = False
        self.controller_parameters = {}
        request = GetParameters.Request()
        request.names = ["interpolate_from_desired_state", "allow_nonzero_velocity_at_trajectory_end"]
        response = await self.call(self.controller_parameter_client, request)
        require(len(response.values) == len(request.names), "Incomplete arm controller parameter response")
        self.controller_parameters = {
            name: value.bool_value if value.type == ParameterType.PARAMETER_BOOL else None
            for name, value in zip(request.names, response.values)}
        for name, value in self.controller_parameters.items():
            require(value is True, f"Arm controller requires {name}=true for Servo; rebuild and restart the M2A launch")
        self.controller_streaming_verified = True

    async def pause_servo(self, paused):
        request = SetBool.Request(); request.data = bool(paused)
        response = await self.call(self.pause_client, request)
        require(response.success, response.message)
        if not paused:
            with self.core.lock:
                self.input_wall = self.output_wall = 0.0
                self.input_deadline_ns = 0
                self.activation_wall = time.monotonic()
                self.first_output = True
                self.servo_status = None

    async def set_servo_type(self, kind):
        request = ServoCommandType.Request(); request.command_type = int(kind)
        response = await self.call(self.type_client, request)
        require(response.success, "Servo rejected command type")

    async def cancel_plan(self):
        await self.wait_until(lambda: not self.plan_slot or self.backend_send is not None)
        future = self.backend_send
        if future is not None:
            handle = await self.wait_future(future)
            if handle.accepted:
                result_future = handle.get_result_async()
                if not result_future.done():
                    await self.wait_future(handle.cancel_goal_async())
                response = await self.wait_future(result_future)
                require(response.status in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED),
                        "Backend goal has not reached a terminal state")
                self.event("backend_cancel_terminal", status=response.status)
        await self.wait_until(lambda: not self.plan_slot)

    def receive_controller_reference(self, message):
        stamp = ns(message.header.stamp)
        if self.controller_reference and stamp <= self.controller_reference['stamp_ns']:
            return
        self.controller_reference = dict(stamp_ns=stamp, received_wall=time.monotonic(),
                                         names=list(message.joint_names), positions=list(message.reference.positions))

    def controller_reference_ready(self):
        try:
            select_hold([float(self.q.get(n, float('nan'))) for n in NAMES], self.controller_reference,
                        self.now_ns(), time.monotonic(), True, self.p('maximum_hold_reference_delta_rad'),
                        self.p('maximum_controller_reference_age_s'), self.p('feedback_timeout_wall_s'))
            return True
        except (ValueError, RuntimeError, KeyError):
            return False

    def publish_hold(self, preserve_override=None):
        with self.core.lock:
            require(self.feedback_fresh(), "Cannot seed hold from stale joint feedback")
            measured = [float(self.q[n]) for n in NAMES]
            values, self.hold_reference_diagnostic = select_hold(
                measured, self.controller_reference, self.now_ns(), time.monotonic(),
                self.p('preserve_hold_reference') if preserve_override is None else preserve_override, self.p('maximum_hold_reference_delta_rad'),
                self.p('maximum_controller_reference_age_s'), self.p('feedback_timeout_wall_s'))
            trajectory = JointTrajectory(); trajectory.joint_names = NAMES
            point = JointTrajectoryPoint(); point.positions = values; point.velocities = [0.0]*7
            point.time_from_start.nanosec = 250_000_000
            trajectory.points = [point]
            self.raw_pub.publish(trajectory)
            self.event("hardware_hold_sent", source="HOLD", positions=values, measured_positions=measured, reference=self.hold_reference_diagnostic)
        return values

    async def hold_and_wait(self):
        sequence = self.feedback_sequence
        start = self.now_ns()
        self.publish_hold()
        window = StoppedFeedbackWindow(start, sequence, self.p("stopped_velocity_rad_s"))
        self.last_handoff_feedback_gap = None
        self.handoff_check = window.snapshot()
        in_gap = False
        def stopped():
            nonlocal in_gap
            with self.core.lock:
                fresh = self.feedback_fresh()
                if not fresh and not in_gap:
                    self.last_handoff_feedback_gap = self.feedback_snapshot()
                in_gap = not fresh
                ready = window.observe(self.feedback_sequence, self.feedback_stamp_ns, self.now_ns(),
                                       [self.v.get(n, float("nan")) for n in NAMES], fresh)
                self.handoff_check = window.snapshot()
                return ready
        # Keep all motion gates closed while a transient feedback delay clears.
        # Existing freshness limits and the wall timeout remain unchanged.
        end = time.monotonic()+self.p("transition_timeout_wall_s")
        while True:
            remaining = end-time.monotonic()
            require(remaining > 0, "Stopped feedback proof expired during handoff")
            await self.wait_until(stopped, timeout=remaining, label="fresh stopped joint samples after hold")
            with self.core.lock:
                # A new message may arrive between predicate completion and
                # coroutine resumption. Check the current sample again.
                if stopped():
                    return [float(self.q[n]) for n in NAMES]

    async def failure_stop(self):
        # On uncertain cancellation, override output with a hold and pause physics.
        try:
            self.publish_hold()
        except Exception as error:
            self.event("failure_hold_unavailable", error=str(error))
        try:
            names = [name for name, types in self.get_service_names_and_types()
                     if "mujoco_ros2_control_msgs/srv/SetPause" in types]
            require(len(names) == 1, "Cannot identify one simulation pause service")
            client = self.create_client(SetPause, names[0], callback_group=self.group)
            request = SetPause.Request(); request.paused = True
            response = await self.call(client, request)
            require(response.success, response.message)
            self.event("failure_simulation_paused")
            self.destroy_client(client)
        except Exception as error:
            self.event("failure_pause_unavailable", error=str(error))

    async def check_graph(self):
        response = await self.call(self.controllers, ListControllers.Request())
        rows = [{"name": c.name, "state": c.state, "claimed_interfaces": list(c.claimed_interfaces)} for c in response.controller]
        arm_claims = {n+"/position" for n in NAMES}
        active = [r for r in rows if r["state"] == "active" and arm_claims.intersection(r["claimed_interfaces"])]
        require(len(active) == 1 and active[0]["name"] == "panda_arm_controller"
                and arm_claims.issubset(active[0]["claimed_interfaces"]), "Unexpected active arm controller claims")
        for info in self.get_publishers_info_by_topic(RAW_TOPIC):
            require(info.node_name == self.get_name() and info.node_namespace == self.get_namespace(),
                    "Another node publishes directly to the arm controller")
        servo_publishers = self.get_publishers_info_by_topic("/sorting/arm/servo_output")
        require(len(servo_publishers) == 1 and servo_publishers[0].node_name == "sorting_servo"
                and servo_publishers[0].node_namespace == "/", "Servo output must have one installed Servo publisher")
        require(len(self.get_publishers_info_by_topic("/sorting/arm/servo_command")) <= 1,
                "More than one source publishes Servo input")
        for name, namespace in self.get_node_names_and_namespaces():
            if name == self.get_name() and namespace == self.get_namespace():
                continue
            for action, _ in get_action_client_names_and_types_by_node(self, name, namespace):
                require(action != RAW_ACTION, "Another node bypasses the planned-motion gate")
        with self.core.lock:
            self.controller_snapshot = rows

    async def graph_check_task(self):
        try:
            await self.check_graph()
        except Exception as error:
            self.request_stop("graph_violation: "+str(error))
        finally:
            self.graph_pending = False

    def request_stop(self, reason):
        with self.core.lock:
            previous = self.core.snapshot()
            feedback = self.feedback_snapshot()
            ticket = self.core.reserve_stop(reason)
            if ticket is not None:
                self.last_stop = {"reason": reason, "from_owner": previous["owner"],
                                  "from_epoch": previous["epoch"], "stop_epoch": ticket,
                                  "feedback": feedback, "wall_time": time.monotonic()}
                self.event("stop_requested", **self.last_stop)
                self.executor.create_task(self.core.finish(ticket))

    def watchdog(self):
        with self.core.lock:
            now = self.now_ns(); wall = time.monotonic()
            if self.watchdog_wall:
                self.maximum_watchdog_gap_wall_s = max(self.maximum_watchdog_gap_wall_s, wall-self.watchdog_wall)
            self.watchdog_wall = wall
            clock_reversed = self.last_sim_ns > 0 and now < self.last_sim_ns-1_000_000
            self.last_sim_ns = now
            if clock_reversed and self.core.ready:
                ticket = self.core.invalidate_pending_transition("simulation_clock_reset")
                if ticket is not None:
                    self.executor.create_task(self.core.finish(ticket))
                # Permit samples from the new episode, but do not reuse the old
                # q/v or stamp as proof that its arm feedback is fresh.
                self.q, self.v = {}, {}
                self.controller_reference = None
                self.feedback_wall = 0.0
                self.feedback_stamp_ns = 0
            if self.core.owner in {"PLANNED", "SERVO"} and not self.core.busy:
                if not self.feedback_fresh():
                    self.request_stop("feedback_lost")
                elif now >= self.core.deadline_ns:
                    self.request_stop("owner_deadline")
                elif self.core.owner == "PLANNED" and self.plan_slot and now >= self.plan_deadline_ns:
                    self.request_stop("planned_command_deadline")
                elif self.core.owner == "SERVO":
                    if self.input_deadline_ns and now >= self.input_deadline_ns:
                        self.request_stop("servo_command_deadline")
                    elif wall-self.activation_wall > self.p("servo_input_timeout_wall_s") and (
                            not self.input_wall or wall-self.input_wall > self.p("servo_input_timeout_wall_s")):
                        self.request_stop("servo_input_lost")
                    elif self.input_wall and wall-self.activation_wall > self.p("servo_output_timeout_wall_s") and (
                            not self.output_wall or wall-self.output_wall > self.p("servo_output_timeout_wall_s")):
                        self.request_stop("servo_output_lost")
                    elif self.input_wall and wall-self.activation_wall > 0.75 and wall-self.servo_status_wall > 0.5:
                        self.request_stop("servo_status_lost")
                    elif self.input_wall and wall-self.activation_wall > 0.25 and self.servo_status in (-1, 2, 5, 6):
                        self.request_stop("servo_halt_"+str(self.servo_status))
            if self.core.ready and not self.core.busy and wall >= self.next_graph_check and not self.graph_pending:
                self.next_graph_check = wall+1.0
                self.graph_pending = True
                self.executor.create_task(self.graph_check_task())

    def status(self, message):
        with self.core.lock:
            self.servo_status = int(message.code)
            self.servo_status_wall = time.monotonic()

    async def set_owner(self, request, response):
        try:
            response.success = await self.core.change(request.owner, ns(request.deadline), request.servo_command_type)
        except Exception as error:
            response.success = False
            response.reason = str(error)
        state = self.core.snapshot()
        response.epoch = state["epoch"]; response.owner = state["owner"]
        if not response.reason:
            response.reason = state["reason"]
        return response

    async def reset_guard(self, request, response):
        try:
            if request.data:
                with self.core.lock:
                    require(not self.resetting and self.core.ready and not self.core.busy and self.core.owner == "HOLD",
                            "Begin reset only from verified HOLD")
                    require(not self.plan_slot and self.backend_send is None, "A backend goal remains")
                    require(self.feedback_fresh() and max(abs(self.v[n]) for n in NAMES) < self.p("stopped_velocity_rad_s"),
                            "Reset requires fresh stopped arm feedback")
                    self.core.epoch += 1; self.core.owner = "RESETTING"; self.core.busy = True
                    self.core.ready = False; self.core.deadline_ns = 0; self.core.reason = "guarded_scene_reset"
                    self.resetting = True
                    self.core.event("reset_guard_entered")
                response.success = True; response.message = "All arm command paths closed for reset"
            else:
                with self.core.lock:
                    require(self.resetting and self.core.owner == "RESETTING", "No reset guard is active")
                    self.core.busy = False; self.core.owner = "HOLD"; self.resetting = False
                response.success = await self.core.change("HOLD", reason="reset_complete")
                response.message = self.core.reason
        except Exception as error:
            response.success = False; response.message = str(error)
        return response

    def goal(self, request):
        with self.core.lock:
            try:
                require(self.core.allowed("PLANNED", request.epoch, ns(request.deadline), self.now_ns()), "Planned source does not own the arm")
                require(not self.plan_slot, "A planned goal is already active")
                require(self.feedback_fresh(), "Stale joint feedback")
                validate_trajectory(request.trajectory, self.now_ns(), ns(request.deadline))
                self.plan_slot = True
                self.plan_deadline_ns = ns(request.deadline)
                self.core.stats["planned_accepted"] += 1
                self.core.event("planned_goal_accepted")
                return GoalResponse.ACCEPT
            except Exception as error:
                self.core.stats["planned_rejected"] += 1
                self.core.event("planned_goal_rejected", reason=str(error))
                return GoalResponse.REJECT

    def cancel(self, handle):
        self.request_stop("planned_goal_cancelled")
        return CancelResponse.ACCEPT

    async def execute(self, handle):
        result = ExecuteArmTrajectory.Result(); result.success = False; result.controller_error_code = -1
        try:
            with self.core.lock:
                request = handle.request
                require(self.core.allowed("PLANNED", request.epoch, ns(request.deadline), self.now_ns()), "Owner changed before trajectory dispatch")
                raw = FollowJointTrajectory.Goal(); raw.trajectory = request.trajectory
                self.backend_send = self.backend.send_goal_async(raw)
                self.core.event("hardware_planned_sent", source="PLANNED")
            raw_handle = await self.wait_future(self.backend_send)
            require(raw_handle.accepted, "Arm controller rejected the trajectory")
            with self.core.lock:
                self.backend_goal = raw_handle
                self.backend_result = raw_handle.get_result_async()
            feedback = ExecuteArmTrajectory.Feedback(); feedback.phase = "EXECUTING"; handle.publish_feedback(feedback)
            # The lease watchdog handles motion deadlines. This wait is bounded in wall time.
            await self.wait_until(self.backend_result.done, 90.0)
            raw_result = self.backend_result.result()
            result.controller_error_code = int(raw_result.result.error_code)
            self.core.event("backend_goal_terminal", status=raw_result.status,
                            controller_error_code=result.controller_error_code,
                            controller_message=raw_result.result.error_string)
            with self.core.lock:
                valid = self.core.allowed("PLANNED", request.epoch, ns(request.deadline), self.now_ns())
            if handle.is_cancel_requested:
                handle.canceled(); result.reason = "cancelled"
            elif valid and raw_result.status == GoalStatus.STATUS_SUCCEEDED and result.controller_error_code == 0:
                handle.succeed(); result.success = True; result.reason = "completed"
            else:
                handle.abort()
                result.reason = (f"owner_changed: {self.core.reason}; backend_status={raw_result.status}; "
                                 f"controller_error_code={result.controller_error_code}; "
                                 f"controller_message={raw_result.result.error_string}")
        except Exception as error:
            result.reason = str(error)
            if handle.is_cancel_requested:
                handle.canceled()
            else:
                handle.abort()
            # Retain the backend handle until terminal cancellation is confirmed.
            # If execution failed unexpectedly, do not restore an apparently
            # healthy owner while an uncertain controller goal can still run.
            if self.backend_send is not None:
                try:
                    raw_handle = await self.wait_future(self.backend_send)
                    if raw_handle.accepted:
                        await self.wait_future(raw_handle.cancel_goal_async())
                        terminal = await self.wait_future(raw_handle.get_result_async())
                        require(terminal.status in (4, 5, 6), "Backend cancellation is uncertain")
                except Exception as cancel_error:
                    self.event("backend_cancellation_failed", error=str(cancel_error))
                with self.core.lock:
                    self.core.owner = "FAULT"; self.core.ready = False; self.core.epoch += 1
                    self.core.deadline_ns = 0; self.core.reason = "planned_execution_failed"
                await self.failure_stop()
        finally:
            with self.core.lock:
                self.plan_slot = False
                self.plan_deadline_ns = 0
                self.backend_send = self.backend_goal = self.backend_result = None
        return result

    def servo_input(self, message):
        with self.core.lock:
            authorized = self.core.allowed("SERVO", message.epoch, ns(message.deadline), self.now_ns())
            try:
                require(authorized, "Servo source has no current ownership epoch")
                require(message.command_type == self.core.command_type, "Wrong Servo input type")
                command = message.joint_jog if message.command_type == 0 else message.twist
                age = self.now_ns()-ns(command.header.stamp)
                require(ns(command.header.stamp) >= self.core.activation_ns and -20_000_000 <= age < self.p("command_age_s")*1e9,
                        "Stale Servo input")
                require(command.header.frame_id == "world", "Servo input frame must be world")
                if message.command_type == 0:
                    names, values = list(command.joint_names), list(command.velocities)
                    require(names and len(names) == len(values) and len(set(names)) == len(names)
                            and set(names).issubset(NAMES) and not command.displacements, "Invalid joint jog")
                    require(all(math.isfinite(v) and abs(v) <= self.p("maximum_servo_joint_speed_rad_s") for v in values), "Joint jog speed exceeds bound")
                    self.jog_pub.publish(command)
                else:
                    linear = (command.twist.linear.x, command.twist.linear.y, command.twist.linear.z)
                    angular = (command.twist.angular.x, command.twist.angular.y, command.twist.angular.z)
                    require(all(math.isfinite(v) for v in linear+angular), "Non-finite twist")
                    require(math.sqrt(sum(v*v for v in linear)) <= self.p("maximum_servo_linear_speed_m_s")
                            and math.sqrt(sum(v*v for v in angular)) <= self.p("maximum_servo_angular_speed_rad_s"), "Twist exceeds speed bound")
                    self.twist_pub.publish(command)
                self.input_wall = time.monotonic()
                self.input_deadline_ns = ns(message.deadline)
                self.core.stats["servo_input_accepted"] += 1
            except Exception as error:
                self.core.stats["servo_input_rejected"] += 1
                self.core.event("servo_input_rejected", reason=str(error))
                if authorized:
                    self.request_stop("invalid_servo_input")
                elif self.core.owner == "SERVO" and message.epoch == self.core.epoch and ns(message.deadline) <= self.now_ns():
                    self.request_stop("owner_deadline" if self.now_ns() >= self.core.deadline_ns else "servo_command_deadline")

    def servo_output(self, message):
        with self.core.lock:
            try:
                if self.core.owner == "SERVO" and not self.core.busy and self.plan_slot:
                    self.core.stats["overlap_violations"] += 1
                    self.request_stop("output_owner_invariant")
                require(self.core.owner == "SERVO" and not self.core.busy and self.core.ready
                        and self.now_ns() < self.core.deadline_ns and not self.plan_slot, "Servo output gate is closed")
                require(self.input_wall and time.monotonic()-self.input_wall < self.p("servo_input_timeout_wall_s"), "No fresh authorised Servo input")
                deadline = min(self.core.deadline_ns, self.input_deadline_ns)
                require(self.now_ns() < deadline, "Servo command deadline expired")
                age = self.now_ns()-ns(message.header.stamp)
                require(ns(message.header.stamp) >= self.core.activation_ns and -50_000_000 <= age < 200_000_000,
                        "Old Servo output")
                validate_trajectory(message, self.now_ns(), deadline, servo=True)
                end_time = ns(message.header.stamp)+ns(message.points[-1].time_from_start)
                require(self.now_ns() < end_time < deadline,
                        "Servo trajectory is past or crosses the command deadline")
                if self.first_output:
                    require(self.feedback_fresh(), "Stale Servo seed feedback")
                    require(max(abs(q-self.q[n]) for q, n in zip(message.points[0].positions, NAMES))
                            < self.p("maximum_first_servo_position_error_rad"), "Servo seed differs from measured state")
                self.raw_pub.publish(copy.deepcopy(message))
                self.first_output = False
                self.output_wall = time.monotonic()
                self.core.stats["servo_output_forwarded"] += 1
            except Exception as error:
                self.core.stats["servo_output_dropped"] += 1
                if self.core.owner == "SERVO" and not self.core.busy and ns(message.header.stamp) >= self.core.activation_ns:
                    self.core.event("servo_output_rejected", reason=str(error))


def main():
    rclpy.init()
    node = CommandGate()
    # All waits below are asynchronous. They yield to service responses,
    # feedback, cancellation and timers on this same executor thread.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.create_task(node.core.change("HOLD"))
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
