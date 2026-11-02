"""MoveIt planning and bounded Cartesian commands through the verified M2A gate."""
import copy
import math
import time
import numpy as np
from rclpy.time import Time as RosTime
from tf2_ros import Buffer, TransformListener
from builtin_interfaces.msg import Time
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, RobotState, PlanningSceneComponents, JointConstraint
from moveit_msgs.srv import GetMotionPlan, GetPositionFK, GetPlanningScene, GetStateValidity
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64
from sorting_interfaces.action import ExecuteArmTrajectory
from sorting_interfaces.msg import ArmServoCommand, OracleTarget, OraclePrediction
from sorting_interfaces.srv import SetArmOwner
from test_arm_ownership import Probe
from m2b_common import config, observation, set_pose, timestamp, xyz, xyzw
from prediction_core import angle_error, tracking_twist
from ownership_core import NAMES, require, ns, validate_trajectory


def robot_state(q):
    state = RobotState(); state.is_diff = False
    state.joint_state.name = NAMES+["panda_finger_joint1", "panda_finger_joint2"]
    state.joint_state.position = [float(q[n]) for n in state.joint_state.name]
    return state


def plan_request(q, position, quaternion, settings, joint_goal=None):
    service = GetMotionPlan.Request(); request = service.motion_plan_request
    request.group_name = settings["group_name"]; request.pipeline_id = "ompl"; request.planner_id = "RRTConnectkConfigDefault"
    request.num_planning_attempts = 1; request.allowed_planning_time = float(settings["planning_time_s"])
    request.max_velocity_scaling_factor = float(settings["planning_velocity_scale"])
    request.max_acceleration_scaling_factor = float(settings["planning_acceleration_scale"])
    request.start_state = robot_state(q)
    request.workspace_parameters.header.frame_id = "world"
    request.workspace_parameters.min_corner.x = -0.3; request.workspace_parameters.min_corner.y = -0.7
    request.workspace_parameters.min_corner.z = 0.0
    request.workspace_parameters.max_corner.x = 1.1; request.workspace_parameters.max_corner.y = 0.7
    request.workspace_parameters.max_corner.z = 1.2
    constraint = Constraints(); constraint.name = "M2B_handle_pregrasp"
    pc = PositionConstraint(); pc.header.frame_id = "world"; pc.link_name = settings["hand_link"]; pc.weight = 1.0
    pc.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.004]*3)]
    pose = Pose(); set_pose(pose, position, [0.0, 0.0, 0.0, 1.0]); pc.constraint_region.primitive_poses = [pose]
    oc = OrientationConstraint(); oc.header.frame_id = "world"; oc.link_name = settings["hand_link"]; oc.weight = 1.0
    oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = map(float, quaternion)
    oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.02
    oc.parameterization = OrientationConstraint.ROTATION_VECTOR
    constraint.position_constraints = [pc]; constraint.orientation_constraints = [oc]
    if joint_goal is not None:
        constraint.position_constraints = []; constraint.orientation_constraints = []
        constraint.joint_constraints = [JointConstraint(joint_name=n, position=float(v), tolerance_above=0.001,
                                                       tolerance_below=0.001, weight=1.0) for n, v in zip(NAMES, joint_goal)]
    request.goal_constraints = [constraint]
    return service


def normalize_plan(trajectory):
    trajectory = copy.deepcopy(trajectory)
    require(set(trajectory.joint_names) == set(NAMES) and len(trajectory.joint_names) == 7, "Unexpected MoveIt trajectory joints")
    indices = [trajectory.joint_names.index(n) for n in NAMES]
    for point in trajectory.points:
        for name in ("positions", "velocities", "accelerations", "effort"):
            values = list(getattr(point, name))
            if values:
                require(len(values) == 7, "Incomplete MoveIt trajectory field")
                setattr(point, name, [float(values[i]) for i in indices])
    trajectory.joint_names = NAMES; trajectory.header.stamp = Time()
    return trajectory


class M2BProbe(Probe):
    def __init__(self, node):
        self.settings = config(); self.q_stamp_ns = 0
        self.oracle = self.prediction = None; self.oracle_history = []; self.prediction_history = []
        self.oracle_received_wall = 0.0; self.allow_belt = False; self.belt_speed = 0.0
        self.command_function = None
        super().__init__(node)
        self.buffer = Buffer(); self.listener = TransformListener(self.buffer, node, spin_thread=False)
        self.planner = node.create_client(GetMotionPlan, "/m2b_planner/plan_kinematic_path")
        self.fk_client = node.create_client(GetPositionFK, "/m2b_planner/compute_fk")
        self.validity = node.create_client(GetStateValidity, "/m2b_planner/check_state_validity")
        self.scene_clients = [node.create_client(GetPlanningScene, p) for p in
                              ("/get_planning_scene", "/m2b_planner/get_planning_scene")]
        self.planner_params = node.create_client(GetParameters, "/m2b_planner/move_group/get_parameters")
        self.belt = node.create_publisher(Float64, "/sorting/conveyor_command", 1)
        node.create_subscription(OracleTarget, "/sorting/debug/oracle_target", self.receive_oracle, 1)
        node.create_subscription(OraclePrediction, "/sorting/debug/oracle_prediction", self.receive_prediction, 1)
        from rclpy.clock import Clock, ClockType
        node.create_timer(0.1, self.send_belt, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def receive_joints(self, message):
        if ns(message.header.stamp) <= self.q_stamp_ns: return
        super().receive_joints(message); self.q_stamp_ns = ns(message.header.stamp)

    def ensure_oracle_current(self):
        require(self.oracle is not None, "Oracle target is missing")
        age = (self.now_ns()-self.oracle.stamp_ns)/1e9
        require(-0.02 <= age < self.settings["maximum_observation_age_s"], "Oracle target is stale or future during motion")
        require(time.monotonic()-self.oracle_received_wall < self.settings["maximum_source_wall_age_s"],
                "Oracle source stopped during motion")

    def receive_owner(self, message):
        super().receive_owner(message)
        if self.guarded and self.owner.get("owner") in {"PLANNED", "SERVO"} and not self.owner.get("busy"):
            self.ensure_oracle_current()

    def receive_contact(self, message):
        import json
        value = json.loads(message.data)
        if self.guarded:
            require(value["forbidden_contacts"] == 0, "Unexpected physical contact")
            require(value["left_force_N"]+value["right_force_N"] < 0.1, "M2B must not grip the tray")
            require(max(abs(p-0.04) for p in value["finger_positions_m"]) < 0.001, "Keep M2B fingers open")
            require(value["support_force_N"] > 0.5, "Tray lost support")
            require(all(math.isfinite(x) for x in value["tray_xyz"]+value["hand_xyz"]), "Invalid contact position")
            require(abs(value.get("external_force_x_N", 0.0)) <= 8.0001, "Conveyor force exceeded 8 N")
            if not self.allow_belt and self.reference:
                require(math.dist(value["tray_xyz"], self.reference["tray_xyz"]) < 0.003, "Tray moved during stationary check")
        self.contact = value

    def receive_oracle(self, message):
        require(message.oracle_only and message.header.frame_id == "world", "Use explicit world-frame oracle debug")
        sample = observation(message)
        if self.oracle is not None and (sample.episode, sample.stamp_ns) <= (self.oracle.episode, self.oracle.stamp_ns): return
        self.oracle = sample; self.oracle_received_wall = time.monotonic(); self.oracle_history.append(sample)

    def receive_prediction(self, message):
        self.prediction = message; self.prediction_history.append(copy.deepcopy(message))

    def send_belt(self): self.belt.publish(Float64(data=float(self.belt_speed)))

    def call(self, client, request, timeout=10.0):
        self.until(client.service_is_ready, timeout=timeout)
        return self.future(client.call_async(request), timeout=timeout)

    def startup(self):
        self.until(lambda: self.owner.get("ready") and self.owner.get("owner") == "HOLD" and not self.owner.get("busy")
                   and self.contact and self.oracle and self.prediction and self.prediction.valid
                   and all(n in self.q for n in NAMES+["panda_finger_joint1", "panda_finger_joint2"]), timeout=45.0)
        self.require_fresh_launch(); self.reference = self.contact.copy()
        require(self.owner.get("controller_streaming_verified") and self.owner.get("collision_scene_verified"), "Gate readiness is incomplete")
        request = GetParameters.Request(names=["allow_trajectory_execution", "moveit_manage_controllers", "moveit_controller_manager"])
        response = self.call(self.planner_params, request)
        require(len(response.values) == 3 and all(v.type == ParameterType.PARAMETER_BOOL and v.bool_value is False
                                               for v in response.values[:2]), "MoveIt must run with execution and controller management disabled")
        require(response.values[2].type == ParameterType.PARAMETER_STRING
                and response.values[2].string_value == "sorting_control/PlanningOnlyControllers",
                "MoveIt must use the planning-only controller plugin")
        self.scene_check()

    def scene_check(self):
        for client in self.scene_clients:
            request = GetPlanningScene.Request()
            request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            deadline = time.monotonic()+10.0
            while True:
                response = self.call(client, request)
                objects = {o.id: o for o in response.scene.world.collision_objects}
                if len(objects) >= 24 and "handle_bar" in objects:
                    obj = objects["handle_bar"]
                    require(obj.primitives and obj.primitive_poses, "Handle collision geometry is absent")
                    from prediction_core import rotation
                    p = np.array(xyz(obj.pose.position))+rotation(xyzw(obj.pose.orientation))@np.array(xyz(obj.primitive_poses[0].position))
                    if np.linalg.norm(p-np.asarray(self.oracle.position)) < 0.01: break
                require(time.monotonic() < deadline, "Planner/Servo collision scene did not receive the current tray")
                self.wait_sim(0.1)

    def hand_pose(self, stamp=None):
        at = RosTime(nanoseconds=stamp) if stamp is not None else RosTime()
        transform = self.buffer.lookup_transform("world", self.settings["hand_link"], at)
        require(stamp is not None or -20_000_000 <= self.now_ns()-ns(transform.header.stamp) < 150_000_000,
                "Hand TF is stale or future")
        return np.array(xyz(transform.transform.translation)), xyzw(transform.transform.rotation)

    def frame_check(self):
        stamp = self.q_stamp_ns; q = self.q.copy()
        self.until(lambda: self.buffer.can_transform("world", self.settings["hand_link"], RosTime(nanoseconds=stamp)))
        p, quat = self.hand_pose(stamp)
        request = GetPositionFK.Request(); request.header.frame_id = "world"
        request.fk_link_names = [self.settings["hand_link"]]; request.robot_state = robot_state(q)
        response = self.call(self.fk_client, request)
        require(response.error_code.val == 1 and len(response.pose_stamped) == 1, "MoveIt FK failed")
        fk = response.pose_stamped[0].pose
        tf_error = float(np.linalg.norm(p-np.array(xyz(fk.position))))
        tf_angle = angle_error(quat, xyzw(fk.orientation))
        # The physical truth comparison is made while the robot is stopped.
        truth = self.contact["hand_xyz"]; wxyz = self.contact["hand_quat_wxyz"]
        truth_angle = angle_error(quat, [wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        truth_error = float(np.linalg.norm(p-np.array(truth)))
        require(max(tf_error, truth_error) < self.settings["frame_position_tolerance_m"], "Hand frame position mismatch")
        require(max(tf_angle, truth_angle) < self.settings["frame_orientation_tolerance_rad"], "Hand frame rotation mismatch")
        return {"fk_tf_position_error_m": tf_error, "fk_tf_angle_error_rad": tf_angle,
                "mujoco_tf_position_error_m": truth_error, "mujoco_tf_angle_error_rad": truth_angle,
                "frame": "world", "link": self.settings["hand_link"], "joint_stamp_ns": stamp}

    def acquire_mode(self, owner, seconds=55.0, kind=0):
        request = SetArmOwner.Request(owner=owner, servo_command_type=int(kind))
        request.deadline = timestamp(self.now_ns()+round(seconds*1e9)) if owner != "HOLD" else Time()
        response = self.call(self.service, request)
        require(response.success and response.owner == owner, response.reason)
        self.until(lambda: self.owner.get("epoch") == response.epoch and self.owner.get("owner") == owner)
        if owner == "SERVO":
            self.input_epoch = response.epoch; self.input_deadline = ns(request.deadline)
        return response.epoch, request.deadline

    def send(self, epoch=None):
        if not self.input_enabled: return
        require(self.command_function is not None, "No Cartesian reference")
        p, quat = self.hand_pose()
        target, velocity, target_quat = self.command_function(self.now_ns())
        linear, angular = tracking_twist(target, velocity, target_quat, p, quat, self.settings)
        message = ArmServoCommand(); message.epoch = int(self.input_epoch)
        message.deadline = timestamp(self.input_deadline); message.command_type = ArmServoCommand.TWIST
        message.twist.header.stamp = self.node.get_clock().now().to_msg(); message.twist.header.frame_id = "world"
        message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z = map(float, linear)
        message.twist.twist.angular.x, message.twist.twist.angular.y, message.twist.twist.angular.z = map(float, angular)
        self.publisher.publish(message)

    def execute_plan(self, position, quaternion, folder, label, joint_goal=None):
        from rosidl_runtime_py.convert import message_to_ordereddict
        import json
        epoch, deadline = self.acquire_mode("PLANNED")
        self.wait_sim(0.2)
        self.ensure_oracle_current()
        begin = self.q.copy(); started = time.monotonic()
        response = self.call(self.planner, plan_request(begin, position, quaternion, self.settings, joint_goal), timeout=12.0)
        planning_wall = time.monotonic()-started
        (folder/(label+"-moveit-response.json")).write_text(json.dumps(message_to_ordereddict(response), indent=2)+"\n")
        result = response.motion_plan_response
        require(result.error_code.val == 1, f"MoveIt planning failed: error code {result.error_code.val}")
        trajectory = normalize_plan(result.trajectory.joint_trajectory)
        validate_trajectory(trajectory, self.now_ns(), ns(deadline))
        duration = ns(trajectory.points[-1].time_from_start)/1e9
        require(0 < duration <= self.settings["maximum_plan_duration_s"], "MoveIt plan duration outside M2B bound")
        require(max(abs(self.q[n]-trajectory.points[0].positions[i]) for i, n in enumerate(NAMES))
                < self.settings["maximum_plan_seed_error_rad"], "Measured state moved away from the plan seed")
        check = GetStateValidity.Request(group_name="panda_arm", robot_state=robot_state(self.q))
        require(self.call(self.validity, check).valid, "Actual plan start is in collision")
        goal = ExecuteArmTrajectory.Goal(epoch=int(epoch), deadline=deadline, trajectory=trajectory)
        handle = self.future(self.action.send_goal_async(goal)); require(handle.accepted, "Gate rejected MoveIt plan")
        executed = self.future(handle.get_result_async(), timeout=45.0)
        require(executed.status == GoalStatus.STATUS_SUCCEEDED and executed.result.success, executed.result.reason)
        self.wait_sim(0.3)
        p, quat = self.hand_pose()
        error = float(np.linalg.norm(p-np.asarray(position)))
        require(error < self.settings["planned_position_tolerance_m"], "Planned physical endpoint exceeded its compliance allowance")
        require(angle_error(quat, quaternion) < 0.05, "Planned endpoint rotation mismatch")
        self.acquire_mode("HOLD")
        return {"pipeline": "ompl", "planner": "RRTConnectkConfigDefault", "points": len(trajectory.points),
                "duration_sim_s": duration, "planning_wall_s": planning_wall,
                "reported_planning_wall_s": float(result.planning_time), "physical_position_error_m": error,
                "execution_path": "/sorting/arm/execute_trajectory", "goal_xyz": list(map(float, position))}

    def refine_pose(self, position, quaternion):
        self.acquire_mode("SERVO", seconds=20.0, kind=1)
        self.command_function = lambda now: (position, [0.0]*3, quaternion)
        self.input_enabled = True
        start = self.now_ns()
        self.until(lambda: self.now_ns()-start > 300_000_000 and
                   np.linalg.norm(self.hand_pose()[0]-np.asarray(position)) < self.settings["pregrasp_position_tolerance_m"]
                   and angle_error(self.hand_pose()[1], quaternion) < self.settings["pregrasp_orientation_tolerance_rad"], timeout=10.0)
        self.wait_sim(0.2)
        p, quat = self.hand_pose()
        require(self.owner["owner"] == "SERVO", "Servo ended during pregrasp refinement")
        measured = {"position_error_m": float(np.linalg.norm(p-np.asarray(position))), "orientation_error_rad": angle_error(quat, quaternion)}
        self.input_enabled = False; self.acquire_mode("HOLD")
        return measured
