#!/usr/bin/env python3
"""Check planned/Servo ownership, small joint motion, rejection and bounded hold."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint
from sorting_interfaces.action import ExecuteArmTrajectory
from sorting_interfaces.msg import ArmServoCommand
from sorting_interfaces.srv import SetArmOwner
from ownership_core import NAMES, require


def timestamp(value):
    return Time(sec=int(value//1_000_000_000), nanosec=int(value%1_000_000_000))


class Probe:
    def __init__(self, node):
        self.node = node
        self.owner, self.q, self.v, self.contact = {}, {}, {}, {}
        self.events = []
        self.guarded = True
        self.reference = None
        self.previous_q = None
        self.maximum_joint_step = 0.0
        self.maximum_joint_speed = 0.0
        self.input_enabled = False
        self.input_epoch = 0
        self.input_deadline = 0
        self.speed = 0.0
        node.create_subscription(String, "/sorting/arm/owner_state", self.receive_owner, 10)
        node.create_subscription(String, "/sorting/arm/events", self.receive_event, 100)
        node.create_subscription(JointState, "/joint_states", self.receive_joints,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        node.create_subscription(String, "/sorting/contact_state", self.receive_contact, 10)
        self.service = node.create_client(SetArmOwner, "/sorting/arm/set_owner")
        self.action = ActionClient(node, ExecuteArmTrajectory, "/sorting/arm/execute_trajectory")
        self.publisher = node.create_publisher(ArmServoCommand, "/sorting/arm/servo_command", 1)
        node.create_timer(0.02, self.send, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def now_ns(self):
        return self.node.get_clock().now().nanoseconds

    def require_fresh_launch(self):
        stats = self.owner["stats"]
        require(all(stats[name] == 0 for name in ("planned_accepted", "servo_input_accepted", "servo_output_forwarded")),
                "This launch already ran an ownership probe. Stop Terminal 1, restart the launch, then run one probe")

    def receive_owner(self, message):
        self.owner = json.loads(message.data)
        if self.guarded:
            require(self.owner["owner"] != "FAULT", "Command gate entered FAULT: "+self.owner.get("reason", ""))
            require(self.owner["stats"]["overlap_violations"] == 0, "Command source overlap")

    def receive_event(self, message):
        self.events.append(json.loads(message.data))

    def receive_joints(self, message):
        self.q = dict(zip(message.name, message.position))
        self.v = dict(zip(message.name, message.velocity))
        if not self.guarded or not all(n in self.q and n in self.v for n in NAMES):
            return
        values = [self.q[n] for n in NAMES]
        require(all(math.isfinite(v) for v in values+[self.v[n] for n in NAMES]), "Invalid joint feedback")
        if self.previous_q is not None:
            step = max(abs(a-b) for a, b in zip(values, self.previous_q))
            self.maximum_joint_step = max(self.maximum_joint_step, step)
            require(step < 0.02, "Joint feedback jumped by more than 0.02 rad")
        self.previous_q = values
        self.maximum_joint_speed = max(self.maximum_joint_speed, max(abs(self.v[n]) for n in NAMES))
        require(self.maximum_joint_speed < 0.25, "Small-motion test exceeded 0.25 rad/s")

    def receive_contact(self, message):
        value = json.loads(message.data)
        if self.guarded:
            require(value["forbidden_contacts"] == 0, "Unexpected physical contact")
            require(value["left_force_N"]+value["right_force_N"] < 0.1, "Probe must not grip the tray")
            require(max(abs(v-0.04) for v in value["finger_positions_m"]) < 0.001, "Probe fingers must stay open")
            require(abs(value.get("external_force_x_N", 1.0)) < 1e-9, "Conveyor force must be disabled")
            if self.reference:
                require(math.dist(value["tray_xyz"], self.reference["tray_xyz"]) < 0.003, "Tray moved during ownership check")
                require(abs(value["support_force_N"]-self.reference["support_force_N"]) < 0.5, "Tray lost its support")
        self.contact = value

    def until(self, predicate, timeout=30.0):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if predicate():
                return
        raise TimeoutError("Ownership probe timed out")

    def future(self, future, timeout=30.0):
        self.until(future.done, timeout=timeout)
        return future.result()

    def safe_hold(self, timeout=12.0):
        """Wait for an existing stop before requesting another owner change."""
        self.input_enabled = False
        self.guarded = False
        end = time.monotonic()+timeout
        def remaining():
            left = end-time.monotonic()
            require(left > 0, "Failure cleanup could not confirm HOLD within the wall timeout")
            return left
        while True:
            self.until(lambda: self.owner and not self.owner.get("busy")
                       and self.owner.get("owner") != "SWITCHING", timeout=remaining())
            require(self.owner.get("owner") != "FAULT",
                    "Gate is in FAULT; retain its failure state and restart the launch")
            if self.owner.get("owner") == "HOLD":
                return
            request = SetArmOwner.Request(); request.owner = "HOLD"
            response = self.future(self.service.call_async(request), timeout=remaining())
            if response.success and response.owner == "HOLD":
                self.until(lambda: self.owner.get("owner") == "HOLD" and not self.owner.get("busy")
                           and self.owner.get("epoch") == response.epoch, timeout=remaining())
                return
            # The watchdog can reserve its stop between the snapshot and call.
            require(response.reason == "An owner transition is already running", response.reason)

    def wait_sim(self, seconds):
        start = self.now_ns()
        self.until(lambda: self.now_ns()-start >= round(seconds*1e9))

    def acquire(self, owner, seconds=20.0):
        request = SetArmOwner.Request(); request.owner = owner; request.servo_command_type = 0
        request.deadline = timestamp(self.now_ns()+round(seconds*1e9)) if owner != "HOLD" else Time()
        response = self.future(self.service.call_async(request))
        require(response.success and response.owner == owner, response.reason)
        self.until(lambda: self.owner.get("epoch") == response.epoch and self.owner.get("owner") == owner)
        if owner == "SERVO":
            self.input_epoch = response.epoch
            self.input_deadline = request.deadline.sec*1_000_000_000+request.deadline.nanosec
        return response.epoch, request.deadline

    def trajectory(self, epoch, deadline, positions, duration=2.0):
        goal = ExecuteArmTrajectory.Goal(); goal.epoch = int(epoch); goal.deadline = deadline
        goal.trajectory.joint_names = NAMES
        point = JointTrajectoryPoint(); point.positions = [float(v) for v in positions]; point.velocities = [0.0]*7
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = round((duration-int(duration))*1e9)
        goal.trajectory.points = [point]
        return self.future(self.action.send_goal_async(goal))

    def send(self, epoch=None):
        if not self.input_enabled and epoch is None:
            return
        message = ArmServoCommand(); message.epoch = int(self.input_epoch if epoch is None else epoch)
        message.deadline = timestamp(self.input_deadline)
        message.command_type = ArmServoCommand.JOINT_JOG
        message.joint_jog.header.stamp = self.node.get_clock().now().to_msg()
        message.joint_jog.header.frame_id = "world"
        message.joint_jog.joint_names = [NAMES[0]]
        message.joint_jog.velocities = [float(self.speed)]
        self.publisher.publish(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    folder = args.output_dir or Path("results")/f"M2A-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True, exist_ok=False)
    share = Path(get_package_share_directory("sorting_control"))
    files = [share/"config"/n for n in ("ownership.yaml", "servo.yaml", "fixtures.yaml", "ownership_controllers.yaml")]
    files += [share/"launch/panda_ownership.launch.py",
              Path(get_package_share_directory("sorting_simulation"))/"launch/panda_bins.launch.py"]
    source = Path(__file__).resolve().parent
    files += [source/n for n in ("arm_command_gate.py", "ownership_core.py", "test_arm_ownership.py")]
    manifest = {"started_at": datetime.now().astimezone().isoformat(),
                "protocol": "Stationary planned/Servo ownership, small joint jog, rejection, cancellation and hold",
                "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                "simulation_time_deadlines": True, "moving_grasp": False}
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    report = {"status": "FAIL", "acceptance_passed": False, "mode": "stationary_ownership_probe",
              "moving_grasp": False, "checks": {}}
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("test_arm_ownership", parameter_overrides=[Parameter("use_sim_time", value=True)])
    probe = Probe(node)
    try:
        probe.until(lambda: probe.owner.get("ready") and probe.owner.get("owner") == "HOLD"
                    and probe.contact and all(n in probe.q for n in NAMES)
                    and probe.service.service_is_ready() and probe.action.server_is_ready())
        probe.reference = probe.contact.copy()
        probe.require_fresh_launch()
        require(probe.owner.get("collision_scene_verified") is True, "Servo collision fixtures were not verified")
        require(probe.owner.get("controller_streaming_verified") is True,
                "Loaded arm controller settings were not verified for Servo")
        initial = [probe.q[n] for n in NAMES]
        report["checks"]["initial_state"] = {"joint_positions": initial, "contact": probe.reference}
        epoch, deadline = probe.acquire("PLANNED")
        target = initial.copy(); target[0] += 0.03
        handle = probe.trajectory(epoch, deadline, target)
        require(handle.accepted, "Planned probe goal was rejected")
        result = probe.future(handle.get_result_async())
        report["planned_result"] = {"action_status": int(result.status), "success": result.result.success,
                                    "controller_error_code": result.result.controller_error_code,
                                    "reason": result.result.reason}
        require(result.status == GoalStatus.STATUS_SUCCEEDED and result.result.success, result.result.reason)
        probe.wait_sim(0.3)
        require(abs(probe.q[NAMES[0]]-target[0]) < 0.01, "Planned probe did not reach joint 1 target")
        report["checks"]["planned_motion"] = "PASS"

        rejected = probe.owner["stats"]["servo_input_rejected"]
        probe.input_deadline = probe.now_ns()+10_000_000_000
        probe.send(epoch=epoch)
        probe.until(lambda: probe.owner["stats"]["servo_input_rejected"] > rejected)
        report["checks"]["servo_rejected_while_planned"] = "PASS"

        epoch, deadline = probe.acquire("PLANNED")
        target = [probe.q[n] for n in NAMES]; target[0] -= 0.04
        active = probe.trajectory(epoch, deadline, target, duration=4.0)
        require(active.accepted, "Cancellation probe goal was rejected")
        probe.wait_sim(0.4)
        old_epoch = epoch
        epoch, _ = probe.acquire("SERVO")
        cancelled = probe.future(active.get_result_async())
        require(not cancelled.result.success and cancelled.status == GoalStatus.STATUS_ABORTED,
                "Ownership handoff did not terminate the old planned goal")
        probe.speed = 0.04; probe.input_enabled = True
        before = probe.q[NAMES[0]]
        forwarded_before = probe.owner["stats"]["servo_output_forwarded"]
        probe.wait_sim(0.8)
        report["servo_motion"] = {"joint1_before_rad": before, "joint1_after_rad": probe.q[NAMES[0]],
                                  "joint1_change_rad": probe.q[NAMES[0]]-before,
                                  "output_forwarded_during_motion": probe.owner["stats"]["servo_output_forwarded"]-forwarded_before,
                                  "owner": probe.owner["owner"], "reason": probe.owner.get("reason"),
                                  "servo_status": probe.owner.get("servo_status"),
                                  "controller_parameters": probe.owner.get("controller_parameters")}
        require(probe.owner["owner"] == "SERVO",
                "Servo ownership ended during motion: "+probe.owner.get("reason", "unknown"))
        require(probe.q[NAMES[0]]-before > 0.005, "Installed Servo did not move joint 1; see servo_motion in the report")
        probe.speed = 0.0; probe.wait_sim(0.3)
        require(probe.owner["stats"]["servo_output_forwarded"]-forwarded_before > 5,
                "Not enough actual Servo output was forwarded during this probe")
        report["checks"]["cancel_then_servo_motion"] = "PASS"

        rejected_goal = probe.trajectory(epoch, timestamp(probe.input_deadline), [probe.q[n] for n in NAMES])
        require(not rejected_goal.accepted, "Planned goal was accepted while Servo owned the arm")
        report["checks"]["planned_rejected_while_servo"] = "PASS"
        rejected = probe.owner["stats"]["servo_input_rejected"]
        probe.send(epoch=old_epoch)
        probe.until(lambda: probe.owner["stats"]["servo_input_rejected"] > rejected)
        require(probe.owner["owner"] == "SERVO",
                "Servo owner ended during the old-epoch check: "+probe.owner.get("reason", "unknown")+
                "; see last_stop for the actual stop trigger")
        report["checks"]["old_epoch_rejected"] = "PASS"

        probe.input_enabled = False
        probe.until(lambda: probe.owner.get("owner") == "HOLD" and not probe.owner.get("busy"))
        require(probe.owner["reason"] == "servo_input_lost", "Missing input did not trigger bounded hold")
        count = probe.owner["stats"]["servo_output_forwarded"]
        probe.wait_sim(0.3)
        require(probe.owner["stats"]["servo_output_forwarded"] == count, "Servo output continued in HOLD")
        report["checks"]["lost_source_hold"] = "PASS"

        probe.acquire("SERVO", seconds=2.0)
        probe.input_enabled = True; probe.speed = 0.0
        probe.until(lambda: probe.owner.get("owner") == "HOLD" and not probe.owner.get("busy"))
        probe.input_enabled = False
        require(probe.owner["reason"] == "owner_deadline", "Owner deadline did not trigger hold")
        report["checks"]["deadline_hold"] = "PASS"

        epoch, deadline = probe.acquire("PLANNED")
        target = [probe.q[n] for n in NAMES]; target[0] += 0.02
        active = probe.trajectory(epoch, deadline, target, duration=3.0)
        require(active.accepted, "Explicit cancel probe was rejected")
        probe.wait_sim(0.3)
        cancellation = probe.future(active.cancel_goal_async())
        require(bool(cancellation.goals_canceling), "Proxy goal cancellation was rejected")
        result = probe.future(active.get_result_async())
        require(result.status == GoalStatus.STATUS_CANCELED and not result.result.success, "Goal did not report cancellation")
        probe.until(lambda: probe.owner.get("owner") == "HOLD" and not probe.owner.get("busy"))
        report["checks"]["explicit_cancel_hold"] = "PASS"

        epoch, deadline = probe.acquire("PLANNED")
        home = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
        active = probe.trajectory(epoch, deadline, home)
        require(active.accepted, "Home goal was rejected")
        result = probe.future(active.get_result_async())
        require(result.result.success, result.result.reason)
        probe.acquire("HOLD")
        require(max(abs(probe.v[n]) for n in NAMES) < 0.01, "Final arm did not stop")
        require(probe.owner["stats"]["overlap_violations"] == 0, "Output overlap was recorded")
        require(any(e["event"] == "backend_cancel_terminal" and e.get("status") == GoalStatus.STATUS_CANCELED
                    for e in probe.events), "No terminal backend cancellation was recorded")
        report["checks"]["return_home_and_hold"] = "PASS"
        report["final_owner_state"] = probe.owner
        report["maximum_joint_feedback_step_rad"] = probe.maximum_joint_step
        report["maximum_joint_speed_rad_s"] = probe.maximum_joint_speed
        required_checks = ["planned_motion", "servo_rejected_while_planned", "cancel_then_servo_motion",
                           "planned_rejected_while_servo", "old_epoch_rejected", "lost_source_hold",
                           "deadline_hold", "explicit_cancel_hold", "return_home_and_hold"]
        require(all(report["checks"].get(name) == "PASS" for name in required_checks), "Ownership run is incomplete")
        report["passed_checks"] = len(required_checks)
        report["acceptance_passed"] = True
        report["status"] = "PASS"
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = str(error) or "Interrupted by user"
        report["owner_at_failure"] = probe.owner.copy()
        report["feedback_at_probe_failure"] = probe.owner.get("feedback_health")
        probe.input_enabled = False; probe.guarded = False
        try:
            probe.safe_hold()
            report["failure_hold"] = "PASS"
        except Exception as stop_error:
            report["failure_hold_error"] = str(stop_error)
    finally:
        report["output_directory"] = str(folder.resolve())
        report["last_owner_state"] = probe.owner
        report["last_contact_state"] = probe.contact
        (folder/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        (folder/"events.json").write_text(json.dumps(probe.events, indent=2, allow_nan=False)+"\n")
        node.destroy_node(); rclpy.shutdown()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": main()
