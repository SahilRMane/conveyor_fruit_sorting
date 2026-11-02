#!/usr/bin/env python3
"""Run one physical grip, lift, hold, lower and release cycle."""
import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from mujoco_ros2_control_msgs.srv import SetPause
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


class LiftTest:
    def __init__(self, node, plan):
        self.node, self.plan = node, plan
        self.names = [f"panda_joint{i}" for i in range(1, 8)]
        self.state, self.joints, self.history = {}, {}, []
        self.received_at = None
        self.reference = None
        self.grasp = None
        self.active_goal = None
        self.stage = "startup"
        self.guarded = True
        self.maximum_slip = 0.0
        self.maximum_placement_force = 0.0
        node.create_subscription(String, "/sorting/contact_state", self.receive, 10)
        node.create_subscription(JointState, "/joint_states", self.receive_joints, qos_profile_sensor_data)
        self.arm = ActionClient(node, FollowJointTrajectory, "/panda_arm_controller/follow_joint_trajectory")
        self.hand = ActionClient(node, GripperCommand, "/panda_hand_controller/gripper_cmd")

    def receive_joints(self, message):
        self.joints.update(zip(message.name, message.position))

    def receive(self, message):
        value = json.loads(message.data)
        if self.guarded and self.state:
            require(value["sim_time"] >= self.state["sim_time"] - 1e-6, "Simulation time moved backwards")
        self.check(value)
        self.state = value
        self.history.append(value)
        self.received_at = time.monotonic()

    def check(self, state):
        if not self.guarded:
            return
        values = state["tray_xyz"] + state["hand_xyz"] + state["tray_quat_wxyz"]
        values += state["hand_quat_wxyz"] + state["finger_positions_m"]
        values += [state["sim_time"], state["left_force_N"], state["right_force_N"], state["support_force_N"]]
        require(all(math.isfinite(v) for v in values), "Non-finite simulation state")
        require(state["forbidden_contacts"] == 0, "Unexpected robot contact")
        require(max(state["left_force_N"], state["right_force_N"]) < 15.0, "Finger force exceeds 15 N")
        require(abs(state["gripper_force_limit_N"] - 20.0) < 1e-6, "Expected 20 N gripper actuator limit")
        require(state.get("friction_cone") == 1 and state.get("friction_impedance_ratio") == 100,
                "Use the updated monitor and panda_lift.launch.py with scene_lift.xml")
        if self.reference is None:
            return
        reference = self.reference
        rise = state["tray_xyz"][2] - reference["tray_xyz"][2]
        require(distance(state["tray_xyz"][:2], reference["tray_xyz"][:2]) < 0.003,
                "Tray horizontal drift exceeds 3 mm")
        require(-0.003 <= rise <= self.plan["lift_height_m"] + 0.01, "Tray height outside lift bounds")
        dot = abs(sum(a*b for a,b in zip(state["tray_quat_wxyz"], reference["tray_quat_wxyz"])))
        require(dot > math.cos(math.radians(3)/2), "Tray tilt exceeds 3 degrees")
        if self.stage in {"approach", "grip", "release"}:
            require(distance(state["tray_xyz"], reference["tray_xyz"]) < 0.003
                    and state["support_force_N"] > 0.5, "Tray must be on the support")
        if self.stage in {"lower", "release"}:
            self.maximum_placement_force = max(self.maximum_placement_force, state["support_force_N"])
            require(state["support_force_N"] < 2*reference["support_force_N"],
                    "Placement support force exceeds twice tray weight")
        if self.grasp is not None and self.stage in {"lift", "hold", "lower"}:
            require(state["left_force_N"] > 0.2 and state["right_force_N"] > 0.2,
                    "Finger-to-handle contact was lost")
            relative = [h-t for h,t in zip(state["hand_xyz"], state["tray_xyz"])]
            baseline = [h-t for h,t in zip(self.grasp["hand_xyz"], self.grasp["tray_xyz"])]
            slip = distance(relative, baseline)
            self.maximum_slip = max(self.maximum_slip, slip)
            require(slip < 0.003, "Tray slip relative to the hand exceeds 3 mm")
            if rise > 0.005:
                require(state["support_force_N"] < 0.1, "Lifted tray still has support force")
        if self.stage == "hold":
            require(rise > self.plan["lift_height_m"] - 0.005 and state["support_force_N"] < 0.1,
                    "Tray did not remain lifted")

    def until(self, predicate, timeout=30.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if self.guarded and self.received_at is not None:
                require(time.monotonic() - self.received_at < 1.0, "Contact data stopped; simulation may be paused")
            if predicate():
                return
        raise TimeoutError("ROS wait timed out")

    def wait_future(self, future, timeout=30.0):
        self.until(future.done, timeout)
        return future.result()

    def wait_sim(self, seconds):
        start = self.state["sim_time"]
        self.until(lambda: self.state["sim_time"] >= start + seconds)

    def action(self, client, goal):
        self.active_goal = self.wait_future(client.send_goal_async(goal))
        require(self.active_goal.accepted, "Controller rejected the goal")
        result = self.wait_future(self.active_goal.get_result_async())
        self.active_goal = None
        require(result.status == GoalStatus.STATUS_SUCCEEDED, f"Action failed with status {result.status}")
        return result.result

    def grip(self, position, contact=False):
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 20.0
        result = self.action(self.hand, goal)
        if contact:
            require(result.stalled and not result.reached_goal, "Expected fingers to stop on the handle")
        else:
            require(result.reached_goal and not result.stalled, "Gripper did not reach its free-space goal")
        return {"position": result.position, "stalled": result.stalled, "reached_goal": result.reached_goal}

    def move(self, phase):
        self.stage = phase["stage"]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.names
        point = JointTrajectoryPoint()
        point.positions = phase["positions"]
        point.velocities = [0.0]*7
        duration = self.plan["duration_s"]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = round((duration-int(duration))*1e9)
        goal.trajectory.points = [point]
        result = self.action(self.arm, goal)
        require(result.error_code == FollowJointTrajectory.Result.SUCCESSFUL, result.error_string)
        self.wait_sim(self.plan["settle_s"])
        require(distance(self.state["hand_xyz"], phase["hand_target_xyz"]) < 0.003,
                "Hand position error exceeds 3 mm")
        quaternion = [0, math.sqrt(0.5), math.sqrt(0.5), 0]
        dot = abs(sum(a*b for a,b in zip(self.state["hand_quat_wxyz"], quaternion)))
        require(dot > math.cos(math.radians(1)/2), "Hand orientation error exceeds 1 degree")
        if self.stage == "approach":
            require(self.state["left_force_N"] + self.state["right_force_N"] < 0.1,
                    "Handle contact occurred before closure")
        print(f"{phase['name']}: tray rise={self.state['tray_xyz'][2]-self.reference['tray_xyz'][2]:.4f} m")
        return self.state.copy()

    def confirm_controller(self):
        client = self.node.create_client(GetParameters, "/panda_arm_controller/get_parameters")
        self.until(client.service_is_ready)
        request = GetParameters.Request()
        request.names = ["interpolate_from_desired_state"]
        response = self.wait_future(client.call_async(request))
        require(len(response.values) == 1 and response.values[0].type == 1
                and response.values[0].bool_value, "Arm controller must interpolate from its last desired state")

    def run(self):
        report = {"status": "FAIL", "checks": {}}
        try:
            self.until(lambda: self.state and all(n in self.joints for n in self.names)
                       and self.arm.server_is_ready() and self.hand.server_is_ready())
            home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
            require(max(abs(self.joints[n]-q) for n,q in zip(self.names, home)) < 0.04,
                    "Start a fresh lift launch with the arm at home")
            self.confirm_controller()
            self.grip(0.04)
            self.wait_sim(1.0)
            require(distance(self.state["tray_xyz"], self.plan["initial_state"]["tray_xyz"]) < 0.002,
                    "Tray pose differs from the validated plan")
            require(abs(self.state["tray_mass_kg"]-self.plan["initial_state"]["tray_mass_kg"]) < 1e-6,
                    "Tray mass differs from the validated plan")
            self.reference = self.state.copy()
            require(self.reference["support_force_N"] > 0.5, "Tray must start on the support")
            report["checks"]["initial_state"] = self.reference
            for phase in self.plan["phases"]:
                if phase["stage"] == "approach":
                    report["checks"][phase["name"]] = self.move(phase)
            self.stage = "grip"
            self.grip(0.025)
            self.grip(0.015)
            report["checks"]["close_action"] = self.grip(0.0, contact=True)
            self.wait_sim(1.0)
            require(self.state["left_force_N"] > 0.2 and self.state["right_force_N"] > 0.2,
                    "Both fingers must contact the handle before lift")
            self.grasp = self.state.copy()
            report["checks"]["grasp_state"] = self.grasp
            for phase in self.plan["phases"]:
                if phase["stage"] == "lift":
                    report["checks"][phase["name"]] = self.move(phase)
            self.stage = "hold"
            self.history.clear()
            self.wait_sim(self.plan["hold_s"])
            require(len(self.history) >= 50, "Not enough hold samples")
            heights = [s["tray_xyz"][2]-self.reference["tray_xyz"][2] for s in self.history]
            require(max(heights)-min(heights) < 0.002, "Tray height drift exceeds 2 mm during hold")
            report["checks"]["held_state"] = self.state.copy()
            report["hold_metrics"] = {
                "minimum_rise_m": min(heights), "maximum_rise_m": max(heights),
                "maximum_support_force_N": max(s["support_force_N"] for s in self.history),
                "minimum_left_force_N": min(s["left_force_N"] for s in self.history),
                "minimum_right_force_N": min(s["right_force_N"] for s in self.history),
            }
            for phase in self.plan["phases"]:
                if phase["stage"] == "lower":
                    report["checks"][phase["name"]] = self.move(phase)
            require(abs(self.state["tray_xyz"][2]-self.reference["tray_xyz"][2]) < 0.002
                    and self.state["support_force_N"] > 0.8*self.reference["support_force_N"],
                    "Tray must return to the support before release")
            report["checks"]["placed_state"] = self.state.copy()
            self.stage = "release"
            self.grip(0.04)
            self.wait_sim(1.0)
            require(self.state["left_force_N"] + self.state["right_force_N"] < 0.1,
                    "Finger contact did not release")
            require(abs(self.state["support_force_N"]-self.reference["support_force_N"]) < 0.5,
                    "Support force must balance tray weight after release")
            report["checks"]["released_state"] = self.state.copy()
            report["maximum_relative_slip_m"] = self.maximum_slip
            report["maximum_placement_support_force_N"] = self.maximum_placement_force
            report["status"] = "PASS"
        except Exception as error:
            report["error"] = str(error)
            report["last_state"] = self.state.copy()
            report["failure_handling"] = self.stop_after_failure()
        return report

    def stop_after_failure(self):
        # Keep the existing grip. Never open automatically with a tray in the air.
        self.guarded = False
        result = {"automatic_open_command": False, "paused": False}
        try:
            names = [name for name, types in self.node.get_service_names_and_types()
                     if "mujoco_ros2_control_msgs/srv/SetPause" in types]
            require(len(names) == 1, "Cannot identify one simulation pause service; pause the viewer manually")
            client = self.node.create_client(SetPause, names[0])
            self.until(client.service_is_ready, 5.0)
            request = SetPause.Request()
            request.paused = True
            response = self.wait_future(client.call_async(request), 5.0)
            require(response.success, response.message)
            result["paused"] = True
        except Exception as error:
            result["pause_error"] = str(error)
        try:
            if self.active_goal is not None:
                response = self.wait_future(self.active_goal.cancel_goal_async(), 5.0)
                result["cancel_requested"] = bool(response.goals_canceling)
        except Exception as error:
            result["cancel_error"] = str(error)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/M1C3-ros-lift.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    require(plan.get("status") == "PASS" and plan.get("schema_version") == 3,
            "Generate a passing M1C3 lift plan first")
    require(plan.get("interpolate_from_desired_state") is True, "Plan requires continuous desired-state interpolation")
    rclpy.init()
    node = rclpy.create_node("static_lift_test")
    try:
        report = LiftTest(node, plan).run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
