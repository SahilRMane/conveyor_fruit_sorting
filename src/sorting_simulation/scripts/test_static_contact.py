#!/usr/bin/env python3
"""Run one static approach/contact/release test on a fresh Panda contact launch."""
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
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/M1C2-ros-contact.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    require(plan.get("status") == "PASS" and plan.get("schema_version") == 1,
            "Generate a passing plan before the ROS test")
    rclpy.init()
    node = rclpy.create_node("static_contact_test")
    arm_names = [f"panda_joint{i}" for i in range(1, 8)]
    latest, joints, history = {}, {}, []
    active_goal = None
    reference = None
    received_at = None
    guarded = True
    report = {"status": "FAIL", "checks": {}}

    def receive_state(message):
        nonlocal received_at
        value = json.loads(message.data)
        if guarded and latest:
            require(value["sim_time"] >= latest["sim_time"] - 1e-6,
                    "Simulation time moved backwards during the test")
        latest.clear()
        latest.update(value)
        received_at = time.monotonic()
        history.append(value)

    def receive_joints(message):
        joints.update(zip(message.name, message.position))

    node.create_subscription(String, "/sorting/contact_state", receive_state, 10)
    node.create_subscription(JointState, "/joint_states", receive_joints, qos_profile_sensor_data)
    arm = ActionClient(node, FollowJointTrajectory, "/panda_arm_controller/follow_joint_trajectory")
    hand = ActionClient(node, GripperCommand, "/panda_hand_controller/gripper_cmd")

    def check_state():
        if not guarded or not latest:
            return
        if history:
            # Inspect each new sample, including samples received during an action wait.
            require(all(s["forbidden_contacts"] == 0 for s in history[-2:]),
                    "Unexpected robot contact with tray or support")
        require(time.monotonic() - received_at < 1.0, "Contact data stopped; simulation may be paused")
        require(latest["forbidden_contacts"] == 0, "Unexpected robot contact with tray or support")
        require(max(latest["left_force_N"], latest["right_force_N"]) < 15.0,
                "Finger contact force exceeds 15 N")
        require(abs(latest["gripper_force_limit_N"] - 20.0) < 1e-6,
                "The loaded scene needs the 20 N gripper actuator limit")
        numeric = latest["tray_xyz"] + latest["hand_xyz"] + latest["tray_quat_wxyz"]
        numeric += latest["hand_quat_wxyz"] + latest["finger_positions_m"]
        numeric += [latest["left_force_N"], latest["right_force_N"], latest["support_force_N"], latest["sim_time"]]
        require(all(math.isfinite(v) for v in numeric), "Non-finite scene state")
        if reference is not None:
            require(distance(latest["tray_xyz"], reference["tray_xyz"]) < 0.003,
                    "Tray moved more than 3 mm")
            dot = abs(sum(a*b for a,b in zip(latest["tray_quat_wxyz"], reference["tray_quat_wxyz"])))
            require(dot > math.cos(math.radians(3)/2), "Tray rotated more than 3 degrees")
            require(latest["support_force_N"] > 0.5, "Tray lost support contact")

    def until(predicate, timeout=20.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)
            check_state()
            if predicate():
                return
        raise TimeoutError("ROS wait timed out")

    def future_result(future, timeout=20.0):
        until(future.done, timeout)
        return future.result()

    def wait_sim(seconds):
        start = latest["sim_time"]
        until(lambda: latest["sim_time"] >= start + seconds)

    def action(client, goal):
        nonlocal active_goal
        active_goal = future_result(client.send_goal_async(goal))
        require(active_goal.accepted, "Controller rejected the goal")
        result = future_result(active_goal.get_result_async())
        active_goal = None
        require(result.status == GoalStatus.STATUS_SUCCEEDED, f"Action failed with status {result.status}")
        return result.result

    def grip(position, expect_contact=False):
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 20.0  # The MJCF actuator supplies the actual force limit.
        result = action(hand, goal)
        if expect_contact:
            require(result.stalled and not result.reached_goal, "Expected fingers to stop on the handle")
        else:
            require(result.reached_goal and not result.stalled, "Gripper did not reach its free-space goal")
        return {"position": result.position, "stalled": result.stalled, "reached_goal": result.reached_goal}

    try:
        until(lambda: latest and all(name in joints for name in arm_names)
              and arm.server_is_ready() and hand.server_is_ready(), 30.0)
        home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
        require(max(abs(joints[n]-q) for n,q in zip(arm_names, home)) < 0.04,
                "Start a fresh contact launch. The arm must be at home")
        grip(0.04)
        wait_sim(1.0)
        require(distance(latest["tray_xyz"], plan["initial_state"]["tray_xyz"]) < 0.002,
                "Tray pose differs from the validated plan")
        require(latest["left_force_N"] + latest["right_force_N"] < 0.1,
                "Finger contact exists at the start")
        reference = latest.copy()
        report["checks"]["initial_state"] = reference

        for phase in plan["phases"]:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = arm_names
            point = JointTrajectoryPoint()
            point.positions = phase["positions"]
            point.velocities = [0.0] * 7
            point.time_from_start.sec = int(plan["duration_s"])
            goal.trajectory.points = [point]
            result = action(arm, goal)
            require(result.error_code == FollowJointTrajectory.Result.SUCCESSFUL, result.error_string)
            wait_sim(plan["settle_s"])
            require(distance(latest["hand_xyz"], phase["hand_target_xyz"]) < 0.003,
                    "Hand target error exceeds 3 mm")
            target_quat = [0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0]
            dot = abs(sum(a*b for a,b in zip(latest["hand_quat_wxyz"], target_quat)))
            require(dot > math.cos(math.radians(1)/2), "Hand orientation error exceeds 1 degree")
            require(latest["left_force_N"] + latest["right_force_N"] < 0.1,
                    "Handle contact occurred before closure")
            report["checks"][phase["name"]] = latest.copy()

        grip(0.025)
        grip(0.015)
        report["checks"]["close_action"] = grip(0.0, expect_contact=True)
        history.clear()
        wait_sim(1.0)
        require(len(history) >= 20, "Not enough contact samples")
        require(all(s["left_force_N"] > 0.2 and s["right_force_N"] > 0.2 for s in history),
                "Both fingers must maintain handle contact for one second")
        report["checks"]["contact_state"] = latest.copy()
        report["checks"]["minimum_left_force_N"] = min(s["left_force_N"] for s in history)
        report["checks"]["minimum_right_force_N"] = min(s["right_force_N"] for s in history)
        grip(0.04)
        wait_sim(1.0)
        require(latest["left_force_N"] + latest["right_force_N"] < 0.1, "Handle contact did not release")
        report["checks"]["released_state"] = latest.copy()
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
        # Cancel motion and release the grip. Do not move the arm after a failure.
        guarded = False
        try:
            if active_goal is not None:
                future_result(active_goal.cancel_goal_async(), 5.0)
            if hand.server_is_ready():
                grip(0.04)
        except Exception as cleanup_error:
            report["cleanup_error"] = str(cleanup_error)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
