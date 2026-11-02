#!/usr/bin/env python3
"""Run after building/sourcing the workspace; launches an isolated headless test."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from controller_manager_msgs.srv import ListControllers, ListHardwareInterfaces
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, default=173)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    report = {"status": "FAIL", "ros_domain_id": args.domain_id, "checks": {}}
    with tempfile.TemporaryDirectory(prefix="panda_ros_test_") as temp:
        env = os.environ.copy()
        env["ROS_LOG_DIR"] = temp
        log_path = Path(temp) / "launch.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                ["ros2", "launch", "sorting_description", "panda_mujoco.launch.py", "headless:=true"],
                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
            )
            rclpy.init()
            node = rclpy.create_node("panda_m1b3_validation")
            states, clocks = [], []
            node.create_subscription(JointState, "/joint_states", states.append, qos_profile_sensor_data)
            node.create_subscription(Clock, "/clock", clocks.append, qos_profile_sensor_data)
            buffer = Buffer()
            listener = TransformListener(buffer, node)

            def until(predicate, timeout=30):
                end = time.monotonic() + timeout
                while time.monotonic() < end:
                    if process.poll() is not None:
                        raise RuntimeError("Panda launch exited early")
                    rclpy.spin_once(node, timeout_sec=0.05)
                    if predicate():
                        return
                raise TimeoutError("Timed out waiting for ROS validation condition")

            def future_result(future, timeout=30):
                until(future.done, timeout)
                return future.result()

            def service(name, service_type):
                client = node.create_client(service_type, name)
                until(client.service_is_ready, 60)
                return future_result(client.call_async(service_type.Request()))

            def action(client, goal):
                until(client.server_is_ready)
                handle = future_result(client.send_goal_async(goal))
                assert handle.accepted, "Action goal rejected"
                result = future_result(handle.get_result_async())
                assert result.status == GoalStatus.STATUS_SUCCEEDED, f"Action status: {result.status}"
                return result.result

            arm_names = [f"panda_joint{i}" for i in range(1, 8)]
            fingers = ["panda_finger_joint1", "panda_finger_joint2"]
            try:
                expected_controllers = {"panda_arm_controller", "panda_hand_controller", "joint_state_broadcaster"}
                end = time.monotonic() + 60
                while True:
                    response = service("/controller_manager/list_controllers", ListControllers)
                    active = {c.name for c in response.controller if c.state == "active"}
                    if expected_controllers <= active:
                        break
                    if time.monotonic() > end:
                        raise TimeoutError(f"Controllers not active: {active}")
                    rclpy.spin_once(node, timeout_sec=0.2)
                report["checks"]["active_controllers"] = sorted(active)
                interfaces = service("/controller_manager/list_hardware_interfaces", ListHardwareInterfaces)
                commands = {i.name for i in interfaces.command_interfaces if i.is_available and i.is_claimed}
                assert commands == {f"{n}/position" for n in arm_names + fingers[:1]}, commands
                report["checks"]["claimed_command_interfaces"] = sorted(commands)
                until(lambda: states and len(clocks) >= 2)
                assert set(states[-1].name) == set(arm_names + fingers)
                assert all(math.isfinite(v) for v in states[-1].position + states[-1].velocity + states[-1].effort)
                first_clock = clocks[-1].clock.sec + clocks[-1].clock.nanosec * 1e-9
                until(lambda: clocks[-1].clock.sec + clocks[-1].clock.nanosec * 1e-9 > first_clock + 0.2)
                report["checks"]["clock_advances"] = True
                until(lambda: buffer.can_transform("panda_link0", "panda_hand", Time()))
                transform = buffer.lookup_transform("panda_link0", "panda_hand", Time())
                report["checks"]["hand_translation"] = [getattr(transform.transform.translation, a) for a in "xyz"]

                arm = ActionClient(node, FollowJointTrajectory, "/panda_arm_controller/follow_joint_trajectory")
                hand = ActionClient(node, GripperCommand, "/panda_hand_controller/gripper_cmd")
                home = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
                target = [0.10, -0.10, 0.10, -1.67079, 0.10, 1.67079, -0.6853]
                for label, positions in [("arm_move", target), ("arm_return", home)]:
                    goal = FollowJointTrajectory.Goal()
                    goal.trajectory.joint_names = arm_names
                    point = JointTrajectoryPoint()
                    point.positions = positions
                    point.velocities = [0.0] * 7
                    point.time_from_start.sec = 3
                    goal.trajectory.points = [point]
                    result = action(arm, goal)
                    assert result.error_code == FollowJointTrajectory.Result.SUCCESSFUL, result.error_string
                    prior_count = len(states)
                    until(lambda: len(states) > prior_count)
                    measured = dict(zip(states[-1].name, states[-1].position))
                    errors = [abs(measured[n] - q) for n, q in zip(arm_names, positions)]
                    assert max(errors) < 0.05, errors
                    report["checks"][label] = {"positions": [measured[n] for n in arm_names], "max_error_rad": max(errors)}
                for opening in [0.02, 0.0, 0.04]:
                    goal = GripperCommand.Goal()
                    goal.command.position = opening
                    goal.command.max_effort = 20.0
                    result = action(hand, goal)
                    assert result.reached_goal and not result.stalled
                    prior_count = len(states)
                    until(lambda: len(states) > prior_count)
                    measured = dict(zip(states[-1].name, states[-1].position))
                    values = [measured[n] for n in fingers]
                    assert max(abs(q-opening) for q in values) < 0.002, values
                    assert abs(values[0]-values[1]) < 0.001, values
                    report["checks"][f"gripper_{opening:.2f}"] = values
                report["status"] = "PASS"
            except Exception as error:
                report["error"] = str(error)
            finally:
                node.destroy_node()
                rclpy.shutdown()
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=5)
                if report["status"] != "PASS":
                    print(log_path.read_text()[-12000:])
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
