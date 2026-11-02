#!/usr/bin/env python3
"""Physically place a stationary tray in each bin, with verified ROS resets."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

import rclpy
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from rclpy.signals import SignalHandlerOptions
from trajectory_msgs.msg import JointTrajectoryPoint

from bin_placement_rules import check_state, footprint_inside
from grasp_results import write_json
from static_reset import ResetObserver, ResetManager
from test_static_lift import require, distance


class BinTransferTest(ResetObserver):
    def __init__(self, node, plan):
        self.minimum_transfer_rise = float("inf")
        super().__init__(node, plan)

    def check(self, state):
        if not self.guarded:
            return
        slip = check_state(state, self.plan, self.reference, self.grasp, self.stage)
        self.maximum_slip = max(self.maximum_slip, slip)
        if self.reference is not None and self.stage == "transfer":
            self.minimum_transfer_rise = min(self.minimum_transfer_rise,
                                             state["tray_xyz"][2]-self.reference["tray_xyz"][2])
        if self.reference is not None and self.stage in {"lower", "release", "retreat", "return", "home", "verify"}:
            self.maximum_placement_force = max(self.maximum_placement_force,
                                               state["bin_floor_force_N"][self.plan["bin_index"]])

    def move(self, phase):
        self.stage = phase["stage"]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.names
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in phase["positions"]]
        point.velocities = [0.0]*7
        seconds = self.plan["duration_s"]
        point.time_from_start.sec = int(seconds)
        point.time_from_start.nanosec = round((seconds-int(seconds))*1e9)
        goal.trajectory.points = [point]
        result = self.action(self.arm, goal)
        require(result.error_code == FollowJointTrajectory.Result.SUCCESSFUL, result.error_string)
        self.wait_sim(self.plan["settle_s"])
        require(distance(self.state["hand_xyz"], phase["hand_target_xyz"]) < 0.003,
                "Hand position error exceeded 3 mm")
        if self.stage != "home":
            quaternion = [0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0]
            dot = abs(sum(a*b for a, b in zip(self.state["hand_quat_wxyz"], quaternion)))
            require(dot > math.cos(math.radians(1)/2), "Hand orientation error exceeded 1 degree")
        if self.stage == "approach":
            require(self.state["left_force_N"]+self.state["right_force_N"] < 0.1, "Contact before closure")
        print(f"{self.plan['bin_name']}/{phase['name']}", flush=True)
        return self.state.copy()

    def run(self):
        report = {"status": "FAIL", "bin": self.plan["bin_name"], "checks": {}}
        self.minimum_transfer_rise = float("inf")
        try:
            self.until(lambda: self.state and all(n in self.joints for n in self.names)
                       and self.arm.server_is_ready() and self.hand.server_is_ready())
            self.confirm_controller()
            self.grip(0.04)
            self.wait_sim(1.0)
            require(distance(self.state["tray_xyz"], self.plan["initial_state"]["tray_xyz"]) < 0.002,
                    "Tray pose differs from the native plan")
            self.reference = self.state.copy()
            require(self.reference["support_force_N"] > 0.5, "Tray must start on the support")
            report["checks"]["initial_state"] = self.reference
            for phase in self.plan["phases"]:
                self.stage = phase["stage"]
                if phase["operation"] == "arm":
                    report["checks"][phase["name"]] = self.move(phase)
                elif phase["operation"] == "gripper":
                    if phase["name"] == "release":
                        index = self.plan["bin_index"]
                        require(self.state["bin_floor_force_N"][index] > 0.8*self.reference["support_force_N"]
                                and footprint_inside(self.state, self.plan), "Bin floor must support the tray before release")
                        report["checks"]["placed_state"] = self.state.copy()
                    result = self.grip(float(phase["opening_m"]), contact=phase["expect_contact"])
                    self.wait_sim(1.0)
                    report["checks"][phase["name"]] = {"action": result, "state": self.state.copy()}
                    if phase["expect_contact"]:
                        require(min(self.state["left_force_N"], self.state["right_force_N"]) > 0.2,
                                "Both fingers must contact the handle")
                        self.grasp = self.state.copy()
                    if phase["name"] == "release":
                        require(self.state["left_force_N"]+self.state["right_force_N"] < 0.1,
                                "Fingers did not release")
                elif phase["operation"] == "wait":
                    self.wait_sim(phase["seconds"])
                    report["checks"][phase["name"]] = self.state.copy()
                else:
                    raise RuntimeError("Unknown plan operation")
            self.stage = "verify"
            self.history.clear()
            self.wait_sim(self.plan["placement_verify_s"])
            require(len(self.history) >= 50, "Not enough settled placement samples")
            require(math.isfinite(self.minimum_transfer_rise), "No transfer samples were recorded")
            report["checks"]["final_state"] = self.state.copy()
            report["metrics"] = {
                "maximum_relative_slip_m": self.maximum_slip,
                "maximum_bin_floor_force_N": self.maximum_placement_force,
                "minimum_transfer_rise_m": self.minimum_transfer_rise,
                "maximum_settled_linear_velocity_m_s": max(abs(v) for s in self.history for v in s["tray_linear_velocity_m_s"]),
                "maximum_settled_angular_velocity_rad_s": max(abs(v) for s in self.history for v in s["tray_angular_velocity_rad_s"]),
                "placement_samples": len(self.history),
            }
            report["status"] = "PASS"
        except (Exception, KeyboardInterrupt) as error:
            report["error"] = str(error) or "Interrupted by user"
            report["last_state"] = self.state.copy()
            report["failure_handling"] = self.stop_after_failure()
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--bin", choices=("both", "bin_a", "bin_b"), default="both")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    require(plan.get("status") == "PASS" and plan.get("schema_version") == 5
            and plan.get("mode") == "stationary_bin_transfer", "Generate a passing M1E plan first")
    require(plan.get("interpolate_from_desired_state") is True, "Continuous desired-state interpolation is required")
    require(1 <= args.repeats <= 20, "Use one to 20 repeats")
    destinations = [p for p in plan["bins"] if args.bin in ("both", p["bin_name"])]
    require(len(destinations) == (2 if args.bin == "both" else 1), "Plan has missing destinations")
    folder = args.output_dir or Path("results")/f"M1E-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True, exist_ok=False)
    description = Path(get_package_share_directory("sorting_description"))
    simulation = Path(get_package_share_directory("sorting_simulation"))
    recorded = [description/"mujoco/static_grasp"/n for n in ("scene_bins.xml", "bins.xml", "props.xml", "panda_static.xml")]
    recorded += [description/"config/panda_controllers.yaml", simulation/"config/lift_controllers.yaml"]
    manifest = {"protocol": "Stationary tray, one physical transfer per bin, verified reset before every trial and after final placement",
                "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in recorded},
                "requested_bins": [p["bin_name"] for p in destinations], "repeats": args.repeats,
                "ros_distro": os.environ.get("ROS_DISTRO"), "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
                "python": sys.version, "started_at": datetime.now().astimezone().isoformat(),
                "oracle_geometry": True, "moving_grasp": False}
    write_json(folder/"manifest.json", manifest)
    write_json(folder/"plan.json", plan)
    selected = {**plan, **destinations[0]}
    summary = {"status": "RUNNING", "acceptance_passed": False, "trials": [],
               "requested_placements": args.repeats*len(destinations), "completed_placements": 0,
               "successful_placements": 0, "failed_placements": 0,
               "final_reset_status": "NOT_RUN", "output_directory": str(folder.resolve())}
    write_json(folder/"summary.json", summary)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("stationary_bin_transfer_test")
    observer = BinTransferTest(node, selected)
    manager = ResetManager(observer)
    try:
        for repeat in range(1, args.repeats+1):
            for destination in destinations:
                observer.plan = {**plan, **destination}
                label = f"{repeat:03d}-{destination['bin_name']}"
                reset = manager.reset_home()
                write_json(folder/f"reset-{label}.json", reset)
                require(reset["status"] == "PASS", reset.get("error", "Initial reset failed"))
                result = observer.run()
                write_json(folder/f"placement-{label}.json", result)
                summary["trials"].append({"repeat": repeat, "bin": destination["bin_name"],
                                           "status": result["status"], "reset_status": reset["status"],
                                           "metrics": result.get("metrics"), "error": result.get("error")})
                summary["completed_placements"] = len(summary["trials"])
                summary["successful_placements"] = sum(t["status"] == "PASS" for t in summary["trials"])
                summary["failed_placements"] = summary["completed_placements"]-summary["successful_placements"]
                write_json(folder/"summary.json", summary)
                require(result["status"] == "PASS", result.get("error", "Placement failed"))
        final_reset = manager.reset_home()
        write_json(folder/"final_reset.json", final_reset)
        summary["final_reset_status"] = final_reset["status"]
        require(final_reset["status"] == "PASS", final_reset.get("error", "Final reset failed"))
        require(summary["completed_placements"] == summary["requested_placements"], "Placement run is incomplete")
        summary["acceptance_passed"] = args.bin == "both"
        summary["status"] = "PASS" if summary["acceptance_passed"] else "SMOKE_PASS"
    except (Exception, KeyboardInterrupt) as error:
        summary["status"] = "FAIL"
        summary["error"] = str(error) or "Interrupted by user"
        summary["failure_handling"] = observer.stop_after_failure()
    finally:
        summary["final"] = True
        write_json(folder/"summary.json", summary)
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] in ("PASS", "SMOKE_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
