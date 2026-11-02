#!/usr/bin/env python3
"""Verify reset after a short arm and finger movement, three times by default."""
import argparse
import json
from pathlib import Path

import rclpy
from rclpy.signals import SignalHandlerOptions
from grasp_results import write_json
from static_reset import ResetObserver, ResetManager
from test_static_lift import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--checks", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/M1C4-ros-reset.json"))
    args = parser.parse_args()
    require(1 <= args.checks <= 10, "Use between one and ten reset checks")
    plan = json.loads(args.plan.read_text())
    require(plan.get("status") == "PASS" and plan.get("schema_version") == 3, "Use a passing M1C3 plan")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAIL", "checks": []}
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("verify_static_reset")
    observer = ResetObserver(node, plan)
    manager = ResetManager(observer)
    try:
        initial = manager.reset_home()
        report["initial_reset"] = initial
        require(initial["status"] == "PASS", initial.get("error", "Initial reset failed"))
        for index in range(1, args.checks+1):
            observer.reference = observer.state.copy()
            observer.confirm_controller()
            observer.move(plan["phases"][0])
            observer.grip(0.025)
            observer.wait_sim(0.5)
            disturbance = {"state": observer.state.copy(), "joint_positions": observer.joints.copy()}
            require(max(abs(observer.joints[n]-q) for n,q in zip(observer.names, manager.expected["qpos"][:7])) > 0.04,
                    "Probe did not move the arm away from home")
            result = manager.reset_home()
            result["probe_before_reset"] = disturbance
            report["checks"].append(result)
            write_json(args.output, report)
            require(result["status"] == "PASS", result.get("error", "Reset failed"))
            print(f"Reset check {index}/{args.checks}: PASS")
        report["status"] = "PASS"
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = str(error) or "Interrupted by user"
        report["failure_handling"] = observer.stop_after_failure()
    finally:
        write_json(args.output, report)
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
