#!/usr/bin/env python3
"""Reset and run 20 complete nominal grasp cycles with a separate report for each."""
import argparse
import contextlib
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.signals import SignalHandlerOptions
from grasp_results import assess, write_json
from static_reset import ResetObserver, ResetManager
from test_static_lift import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    require(1 <= args.cycles <= 100, "Use between one and 100 cycles")
    plan = json.loads(args.plan.read_text())
    require(plan.get("status") == "PASS" and plan.get("schema_version") == 3, "Use a passing M1C3 lift plan")
    started = datetime.now(ZoneInfo("Europe/Berlin"))
    folder = args.output_dir or Path("results")/f"M1C4-{started:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"started_at": started.isoformat(), "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                "ros_distro": os.environ.get("ROS_DISTRO"), "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
                "python": sys.version, "platform": platform.platform(), "requested_cycles": args.cycles,
                "protocol": "Identical nominal home reset and fixed M1C3 lift plan for every trial"}
    description = Path(get_package_share_directory("sorting_description"))
    simulation = Path(get_package_share_directory("sorting_simulation"))
    recorded = [description/"mujoco/static_grasp"/name for name in ("scene_lift.xml", "props.xml", "panda_static.xml")]
    recorded += [description/"config/panda_controllers.yaml", simulation/"config/lift_controllers.yaml"]
    manifest["file_sha256"] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in recorded}
    packages = ["ros-jazzy-mujoco-ros2-control", "ros-jazzy-controller-manager", "ros-jazzy-joint-trajectory-controller"]
    versions = subprocess.run(["dpkg-query", "-W", *packages], capture_output=True, text=True)
    manifest["dependency_versions"] = versions.stdout.splitlines()
    revision = Path("third_party/MENAGERIE_COMMIT.txt")
    manifest["menagerie_commit"] = revision.read_text().strip() if revision.is_file() else None
    write_json(folder/"manifest.json", manifest)
    write_json(folder/"plan.json", plan)
    trials = []
    interrupted, infrastructure_error = False, None
    final_reset_status = "NOT_RUN"
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("nominal_static_grasps")
    observer = ResetObserver(node, plan)
    manager = ResetManager(observer)

    def save_progress(final=False):
        summary = assess(args.cycles, trials, interrupted, infrastructure_error)
        summary["final"] = final
        if not final:
            summary["assessment_so_far"] = summary["status"]
            if not interrupted and infrastructure_error is None:
                summary["status"] = "RUNNING"
            summary["acceptance_passed"] = False
        summary["trials"] = trials
        summary["output_directory"] = str(folder.resolve())
        summary["final_reset_status"] = final_reset_status
        write_json(folder/"summary.json", summary)
        with (folder/"cycles.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["cycle", "status", "reset_status", "minimum_rise_m",
                                                       "maximum_relative_slip_m", "error"])
            writer.writeheader()
            writer.writerows(trials)
        return summary

    save_progress()
    try:
        for index in range(1, args.cycles+1):
            print(f"Cycle {index}/{args.cycles}: reset", flush=True)
            reset = manager.reset_home()
            write_json(folder/f"reset_{index:03d}.json", reset)
            if reset["status"] != "PASS":
                infrastructure_error = reset.get("error", "Reset verification failed")
                trials.append({"cycle": index, "status": "NOT_RUN", "reset_status": "FAIL",
                               "minimum_rise_m": None, "maximum_relative_slip_m": None, "error": infrastructure_error})
                break
            print(f"Cycle {index}/{args.cycles}: grip, lift, hold, lower and release", flush=True)
            try:
                with (folder/f"cycle_{index:03d}.log").open("w") as stream, contextlib.redirect_stdout(stream):
                    result = observer.run()
            except KeyboardInterrupt:
                interrupted = True
                result = {"status": "INTERRUPTED", "error": "Interrupted by user",
                          "last_state": observer.state.copy(), "failure_handling": observer.stop_after_failure()}
            write_json(folder/f"cycle_{index:03d}.json", result)
            trials.append({"cycle": index, "status": result["status"], "reset_status": "PASS",
                           "minimum_rise_m": result.get("hold_metrics", {}).get("minimum_rise_m"),
                           "maximum_relative_slip_m": result.get("maximum_relative_slip_m"),
                           "error": result.get("error")})
            summary = save_progress()
            print(f"Cycle {index}: {result['status']}; successes={summary['successful_cycles']}", flush=True)
            if interrupted:
                break
            # A grasp failure remains in the results. The next trial starts only
            # after a fresh reset passes. A reset failure aborts the batch.
        if not interrupted and infrastructure_error is None and len(trials) == args.cycles:
            print("Final home reset", flush=True)
            final_reset = manager.reset_home()
            write_json(folder/"final_reset.json", final_reset)
            final_reset_status = final_reset["status"]
            if final_reset_status != "PASS":
                infrastructure_error = final_reset.get("error", "Final reset failed")
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        infrastructure_error = str(error) or "Interrupted by user"
        write_json(folder/"abort.json", {"error": infrastructure_error,
                                        "failure_handling": observer.stop_after_failure()})
    finally:
        summary = save_progress(final=True)
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] in {"PASS", "SMOKE_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
