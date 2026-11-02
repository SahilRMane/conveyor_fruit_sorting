#!/usr/bin/env python3
"""Run native moving-grasp feasibility and file checks without ROS nodes."""
import argparse
import json
from pathlib import Path
import subprocess
from ament_index_python.packages import get_package_share_directory,get_package_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("results/M2C-preflight"))
    args = parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    report = {"status":"FAIL","oracle_only":True,"live_test":"NOT_RUN"}
    try:
        scene = Path(get_package_share_directory("sorting_description"))/"mujoco/static_grasp/scene_tracking.xml"
        config = Path(get_package_share_directory("sorting_control"))/"config/m2c.yaml"
        exe = Path(get_package_prefix("sorting_simulation"))/"lib/sorting_simulation/verify_moving_grasp"
        output = args.output_dir/"native-moving.json"
        subprocess.run([str(exe),str(scene),str(config),str(output)],check=True)
        report["native"] = json.loads(output.read_text())
        if report["native"]["status"] != "PASS": raise RuntimeError("Native moving grasp check failed")
        report["status"] = "PASS"
    except Exception as error:report["error"] = str(error)
    (args.output_dir/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2));return 0 if report["status"]=="PASS" else 1


if __name__ == "__main__":raise SystemExit(main())
