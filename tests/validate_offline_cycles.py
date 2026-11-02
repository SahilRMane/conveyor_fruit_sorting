"""Replay 20 native MuJoCo lift cycles; this does not verify ROS controller reset."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("planner", type=Path)
parser.add_argument("scene", type=Path)
parser.add_argument("--output-dir", type=Path, default=Path("results/offline-20-cycles"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=False)
rows = []
for index in range(1, 21):
    target = args.output_dir/f"cycle_{index:03d}.json"
    result = subprocess.run([str(args.planner.resolve()), str(args.scene.resolve()), str(target)],
                            capture_output=True, text=True)
    data = json.loads(target.read_text())
    row = {"cycle":index, "status":data["status"], "exit_code":result.returncode,
           "minimum_rise_m":data.get("hold_metrics", {}).get("minimum_rise_m"),
           "maximum_relative_slip_m":data.get("maximum_relative_slip_m")}
    rows.append(row)
    print(f"Offline cycle {index}/20: {row['status']}", flush=True)
    if result.returncode:
        print(result.stderr)
summary = {"status":"PASS" if all(r["status"]=="PASS" and r["exit_code"]==0 for r in rows) else "FAIL",
           "cycles":rows, "ros_reset_lifecycle":"NOT_RUN"}
(args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
raise SystemExit(0 if summary["status"]=="PASS" else 1)
