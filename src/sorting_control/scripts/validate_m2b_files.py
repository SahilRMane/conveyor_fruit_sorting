#!/usr/bin/env python3
"""Check M2B files and native geometry without starting ROS nodes."""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from ownership_core import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/M2B-preflight"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    description = Path(get_package_share_directory("sorting_description"))
    control = Path(get_package_share_directory("sorting_control"))
    scene = description/"mujoco/static_grasp/scene_tracking.xml"
    report = {"status": "FAIL", "oracle_only": True, "live_test": "NOT_RUN"}
    try:
        require(scene.exists(), "Build sorting_description to install scene_tracking.xml")
        includes = {i.get("file") for i in ET.parse(scene).getroot().findall("include")}
        require({"panda_static.xml", "props.xml", "bins.xml"}.issubset(includes), "Combined scene is incomplete")
        robot = xacro.process_file(str(description/"urdf/panda_mujoco.urdf.xacro"),
                                  mappings={"headless": "true", "mujoco_model": str(scene)}).toxml()
        urdf = args.output_dir/"panda.urdf"; urdf.write_text(robot)
        semantic = ET.parse(Path(get_package_share_directory("moveit_resources_panda_moveit_config"))/"config/panda.srdf").getroot()
        for joint in semantic.findall("virtual_joint"): semantic.remove(joint)
        for group in semantic.findall("group"):
            if group.get("name") == "panda_arm": group.find("chain").set("tip_link", "panda_hand")
        srdf = args.output_dir/"panda-fixed.srdf"; srdf.write_text(ET.tostring(semantic, encoding="unicode"))
        native = args.output_dir/"native-model.json"
        executable = Path(get_package_prefix("sorting_control"))/"lib/sorting_control/verify_m2b_model"
        subprocess.run([str(executable.parent/"verify_planning_only_manager")], check=True)
        report["planning_only_plugin"] = "PASS"
        subprocess.run([str(executable), str(urdf), str(srdf), str(control/"config/fixtures.yaml"),
                        str(scene), str(control/"config/m2b.yaml"), str(native)], check=True)
        report["native"] = json.loads(native.read_text()); require(report["native"]["status"] == "PASS", "Native model check failed")
        report["status"] = "PASS"
    except Exception as error: report["error"] = str(error)
    (args.output_dir/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2)); return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
