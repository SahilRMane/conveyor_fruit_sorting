#!/usr/bin/env python3
"""Verify M2A files and MoveIt/MuJoCo geometry without starting a ROS node."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import yaml
import xacro
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.srv import ServoCommandType
from ownership_core import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/M2A-preflight"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    description = Path(get_package_share_directory("sorting_description"))
    control = Path(get_package_share_directory("sorting_control"))
    servo = Path(get_package_share_directory("moveit_servo"))
    moveit = Path(get_package_share_directory("moveit_resources_panda_moveit_config"))
    root = description/"mujoco/static_grasp"
    report = {"status": "FAIL", "live_ros_ownership": "NOT_RUN"}
    try:
        require(ServoCommandType.Request.JOINT_JOG == 0 and ServoCommandType.Request.TWIST == 1,
                "Installed Servo command constants differ")
        configuration = yaml.safe_load((control/"config/servo.yaml").read_text())["moveit_servo"]
        require(configuration["command_out_topic"] == "/sorting/arm/servo_output" and configuration["check_collisions"],
                "Servo must use the private output topic with collision checking enabled")
        expected = []
        for name in ("props.xml", "bins.xml"):
            for body in ET.parse(root/name).getroot().findall("worldbody/body"):
                require(body.get("quat", "1 0 0 0") == "1 0 0 0", "Fixture publisher requires unrotated boxes")
                offset = [float(v) for v in body.get("pos", "0 0 0").split()]
                for geom in body.findall("geom"):
                    local = [float(v) for v in geom.get("pos", "0 0 0").split()]
                    expected.append({"id": geom.get("name"), "dimensions": [2*float(v) for v in geom.get("size").split()],
                                     "position": [a+b for a, b in zip(offset, local)]})
        expected.append({"id": "static_floor", "dimensions": [4.0, 4.0, 0.02], "position": [0.0, 0.0, -0.01]})
        fixtures = yaml.safe_load((control/"config/fixtures.yaml").read_text())
        require(fixtures["frame"] == "world" and fixtures["boxes"] == expected,
                "Collision fixtures differ from the current MJCF; regenerate fixtures.yaml before testing")
        urdf_path = args.output_dir/"panda.urdf"
        urdf_path.write_text(xacro.process_file(str(description/"urdf/panda_mujoco.urdf.xacro"),
                              mappings={"headless": "true", "mujoco_model": str(root/"scene_bins.xml")}).toxml())
        semantic = ET.parse(moveit/"config/panda.srdf").getroot()
        for joint in semantic.findall("virtual_joint"):
            semantic.remove(joint)
        srdf_path = args.output_dir/"panda-fixed.srdf"
        srdf_path.write_text(ET.tostring(semantic, encoding="unicode"))
        native_path = args.output_dir/"native-model.json"
        result = subprocess.run(["ros2", "run", "sorting_control", "verify_m2a_model", str(urdf_path),
                                 str(srdf_path), str(control/"config/fixtures.yaml"),
                                 str(root/"scene_bins.xml"), str(native_path)])
        require(result.returncode == 0, "Native model check failed")
        native = json.loads(native_path.read_text()); require(native["status"] == "PASS", "Native model check failed")
        recorded = [root/n for n in ("scene_bins.xml", "bins.xml", "props.xml", "panda_static.xml")]
        recorded += [control/"config"/n for n in ("servo.yaml", "ownership.yaml", "fixtures.yaml")]
        recorded += [moveit/"config/panda.srdf", moveit/"config/joint_limits.yaml", moveit/"config/kinematics.yaml"]
        report.update(status="PASS", native_model=native,
                      moveit_servo_version=ET.parse(servo/"package.xml").getroot().findtext("version"),
                      file_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in recorded})
    except Exception as error:
        report["error"] = str(error)
    (args.output_dir/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": main()
