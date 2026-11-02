"""Check that absent contact and a missing force limit cannot produce PASS."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("planner", type=Path)
parser.add_argument("scene", type=Path)
args = parser.parse_args()
scene = args.scene.resolve()
with tempfile.TemporaryDirectory(prefix="m1c2_negative_") as temp:
    folder = Path(temp)
    for name in ("scene.xml", "props.xml", "panda_static.xml"):
        (folder / name).write_text((scene.parent / name).read_text())
    robot = ET.parse(folder / "panda_static.xml")
    assets = scene.parent / "../franka_emika_panda/assets"
    robot.getroot().find("compiler").set("meshdir", str(assets.resolve()))
    robot.write(folder / "panda_static.xml")
    props = ET.parse(folder / "props.xml")
    bar = props.getroot().find("worldbody/body[@name='tray']/geom[@name='handle_bar']")
    bar.set("contype", "0")
    bar.set("conaffinity", "0")
    props.write(folder / "props.xml")
    output = folder / "result.json"
    process = subprocess.run([str(args.planner.resolve()), str(folder / "scene.xml"), str(output)],
                             capture_output=True, text=True)
    assert process.returncode == 1, process.stdout + process.stderr
    assert "No sustained bilateral handle contact" in process.stderr, process.stderr
    assert json.loads(output.read_text())["status"] == "FAIL"
    print("PASS: disabled handle collision cannot pass the contact test")

    (folder / "props.xml").write_text((scene.parent / "props.xml").read_text())
    gripper = robot.getroot().find("actuator/position[@name='panda_finger_joint1']")
    gripper.set("forcelimited", "false")
    robot.write(folder / "panda_static.xml")
    process = subprocess.run([str(args.planner.resolve()), str(folder / "scene.xml"), str(output)],
                             capture_output=True, text=True)
    assert process.returncode == 1, process.stdout + process.stderr
    assert "gripper limit must be +/-20 N" in process.stderr, process.stderr
    assert json.loads(output.read_text())["status"] == "FAIL"
    print("PASS: missing gripper actuator force limit is rejected")
