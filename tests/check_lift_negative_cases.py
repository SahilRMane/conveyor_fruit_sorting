"""Check that a heavy tray and the old contact model cannot pass this lift plan."""
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
with tempfile.TemporaryDirectory(prefix="m1c3_negative_") as temp:
    folder = Path(temp)
    for name in ("scene_lift.xml", "props.xml", "panda_static.xml"):
        (folder/name).write_text((scene.parent/name).read_text())
    robot = ET.parse(folder/"panda_static.xml")
    robot.getroot().find("compiler").set("meshdir", str((scene.parent/"../franka_emika_panda/assets").resolve()))
    robot.write(folder/"panda_static.xml")
    props = ET.parse(folder/"props.xml")
    props.getroot().find("worldbody/body[@name='tray']/geom[@name='tray_base']").set("mass", "3.30")
    props.write(folder/"props.xml")
    output = folder/"result.json"
    result = subprocess.run([str(args.planner.resolve()),str(folder/"scene_lift.xml"),str(output)],
                            capture_output=True,text=True)
    assert result.returncode == 1, result.stdout+result.stderr
    assert "slip" in result.stderr or "contact was lost" in result.stderr, result.stderr
    assert json.loads(output.read_text())["status"] == "FAIL"
    print("PASS: the 3.52 kg tray fails the physical grip/lift check")
    (folder/"props.xml").write_text((scene.parent/"props.xml").read_text())
    xml = ET.parse(folder/"scene_lift.xml")
    xml.getroot().find("option").set("cone","pyramidal")
    xml.getroot().find("option").set("impratio","1")
    xml.write(folder/"scene_lift.xml")
    result = subprocess.run([str(args.planner.resolve()),str(folder/"scene_lift.xml"),str(output)],
                            capture_output=True,text=True)
    assert result.returncode == 1 and "Use scene_lift.xml" in result.stderr, result.stdout+result.stderr
    assert json.loads(output.read_text())["status"] == "FAIL"
    print("PASS: the old contact settings are rejected for this validated lift")
