"""Check the installed Xacro, controller configuration and prepared MJCF."""
from pathlib import Path
import xml.etree.ElementTree as ET

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory

share = Path(get_package_share_directory("sorting_description"))
arm = [f"panda_joint{i}" for i in range(1, 8)]
fingers = ["panda_finger_joint1", "panda_finger_joint2"]
root = ET.fromstring(xacro.process_file(
    str(share / "urdf/panda_mujoco.urdf.xacro"), mappings={"headless": "true"}
).toxml())
assert len(root.findall("ros2_control")) == 1
hardware = root.find("ros2_control/hardware")
assert hardware.findtext("plugin") == "mujoco_ros2_control/MujocoSystemInterface"
scene = Path(hardware.find("param[@name='mujoco_model']").text)
assert scene.is_file()
assert hardware.find("param[@name='headless']").text == "true"
kinematic = {joint.attrib["name"]: joint for joint in root.findall("joint")}
controlled = {joint.attrib["name"]: joint for joint in root.findall("ros2_control/joint")}
assert set(controlled) == set(arm + fingers)
for name in arm + fingers[:1]:
    joint = controlled[name]
    assert [i.attrib["name"] for i in joint.findall("command_interface")] == ["position"]
    assert {i.attrib["name"] for i in joint.findall("state_interface")} == {"position", "velocity", "effort"}
    assert name in kinematic
assert not controlled[fingers[1]].findall("command_interface")
assert controlled[fingers[1]].attrib["mimic"] == "false"
assert kinematic[fingers[1]].find("mimic").attrib["joint"] == fingers[0]
assert kinematic["world_to_panda"].find("child").attrib["link"] == "panda_link0"
model = ET.parse(scene.parent / "panda.xml").getroot()
assert {j.attrib["name"] for j in model.findall(".//worldbody//joint")} == set(arm + fingers)
assert {a.attrib["name"] for a in model.findall("actuator/*")} == set(arm + fingers[:1])
home = [float(v) for v in model.find("keyframe/key[@name='home']").attrib["qpos"].split()]
for name, value in zip(arm + fingers, home):
    initial = float(controlled[name].find("state_interface[@name='position']/param[@name='initial_value']").text)
    assert initial == value, (name, initial, value)
config = yaml.safe_load((share / "config/panda_controllers.yaml").read_text())
assert config["panda_arm_controller"]["ros__parameters"]["joints"] == arm
assert config["panda_hand_controller"]["ros__parameters"]["joint"] == fingers[0]
assert config["joint_state_broadcaster"]["ros__parameters"]["joints"] == arm + fingers
print("PASS: Xacro expands; nine states/eight commands; passive second finger; home values; model paths; controller joints")
