"""Make a scene-specific Panda file without changing the M1B model."""

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "src" / "sorting_description" / "mujoco"

SOURCE = MJCF / "franka_emika_panda" / "panda.xml"
DESTINATION = MJCF / "static_grasp"

robot = ET.parse(SOURCE).getroot()
props = ET.parse(DESTINATION / "props.xml").getroot()

tray = props.find("worldbody/body[@name='tray']")

if (
    tray is None
    or tray.find("freejoint[@name='tray_free_joint']") is None
    or len(props.findall(".//freejoint")) != 1
):
    raise RuntimeError("Expected one free joint on the tray")

expected = [f"panda_joint{i}" for i in range(1, 8)]
expected += ["panda_finger_joint1", "panda_finger_joint2"]

names = [
    joint.get("name")
    for joint in robot.findall("worldbody//joint")
]

if names != expected:
    raise RuntimeError(f"Unexpected Panda joint order: {names}")

compiler = robot.find("compiler")

if compiler is None:
    raise RuntimeError("Panda compiler element is missing")

compiler.set("meshdir", "../franka_emika_panda/assets")

keyframe = robot.find("keyframe")

if keyframe is None or keyframe.find("key[@name='home']") is None:
    raise RuntimeError("Panda home keyframe is missing")

tray_pose = tray.get("pos", "0 0 0").split()
tray_pose += tray.get("quat", "1 0 0 0").split()

if len(tray_pose) != 7:
    raise RuntimeError(
        "Tray pose must contain three position "
        "and four quaternion values"
    )

for key in keyframe.findall("key"):
    qpos = key.get("qpos", "").split()

    if len(qpos) != 9 or len(key.get("ctrl", "").split()) != 8:
        raise RuntimeError(
            "Expected nine Panda positions and eight control values"
        )

    key.set("qpos", " ".join(qpos + tray_pose))

    if key.get("qvel") is not None:
        qvel = key.get("qvel").split()

        if len(qvel) != 9:
            raise RuntimeError(
                "Unexpected Panda keyframe velocity size"
            )

        key.set("qvel", " ".join(qvel + ["0"] * 6))

ET.indent(robot, space="  ")
gripper = robot.find("actuator/position[@name='panda_finger_joint1']")
if gripper is None:
    raise RuntimeError("Prepared Panda finger position actuator is missing")
gripper.set("forcelimited", "true")
gripper.set("forcerange", "-20 20")

output = DESTINATION / "panda_static.xml"

ET.ElementTree(robot).write(
    output,
    encoding="utf-8",
    xml_declaration=True,
)

shutil.copy2(
    SOURCE.parent / "LICENSE",
    DESTINATION / "LICENSE",
)

print(f"Prepared: {output}")
print(
    "Home keyframe: nine Panda positions "
    "plus seven tray pose values"
)
