from pathlib import Path
import shutil
import re

ROOT = Path(__file__).resolve().parents[1]

src = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"
dst = ROOT / "src" / "sorting_description" / "mujoco" / "franka_emika_panda"

if not src.exists():
    raise FileNotFoundError(f"Menagerie Panda not found: {src}")

# Re-create project-owned copy from the pinned upstream model.
if dst.exists():
    shutil.rmtree(dst)

shutil.copytree(src, dst)

panda_xml = dst / "panda.xml"
text = panda_xml.read_text()

# ------------------------------------------------------------------
# 1. Align MuJoCo joint names with the ROS / MoveIt Panda joint names.
# ------------------------------------------------------------------

joint_mapping = {
    "finger_joint1": "panda_finger_joint1",
    "finger_joint2": "panda_finger_joint2",
    "joint1": "panda_joint1",
    "joint2": "panda_joint2",
    "joint3": "panda_joint3",
    "joint4": "panda_joint4",
    "joint5": "panda_joint5",
    "joint6": "panda_joint6",
    "joint7": "panda_joint7",
}

for old, new in joint_mapping.items():
    text = text.replace(f'"{old}"', f'"{new}"')

# ------------------------------------------------------------------
# 2. Rename arm actuators to match the controlled ROS joints.
#
# The MuJoCo ros2_control examples use matching actuator / joint
# names for the simple one-actuator-per-joint position-control case.
# ------------------------------------------------------------------

for i in range(1, 8):
    text = text.replace(
        f'name="actuator{i}"',
        f'name="panda_joint{i}"',
    )

# ------------------------------------------------------------------
# 3. Replace the original Menagerie gripper command convention.
#
# Upstream Menagerie exposes the gripper control as 0..255.
# For ROS we want the command to mean finger opening directly
# in metres: 0.00 .. 0.04.
# ------------------------------------------------------------------

gripper_pattern = re.compile(
    r'<general class="panda" name="actuator8" tendon="split".*?/>',
    re.DOTALL,
)

replacement = (
    '<position name="panda_finger_joint1" '
    'tendon="split" '
    'kp="1000" '
    'dampratio="3.0" '
    'ctrlrange="0 0.04"/>'
)

text, count = gripper_pattern.subn(replacement, text)

if count != 1:
    raise RuntimeError(
        f"Expected exactly one gripper actuator8, found {count}. "
        "The upstream Menagerie model may have changed."
    )

# The Menagerie home keyframe previously commanded the gripper with 255.
# It must now command the equivalent physical opening: 0.04 m.
text = text.replace(
    'ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255"',
    'ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 0.04"',
)

panda_xml.write_text(text)

# ------------------------------------------------------------------
# Basic validation.
# ------------------------------------------------------------------

for i in range(1, 8):
    expected = f'name="panda_joint{i}"'
    if expected not in text:
        raise RuntimeError(f"Missing expected name: {expected}")

for name in ("panda_finger_joint1", "panda_finger_joint2"):
    if name not in text:
        raise RuntimeError(f"Missing expected finger joint: {name}")

print(f"Prepared Panda MJCF at:\n{dst}")
print()
print("ROS/MuJoCo joint naming aligned.")
print("Gripper command range converted to 0.00 .. 0.04 m.")