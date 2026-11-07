# Setup

## Requirements

The reference environment uses Ubuntu 24.04, ROS 2 Jazzy and Python 3.12.
It includes MoveIt 2, MoveIt Servo, ros2_control and mujoco_ros2_control.
The recorded integration uses mujoco_ros2_control 0.1.2 and MoveIt 2.12.4.
The package inventory is in [environment-20261102.json](environment-20261102.json).

Install ROS 2 Jazzy, colcon and rosdep through their official installation procedures.
Configure rosdep before resolving project dependencies.

## Clone and build

```bash
mkdir -p ~/Projects
cd ~/Projects
git clone https://github.com/SahilRMane/conveyor_fruit_sorting.git
cd conveyor_fruit_sorting

source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --base-paths src --executor sequential
source install/setup.bash
```

The checked-in source includes the project-owned Panda model, meshes, scene files and classifier weights.
A build does not require a separate Menagerie clone.

## Optional upstream asset preparation

Only clone Menagerie when regenerating the base Panda assets.
Preparation scripts replace generated model copies, so use a separate development checkout.

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie checkout 0059d4335f8156206f63a35662313385f7ad6d74
```

The same dependency pin is recorded in `dependencies.repos`.

## Terminal environment

Use this environment in each terminal:

```bash
cd ~/Projects/conveyor_sorting_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=173
unset LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE __EGL_VENDOR_LIBRARY_FILENAMES
export SORTING_RENDER_BACKEND=nvidia
```

Run one simulator launch at a time.
Rebuild and generate a fresh preflight after a source or environment change.
The environment inventory records versions. It is not a complete lockfile.

Read [GPU rendering](GPU_RENDERING.md) for device selection and the verified rendering scope.
