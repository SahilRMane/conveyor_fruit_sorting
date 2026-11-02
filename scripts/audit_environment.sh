#!/usr/bin/env bash

echo "================================="
echo " OS"
echo "================================="
cat /etc/os-release

echo
echo "================================="
echo " Kernel"
echo "================================="
uname -a

echo
echo "================================="
echo " CPU"
echo "================================="
lscpu | grep -E "Model name|Socket|Core|Thread|CPU\(s\)"

echo
echo "================================="
echo " RAM"
echo "================================="
free -h

echo
echo "================================="
echo " GPU"
echo "================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi NOT FOUND"
fi

echo
echo "================================="
echo " ROS"
echo "================================="
echo "ROS_DISTRO=${ROS_DISTRO:-NOT_SET}"

if command -v ros2 &> /dev/null; then
    which ros2
    ros2 --help > /dev/null && echo "ros2 command works"
else
    echo "ros2 NOT FOUND"
fi

echo
echo "================================="
echo " Python"
echo "================================="
python3 --version
which python3

echo
echo "================================="
echo " Compiler"
echo "================================="
gcc --version | head -n 1
g++ --version | head -n 1

echo
echo "================================="
echo " CMake"
echo "================================="
cmake --version | head -n 1

echo
echo "================================="
echo " Colcon"
echo "================================="
if command -v colcon &> /dev/null; then
    colcon --version 2>/dev/null || echo "colcon installed"
else
    echo "colcon NOT FOUND"
fi

echo
echo "================================="
echo " Important ROS packages"
echo "================================="

if command -v ros2 &> /dev/null; then

    echo "--- MoveIt ---"
    ros2 pkg list | grep -E "^moveit" | head -20

    echo
    echo "--- ros2_control ---"
    ros2 pkg list | grep -E "controller_manager|ros2_control|joint_trajectory_controller"

    echo
    echo "--- MuJoCo ---"
    ros2 pkg list | grep -i mujoco
fi

echo
echo "================================="
echo " Python MuJoCo"
echo "================================="

python3 - <<'PY'
try:
    import mujoco
    print("MuJoCo Python installed")
    print("MuJoCo version:", mujoco.__version__)
except Exception as e:
    print("MuJoCo Python NOT available")
    print(e)
PY

echo
echo "================================="
echo " Disk"
echo "================================="
df -h ~