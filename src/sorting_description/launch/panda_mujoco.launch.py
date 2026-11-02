"""M1B.3: Panda controllers with the project-owned MuJoCo scene."""

from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_setup(context):
    share = Path(get_package_share_directory("sorting_description"))

    description = xacro.process_file(
        str(share / "urdf" / "panda_mujoco.urdf.xacro"),
        mappings={
            "headless": LaunchConfiguration("headless").perform(context),
            "mujoco_model": LaunchConfiguration("mujoco_model").perform(context),
        },
    ).toxml()

    controllers = str(share / "config" / "panda_controllers.yaml")

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[
                {
                    "robot_description": ParameterValue(
                        description, value_type=str
                    ),
                    "use_sim_time": True,
                },
            ],
            output="screen",
        ),
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            parameters=[controllers, {"use_sim_time": True}],
            output="screen",
            on_exit=Shutdown(),
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "joint_state_broadcaster",
                "panda_arm_controller",
                "panda_hand_controller",
                "--controller-manager",
                "/controller_manager",
                "--param-file",
                controllers,
                "--activate-as-group",
                "--controller-manager-timeout",
                "60",
            ],
            output="screen",
        ),
    ]


def generate_launch_description():
    share = Path(get_package_share_directory("sorting_description"))

    return LaunchDescription([
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Run without the MuJoCo viewer",
        ),
        DeclareLaunchArgument(
            "mujoco_model",
            default_value=str(
                share / "mujoco" / "franka_emika_panda" / "scene.xml"
            ),
            description="Path to a native MJCF scene with the prepared Panda",
        ),
        OpaqueFunction(function=launch_setup),
    ])