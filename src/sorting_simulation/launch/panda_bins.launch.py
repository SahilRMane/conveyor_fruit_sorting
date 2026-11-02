"""Start stationary bin placement with continuous arm commands and contact feedback."""
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def setup(context):
    description_share = Path(get_package_share_directory("sorting_description"))
    simulation_share = Path(get_package_share_directory("sorting_simulation"))
    description = xacro.process_file(
        str(description_share / "urdf/panda_mujoco.urdf.xacro"),
        mappings={"headless": LaunchConfiguration("headless").perform(context),
                  "mujoco_model": str(description_share / "mujoco/static_grasp/scene_bins.xml")},
    ).toxml()
    controllers = str(description_share / "config/panda_controllers.yaml")
    overrides = LaunchConfiguration("controller_overrides").perform(context)
    return [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": ParameterValue(description, value_type=str),
                          "use_sim_time": True}], output="screen"),
        Node(package="mujoco_ros2_control", executable="ros2_control_node",
             parameters=[controllers, str(simulation_share / "config/contact_plugins.yaml"),
                         {"use_sim_time": True}], output="screen", on_exit=Shutdown()),
        Node(package="controller_manager", executable="spawner",
             arguments=["joint_state_broadcaster", "panda_arm_controller", "panda_hand_controller",
                        "--controller-manager", "/controller_manager", "--param-file", controllers,
                        "--param-file", overrides, "--activate-as-group",
                        "--controller-manager-timeout", "60"], output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument(
            "controller_overrides",
            default_value=str(Path(get_package_share_directory("sorting_simulation")) / "config/lift_controllers.yaml"),
            description="Controller override file; default retains the M1 bin settings"),
        OpaqueFunction(function=setup),
    ])
