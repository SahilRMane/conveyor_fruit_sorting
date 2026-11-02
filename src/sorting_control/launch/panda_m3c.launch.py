"""M3C stationary classifier path; arm ownership remains HOLD."""
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def setup(context):
    description = Path(get_package_share_directory("sorting_description"))
    simulation = Path(get_package_share_directory("sorting_simulation"))
    control = Path(get_package_share_directory("sorting_control"))
    moveit = Path(get_package_share_directory("moveit_resources_panda_moveit_config"))
    robot = xacro.process_file(str(description/"urdf/panda_cameras.urdf.xacro"),
                              mappings={"camera_config": "cameras_m3c.yaml", "headless": LaunchConfiguration("headless").perform(context),
                                        "mujoco_model": str(description/("mujoco/static_grasp/scene_camera_m3c_"+LaunchConfiguration("scene_fixture").perform(context)+".xml"))}).toxml()
    semantic = ET.parse(moveit/"config/panda.srdf").getroot()
    for joint in semantic.findall("virtual_joint"): semantic.remove(joint)
    # Make the grasp reference the IK tip. The hand is fixed to link8; the
    # group still has exactly the same seven active arm joints.
    for group in semantic.findall("group"):
        if group.get("name") == "panda_arm": group.find("chain").set("tip_link", "panda_hand")
    model = {"robot_description": ParameterValue(robot, value_type=str),
             "robot_description_semantic": ParameterValue(ET.tostring(semantic, encoding="unicode"), value_type=str),
             "robot_description_kinematics": yaml.safe_load((moveit/"config/kinematics.yaml").read_text()),
             "robot_description_planning": yaml.safe_load((moveit/"config/joint_limits.yaml").read_text()),
             "use_sim_time": True}
    servo = yaml.safe_load((control/"config/servo.yaml").read_text())
    servo["moveit_servo"]["apply_twist_commands_about_ee_frame"] = True
    smoother = {"update_period": float(servo["moveit_servo"]["publish_period"]),
                "planning_group_name": servo["moveit_servo"]["move_group_name"]}
    planning = {"planning_pipelines": ["ompl"], "default_planning_pipeline": "ompl",
                "ompl": yaml.safe_load((moveit/"config/ompl_planning.yaml").read_text()),
                "allow_trajectory_execution": False, "moveit_manage_controllers": False,
                "moveit_controller_manager": "sorting_control/PlanningOnlyControllers",
                "disable_capabilities": "move_group/MoveGroupExecuteTrajectoryAction move_group/MoveGroupMoveAction move_group/MoveGroupExecuteService",
                "publish_planning_scene": True, "publish_geometry_updates": True,
                "publish_state_updates": True, "publish_transforms_updates": True,
                "monitor_dynamics": True}
    controllers = str(description/"config/panda_controllers.yaml")
    evaluator = ParameterValue(LaunchConfiguration("enable_evaluator"), value_type=bool)
    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[model], output="screen"),
        Node(package="mujoco_ros2_control", executable="ros2_control_node",
             parameters=[controllers, str(simulation/"config/camera_plugins_m3b.yaml"), {"use_sim_time": True, "mujoco_plugins.camera_evidence.enable_evaluator": evaluator}],
             output="screen", on_exit=Shutdown()),
        Node(package="controller_manager", executable="spawner",
             arguments=["joint_state_broadcaster", "panda_arm_controller", "panda_hand_controller",
                        "--controller-manager", "/controller_manager", "--param-file", controllers,
                        "--param-file", str(control/"config/ownership_controllers.yaml"), "--activate-as-group",
                        "--controller-manager-timeout", "60"], output="screen"),
        Node(package="moveit_servo", executable="servo_node", name="sorting_servo",
             parameters=[servo, smoother, model], output="screen"),
        # Keep GetPlanningScene services separate from the Servo service.
        Node(package="moveit_ros_move_group", executable="move_group", namespace="m2b_planner",
             parameters=[model, planning], remappings=[("joint_states", "/joint_states"),
                                                       ("planning_scene", "/planning_scene")], output="screen"),
        Node(package="sorting_control", executable="arm_command_gate.py",
             parameters=[str(control/"config/ownership.yaml"), {"planning_scene_service": "/get_planning_scene"}], output="screen"),

    ]
    nodes += [
        Node(package="sorting_perception", executable="handle_localizer.py",
             parameters=[{"use_sim_time": True, "camera_config": "cameras_m3c.yaml", "localization_config": "m3b_localization.yaml"}], output="screen"),
        Node(package="sorting_perception", executable="motion_filter.py", parameters=[{"use_sim_time": True}], output="screen"),
        Node(package="sorting_control", executable="camera_scene.py", parameters=[{"use_sim_time": True}], output="screen"),
        Node(package="sorting_perception", executable="fruit_classifier.py", parameters=[{"use_sim_time": True, "model_directory": LaunchConfiguration("model_directory")}], output="screen"),
    ]
    return nodes


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("scene_fixture", default_value="red_clear", choices=["red_clear","green_clear","red_covered","green_covered"]),
                              DeclareLaunchArgument("model_directory", default_value=str(Path(get_package_share_directory("sorting_perception"))/"models/m3c_baseline")),
                              DeclareLaunchArgument("headless", default_value="true"),
                              DeclareLaunchArgument("enable_evaluator", default_value="false",
                                                    description="Publish capture truth for separate debugging; the classifier has no truth input"),
                              OpaqueFunction(function=setup)])
