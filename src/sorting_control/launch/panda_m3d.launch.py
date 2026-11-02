"""M3D class-driven moving pickup, bin placement and low-confidence pass."""
from pathlib import Path
import yaml
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
    from ament_index_python.packages import get_package_prefix
    import sys
    sys.path.insert(0,str(Path(get_package_prefix("sorting_control"))/"lib/sorting_control"))
    from m3d_parameters import model_parameters,planning_parameters
    model=model_parameters(LaunchConfiguration("headless").perform(context),LaunchConfiguration("scene_fixture").perform(context))
    for name in ("robot_description","robot_description_semantic"):
        model[name]=ParameterValue(model[name],value_type=str)
    servo = yaml.safe_load((control/"config/servo.yaml").read_text())
    servo["moveit_servo"]["apply_twist_commands_about_ee_frame"] = True
    servo["moveit_servo"]["smoothing_filter_plugin_name"] = "sorting_control/PrecisionAccelerationLimitedPlugin"
    smoother = {"update_period": float(servo["moveit_servo"]["publish_period"]),
                "planning_group_name": servo["moveit_servo"]["move_group_name"]}
    planning=planning_parameters()
    controllers = str(description/"config/panda_controllers.yaml")
    evaluator = ParameterValue(LaunchConfiguration("enable_evaluator"), value_type=bool)
    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             # Wrist samples must have a capture-time TF within 50 ms.
             # The default 20 Hz TF limit leaves no transport margin.
             parameters=[model, {"publish_frequency": 100.0}], output="screen"),
        Node(package="mujoco_ros2_control", executable="ros2_control_node",
             parameters=[controllers, str(control/"config/wrist_sensor.yaml"), str(simulation/"config/camera_plugins_m3d.yaml"), {"use_sim_time": True, "mujoco_plugins.camera_evidence.enable_evaluator": evaluator}],
             output="screen", on_exit=Shutdown()),
        Node(package="controller_manager", executable="spawner",
             arguments=["joint_state_broadcaster", "panda_arm_controller", "panda_hand_controller",
                        "panda_hand_wrench_broadcaster",
                        "--controller-manager", "/controller_manager", "--param-file", controllers,
                        "--param-file", str(control/"config/wrist_sensor.yaml"),
                        "--param-file", str(control/"config/ownership_controllers.yaml"), "--activate-as-group",
                        "--controller-manager-timeout", "60"], output="screen"),
        Node(package="moveit_servo", executable="servo_node", name="sorting_servo",
             # A 0.255 mm/s descent gives a 2.55 um IK target per 10 ms.
             # KDL's default 10 um tolerance can accept the unchanged seed.
             parameters=[servo, smoother, model,
                         {"robot_description_kinematics.panda_arm.epsilon": 1.0e-7}], output="screen"),
        # Keep GetPlanningScene services separate from the Servo service.
        Node(package="moveit_ros_move_group", executable="move_group", namespace="m2b_planner",
             parameters=[model, planning], remappings=[("joint_states", "/joint_states"),
                                                       ("planning_scene", "/planning_scene")], output="screen"),
        Node(package="sorting_control", executable="arm_command_gate.py",
             parameters=[str(control/"config/ownership.yaml"), {"planning_scene_service": "/get_planning_scene", "preserve_hold_reference": True}], output="screen"),

    ]
    nodes += [
        Node(package="sorting_perception", executable="handle_localizer.py",
             parameters=[{"use_sim_time": True, "camera_config": "cameras_m3d.yaml", "localization_config": "m3b_localization.yaml"}], output="screen"),
        Node(package="sorting_perception", executable="motion_filter.py", parameters=[{"use_sim_time": True}], output="screen"),
        Node(package="sorting_control", executable="camera_sort_scene.py", parameters=[{"use_sim_time": True}], output="screen"),
        Node(package="sorting_perception", executable="fruit_classifier.py", parameters=[{"use_sim_time": True, "model_directory": LaunchConfiguration("model_directory")}], output="screen"),
    ]
    nodes += [Node(package="sorting_control", executable="camera_sort.py", parameters=[{"use_sim_time": True, "model_directory": LaunchConfiguration("model_directory")}], output="screen")]
    return nodes


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("scene_fixture", default_value="red_clear", choices=["red_clear","green_clear","red_covered","green_covered"]),
                              DeclareLaunchArgument("model_directory", default_value=str(Path(get_package_share_directory("sorting_perception"))/"models/m3c_baseline")),
                              DeclareLaunchArgument("headless", default_value="true"),
                              DeclareLaunchArgument("enable_evaluator", default_value="true",
                                                    description="Publish capture truth for separate debugging; the classifier has no truth input"),
                              OpaqueFunction(function=setup)])
