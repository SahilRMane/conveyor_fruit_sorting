"""M4A stationary, open-hand inspection. No pickup executor is launched."""
from pathlib import Path
import sys
import yaml
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def setup(context):
    sys.path.insert(0, str(Path(get_package_prefix('sorting_supervisor'))/'lib/sorting_supervisor'))
    from m4a_common import checked_preflight
    folder, _ = checked_preflight(LaunchConfiguration('preflight_dir').perform(context))
    label = LaunchConfiguration('fixture').perform(context)
    return build_nodes(folder, label, LaunchConfiguration('headless').perform(context))


def build_nodes(folder, label, headless):
    from m4a_common import robot_parameters, share
    from m3d_parameters import planning_parameters
    model = robot_parameters(folder/f'scene-{label}.xml', headless)
    for name in ('robot_description', 'robot_description_semantic'):
        model[name] = ParameterValue(model[name], value_type=str)
    control, simulation, description = map(share, ('sorting_control', 'sorting_simulation', 'sorting_description'))
    servo = yaml.safe_load((control/'config/servo.yaml').read_text())
    servo['moveit_servo']['apply_twist_commands_about_ee_frame'] = True
    servo['moveit_servo']['smoothing_filter_plugin_name'] = 'sorting_control/PrecisionAccelerationLimitedPlugin'
    smoother = dict(update_period=float(servo['moveit_servo']['publish_period']),
                    planning_group_name=servo['moveit_servo']['move_group_name'])
    controllers = str(description/'config/panda_controllers.yaml')
    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[model, {'publish_frequency': 100.0}], output='screen'),
        Node(package='mujoco_ros2_control', executable='ros2_control_node',
             parameters=[controllers, str(control/'config/wrist_sensor.yaml'),
                         str(simulation/'config/camera_plugins_m3d.yaml'),
                         {'use_sim_time': True, 'mujoco_plugins.camera_evidence.enable_evaluator': True}],
             output='screen', on_exit=Shutdown()),
        Node(package='controller_manager', executable='spawner',
             arguments=['joint_state_broadcaster', 'panda_arm_controller', 'panda_hand_controller',
                        'panda_hand_wrench_broadcaster', '--controller-manager', '/controller_manager',
                        '--param-file', controllers, '--param-file', str(control/'config/wrist_sensor.yaml'),
                        '--param-file', str(control/'config/ownership_controllers.yaml'),
                        '--activate-as-group', '--controller-manager-timeout', '60'], output='screen'),
        Node(package='moveit_servo', executable='servo_node', name='sorting_servo',
             parameters=[servo, smoother, model, {'robot_description_kinematics.panda_arm.epsilon': 1e-7}], output='screen'),
        Node(package='moveit_ros_move_group', executable='move_group', namespace='m2b_planner',
             parameters=[model, planning_parameters()],
             remappings=[('joint_states', '/joint_states'), ('planning_scene', '/planning_scene')], output='screen'),
        Node(package='sorting_control', executable='publish_static_fixtures.py',
             parameters=[{'use_sim_time': True}], output='screen'),
        Node(package='sorting_control', executable='arm_command_gate.py',
             parameters=[str(control/'config/ownership.yaml'),
                         {'planning_scene_service': '/get_planning_scene', 'preserve_hold_reference': True}], output='screen'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('preflight_dir'),
        DeclareLaunchArgument('fixture', default_value='red', choices=['red', 'green']),
        DeclareLaunchArgument('headless', default_value='false'),
        OpaqueFunction(function=setup),
    ])
