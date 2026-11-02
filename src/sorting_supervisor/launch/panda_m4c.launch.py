"""Continuous belt and overhead tracking during bounded wrist inspection."""
import importlib.util
from pathlib import Path
import sys
from ament_index_python.packages import get_package_prefix,get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction,Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    sys.path.insert(0,str(Path(get_package_prefix('sorting_supervisor'))/'lib/sorting_supervisor'))
    from m4c_common import checked_preflight
    folder,report=checked_preflight(LaunchConfiguration('preflight_dir').perform(context))
    fixture=LaunchConfiguration('fixture').perform(context);arrival=LaunchConfiguration('arrival').perform(context)
    path=Path(report['scene_paths'][fixture+':'+arrival])
    source=Path(get_package_share_directory('sorting_supervisor'))/'launch/panda_m4a.launch.py'
    spec=importlib.util.spec_from_file_location('m4c_base_launch',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    nodes=module.build_nodes(path.parent,fixture.split('_')[0],LaunchConfiguration('headless').perform(context))
    description=Path(get_package_share_directory('sorting_description'));control=Path(get_package_share_directory('sorting_control'))
    nodes[1]=Node(package='mujoco_ros2_control',executable='ros2_control_node',output='screen',on_exit=Shutdown(),
        parameters=[str(description/'config/panda_controllers.yaml'),str(control/'config/wrist_sensor.yaml'),
                    str(folder/'camera_plugins.yaml'),{'use_sim_time':True,'mujoco_plugins.camera_evidence.enable_evaluator':True}])
    nodes += [Node(package='sorting_perception',executable='handle_localizer.py',output='screen',
                   parameters=[{'use_sim_time':True,'timing_trace':True,'camera_config':str(folder/'cameras.yaml'),'localization_config':'m3b_localization.yaml'}]),
              Node(package='sorting_perception',executable='motion_filter.py',output='screen',
                   parameters=[{'use_sim_time':True,'timing_trace':True,'filter_config':str(folder/'filter.yaml')}],
                   remappings=[('/sorting/targets','/sorting/m4c/targets'),('/sorting/perception/filter_status','/sorting/m4c/filter_status'),
                               ('/sorting/perception/filter_timing','/sorting/m4c/filter_timing')])]
    return nodes


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('preflight_dir'),
        DeclareLaunchArgument('fixture',default_value='red_partial',choices=[c+'_'+s for c in ('red','green') for s in ('clear','partial','covered')]),
        DeclareLaunchArgument('arrival',default_value='upstream',choices=['upstream','late']),
        DeclareLaunchArgument('headless',default_value='false'),OpaqueFunction(function=setup)])
