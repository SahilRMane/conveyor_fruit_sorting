"""Moving inspection plus camera-derived physical grasp and placement geometry."""
import importlib.util
from pathlib import Path
import sys
from ament_index_python.packages import get_package_prefix,get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    sys.path.insert(0,str(Path(get_package_prefix('sorting_supervisor'))/'lib/sorting_supervisor'))
    from m4d_common import checked_preflight
    folder,_=checked_preflight(LaunchConfiguration('preflight_dir').perform(context))
    source=Path(get_package_share_directory('sorting_supervisor'))/'launch/panda_m4c.launch.py'
    spec=importlib.util.spec_from_file_location('m4d_inspection_launch',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=context.launch_configurations['preflight_dir']
    try:
        context.launch_configurations['preflight_dir']=str(folder/'inspection')
        nodes=module.setup(context)
    finally:context.launch_configurations['preflight_dir']=original
    for i,node in enumerate(nodes):
        if node.node_executable=='publish_static_fixtures.py':
            nodes[i]=Node(package='sorting_control',executable='camera_sort_scene.py',output='screen',
                parameters=[{'use_sim_time':True}],remappings=[('/sorting/targets','/sorting/m4c/targets')])
        elif node.node_executable=='motion_filter.py':
            nodes[i]=Node(package='sorting_supervisor',executable='m4d_filter.py',output='screen',
                parameters=[{'use_sim_time':True,'timing_trace':True,'filter_config':str(folder/'inspection/filter.yaml')}],
                remappings=[('/sorting/targets','/sorting/m4c/targets'),
                            ('/sorting/perception/filter_status','/sorting/m4c/filter_status'),
                            ('/sorting/perception/filter_timing','/sorting/m4c/filter_timing')])
        elif node.node_executable=='handle_localizer.py':
            nodes[i]=Node(package='sorting_perception',executable='handle_localizer.py',output='screen',
                parameters=[{'use_sim_time':True,'timing_trace':True,'process_period_s':.002,
                             'camera_config':str(folder/'inspection/cameras.yaml'),
                             'localization_config':'m3b_localization.yaml'}])
    return nodes


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('preflight_dir'),
        DeclareLaunchArgument('fixture',default_value='red_partial',choices=[color+'_'+kind
            for color in ('red','green') for kind in ('clear','partial','covered')]),
        DeclareLaunchArgument('arrival',default_value='upstream',choices=['upstream','late']),
        DeclareLaunchArgument('headless',default_value='true'),OpaqueFunction(function=setup)])
