"""M4B stationary visual fixtures; no grasp executor."""
from pathlib import Path
import importlib.util
import sys
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def setup(context):
    sys.path.insert(0,str(Path(get_package_prefix('sorting_supervisor'))/'lib/sorting_supervisor'))
    from m4b_common import checked_preflight
    folder, report = checked_preflight(LaunchConfiguration('preflight_dir').perform(context))
    fixture = LaunchConfiguration('fixture').perform(context)
    path = Path(report['scene_paths'][fixture])
    # Use the passed M4A hardware, planning, Servo and ownership launch actions.
    source = Path(get_package_share_directory('sorting_supervisor'))/'launch/panda_m4a.launch.py'
    spec = importlib.util.spec_from_file_location('m4a_launch',source); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.build_nodes(path.parent, fixture.split('_')[0], LaunchConfiguration('headless').perform(context))


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('preflight_dir'),
        DeclareLaunchArgument('fixture',default_value='red_partial',choices=[c+'_'+s for c in ('red','green') for s in ('clear','partial','covered')]),
        DeclareLaunchArgument('headless',default_value='false'),OpaqueFunction(function=setup)])
