"""Open the M4 simulator and select the scene from the M5D suite."""

import importlib.util
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_m5d import CASES, load_case


def installed_launch(name):
    path = Path(get_package_share_directory('sorting_supervisor')) / 'launch' / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(context):
    suite = Path(LaunchConfiguration('suite_dir').perform(context)).resolve()
    name = LaunchConfiguration('case').perform(context)
    manifest, case = load_case(suite, name)
    context.launch_configurations.update(preflight_dir=manifest['preflight'], fixture=case['fixture'],
                                         arrival=case['arrival'], headless='false')
    nodes = installed_launch('panda_m4d.launch.py').setup(context)
    if name == 'right':
        replacements = installed_launch('panda_m4a.launch.py').build_nodes(Path(case['scene']).parent, 'green', 'false')
        for i in (0, 3, 4):
            if nodes[i].node_executable != replacements[i].node_executable:
                raise RuntimeError('Installed launch order differs')
            nodes[i] = replacements[i]
    return nodes


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('suite_dir'),
                              DeclareLaunchArgument('case', choices=list(CASES)), OpaqueFunction(function=setup)])
