"""Resolve the existing installed project libraries. Source ROS before use."""
from pathlib import Path
import sys


def prepare():
    from ament_index_python.packages import get_package_prefix
    for package in ('sorting_control', 'sorting_perception', 'sorting_supervisor'):
        folder = Path(get_package_prefix(package)) / 'lib' / package
        sys.path.insert(0, str(folder))


def write_json(path, value):
    import json
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
