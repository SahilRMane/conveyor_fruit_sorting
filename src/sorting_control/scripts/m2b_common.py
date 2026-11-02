"""Shared configuration and typed oracle-message conversion."""
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from prediction_core import Observation
from ownership_core import ns


def config():
    return yaml.safe_load((Path(get_package_share_directory("sorting_control"))/"config/m2b.yaml").read_text())


def timestamp(value):
    return Time(sec=int(value//1_000_000_000), nanosec=int(value%1_000_000_000))


def xyz(vector): return [vector.x, vector.y, vector.z]
def xyzw(quaternion): return [quaternion.x, quaternion.y, quaternion.z, quaternion.w]


def observation(message):
    return Observation(int(message.episode_id), int(message.sequence), ns(message.header.stamp),
                       tuple(xyz(message.handle_pose.position)), tuple(xyz(message.velocity)),
                       tuple(xyzw(message.handle_pose.orientation)), message.header.frame_id)


def set_pose(pose, position, quaternion):
    pose.position.x, pose.position.y, pose.position.z = map(float, position)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, quaternion)
