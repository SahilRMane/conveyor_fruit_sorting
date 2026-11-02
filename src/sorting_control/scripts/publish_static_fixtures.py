#!/usr/bin/env python3
"""Publish the known stationary fixture geometry for the M2A Servo check."""
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.msg import CollisionObject, PlanningScene
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive


def scene_message():
    path = Path(get_package_share_directory("sorting_control"))/"config/fixtures.yaml"
    data = yaml.safe_load(path.read_text())
    scene = PlanningScene(); scene.is_diff = True
    # A fixture-only update must preserve the robot state from joint feedback.
    scene.robot_state.is_diff = True
    for box in data["boxes"]:
        obj = CollisionObject(); obj.id = box["id"]; obj.header.frame_id = data["frame"]
        obj.pose.orientation.w = 1.0
        obj.operation = CollisionObject.ADD
        primitive = SolidPrimitive(); primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(v) for v in box["dimensions"]]
        pose = Pose(); pose.orientation.w = 1.0
        pose.position.x, pose.position.y, pose.position.z = map(float, box["position"])
        obj.primitives = [primitive]; obj.primitive_poses = [pose]
        scene.world.collision_objects.append(obj)
    # Only the fixed robot base may contact its mounting floor.
    # Preserve SRDF self-collision allowances when replacing the ACM.
    srdf = ET.parse(Path(get_package_share_directory("moveit_resources_panda_moveit_config"))/"config/panda.srdf").getroot()
    pairs = {(e.get("link1"), e.get("link2")) for e in srdf.findall("disable_collisions")}
    pairs.add(("static_floor", "panda_link0"))
    names = sorted({name for pair in pairs for name in pair})
    scene.allowed_collision_matrix.entry_names = names
    from moveit_msgs.msg import AllowedCollisionEntry
    scene.allowed_collision_matrix.entry_values = [
        AllowedCollisionEntry(enabled=[a == b or (a, b) in pairs or (b, a) in pairs for b in names])
        for a in names]
    return scene


def main():
    rclpy.init()
    node = Node("sorting_static_fixtures")
    publisher = node.create_publisher(PlanningScene, "/planning_scene", 1)
    scene = scene_message()
    node.create_timer(0.2, lambda: publisher.publish(scene))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__": main()
