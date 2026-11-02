#!/usr/bin/env python3
"""Update tray collision boxes in both planning scenes for explicit oracle debug."""
import copy
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from moveit_msgs.msg import PlanningScene
from sorting_interfaces.msg import OracleTarget
from publish_static_fixtures import scene_message
from prediction_core import rotation
from m2b_common import config, observation, set_pose
from ownership_core import require

TRAY_IDS = {"tray_base", "tray_wall_x_minus", "tray_wall_x_plus", "tray_wall_y_minus", "tray_wall_y_plus",
            "handle_post_minus", "handle_post_plus", "handle_bar"}


def update_scene(template, target, settings):
    scene = copy.deepcopy(template)
    r = rotation(target.quaternion)
    tray = np.asarray(target.position)-r@np.asarray(settings["handle_offset_m"])
    for obj in scene.world.collision_objects:
        if obj.id in TRAY_IDS:
            primitive_pose = obj.primitive_poses[0]
            nominal = np.array([primitive_pose.position.x, primitive_pose.position.y, primitive_pose.position.z])
            set_pose(primitive_pose, tray+r@(nominal-np.array([0.55, 0.0, 0.307])), target.quaternion)
    return scene


class Publisher(Node):
    def __init__(self):
        super().__init__("m2b_oracle_scene", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.declare_parameter("enable_oracle", False)
        require(self.get_parameter("enable_oracle").value is True, "Oracle scene requires explicit enable_oracle=true")
        self.settings = config(); self.template = scene_message(); self.latest = None; self.received = 0.0
        self.publisher = self.create_publisher(PlanningScene, "/planning_scene", 1)
        self.create_subscription(OracleTarget, "/sorting/debug/oracle_target", self.receive, 1)
        self.create_timer(0.1, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def receive(self, message):
        if message.oracle_only:
            self.latest = observation(message); self.received = time.monotonic()

    def tick(self):
        if self.latest is not None:
            age = (self.get_clock().now().nanoseconds-self.latest.stamp_ns)/1e9
            if not (-0.02 <= age < self.settings["maximum_observation_age_s"]
                    and time.monotonic()-self.received < self.settings["maximum_source_wall_age_s"]): return
            self.publisher.publish(update_scene(self.template, self.latest, self.settings))


def main():
    rclpy.init(); node = Publisher()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException): pass
    finally: node.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__": main()
