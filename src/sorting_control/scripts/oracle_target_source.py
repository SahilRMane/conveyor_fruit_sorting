#!/usr/bin/env python3
"""Explicit truth-only adapter for the M2 debug checkpoint."""
import json
import math
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from std_srvs.srv import SetBool
from sorting_interfaces.msg import OracleTarget
from prediction_core import ObservationCache, handle_position
from m2b_common import config, observation, set_pose, timestamp
from ownership_core import require


def target_message(value, sequence, settings):
    require(value.get("contact_schema_version", 0) >= 2, "Contact schema 2 is required")
    require(value["support_force_N"] > 0.5, "Oracle debug tray must remain supported")
    require(np.linalg.norm(value["tray_angular_velocity_rad_s"]) < 0.02, "Oracle debug assumes negligible tray rotation")
    quaternion = value["tray_quat_wxyz"]
    quaternion = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    position = handle_position(value["tray_xyz"], quaternion, settings["handle_offset_m"])
    m = OracleTarget(); m.header.frame_id = settings["world_frame"]
    m.header.stamp = timestamp(round(float(value["sim_time"])*1e9))
    m.episode_id = int(value["reset_seq"]); m.sequence = int(sequence); m.oracle_only = True
    set_pose(m.handle_pose, position, quaternion)
    m.velocity.x, m.velocity.y, m.velocity.z = map(float, value["tray_linear_velocity_m_s"])
    return m


class Source(Node):
    def __init__(self):
        super().__init__("m2b_oracle_target_source", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.declare_parameter("enable_oracle", False)
        require(self.get_parameter("enable_oracle").value is True, "Oracle source requires explicit enable_oracle=true")
        self.settings = config(); self.cache = ObservationCache(self.settings)
        self.pending = None; self.published_sequence = 0; self.sequence = 0
        self.publishing = True
        self.target = self.create_publisher(OracleTarget, "/sorting/debug/oracle_target", 1)
        self.diagnostic = self.create_publisher(String, "/sorting/debug/oracle_source_state", 1)
        self.create_subscription(String, "/sorting/contact_state", self.receive, 1)
        self.create_timer(0.02, self.publish_latest, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.create_service(SetBool, "/sorting/debug/oracle_source/set_enabled", self.set_enabled)

    def set_enabled(self, request, response):
        self.publishing = bool(request.data)
        response.success = True; response.message = "Debug oracle publication enabled" if self.publishing else "Debug oracle publication paused"
        return response

    def receive(self, message):
        try:
            value = json.loads(message.data)
            candidate = target_message(value, self.sequence+1, self.settings)
            self.cache.accept(observation(candidate))
            self.sequence += 1; self.pending = candidate
        except Exception as error:
            self.diagnostic.publish(String(data=json.dumps({"oracle_only": True, "valid": False, "reason": str(error)})))

    def publish_latest(self):
        if self.publishing and self.pending is not None and self.pending.sequence != self.published_sequence:
            self.target.publish(self.pending)
            self.published_sequence = self.pending.sequence


def main():
    rclpy.init(); node = Source()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException): pass
    finally: node.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__": main()
