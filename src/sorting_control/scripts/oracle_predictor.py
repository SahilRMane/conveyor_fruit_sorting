#!/usr/bin/env python3
"""Predict a named oracle target while retaining its original capture timestamp."""
import time
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import ExternalShutdownException
from sorting_interfaces.msg import OracleTarget, OraclePrediction
from prediction_core import ObservationCache, predict
from m2b_common import config, observation, set_pose, timestamp
from ownership_core import require


class Predictor(Node):
    def __init__(self):
        super().__init__("m2b_oracle_predictor", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.declare_parameter("enable_oracle", False)
        require(self.get_parameter("enable_oracle").value is True, "Oracle predictor requires explicit enable_oracle=true")
        self.settings = config(); self.cache = ObservationCache(self.settings); self.received = 0.0
        self.last_error = "No observation"
        self.publisher = self.create_publisher(OraclePrediction, "/sorting/debug/oracle_prediction", 1)
        self.create_subscription(OracleTarget, "/sorting/debug/oracle_target", self.receive, 1)
        self.create_timer(0.02, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def receive(self, message):
        try:
            require(message.oracle_only, "This adapter accepts oracle targets only")
            self.cache.accept(observation(message)); self.received = time.monotonic(); self.last_error = ""
        except Exception as error: self.last_error = str(error)

    def tick(self):
        m = OraclePrediction(); now = self.get_clock().now().nanoseconds
        m.header.stamp = timestamp(now); m.header.frame_id = self.settings["world_frame"]; m.oracle_only = True
        target_time = now+round(self.settings["prediction_lead_s"]*1e9); m.target_stamp = timestamp(target_time)
        sample = self.cache.latest
        if sample:
            m.episode_id = sample.episode; m.sequence = sample.sequence
            m.observation_stamp = timestamp(sample.stamp_ns)
            m.observation_age_s = (now-sample.stamp_ns)/1e9
        try:
            require(sample is not None, "No observation")
            require(time.monotonic()-self.received < self.settings["maximum_source_wall_age_s"], "Oracle source wall timeout")
            result = predict(sample, now, target_time, self.settings)
            set_pose(m.handle_pose, result["position"], result["quaternion"])
            m.velocity.x, m.velocity.y, m.velocity.z = map(float, sample.velocity)
            m.observation_age_s = float(result["observation_age_s"])
            m.horizon_s = float(result["horizon_s"]); m.remaining_window_s = float(result["remaining_window_s"])
            m.valid = True; m.reason = "constant_velocity_oracle_debug"
        except Exception as error: m.valid = False; m.reason = str(error)
        self.publisher.publish(m)


def main():
    rclpy.init(); node = Predictor()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException): pass
    finally: node.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__": main()
