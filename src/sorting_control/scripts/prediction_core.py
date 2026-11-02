"""Geometry and constant-velocity prediction, independent of ROS."""
from dataclasses import dataclass
import math
import numpy as np
from ownership_core import require


@dataclass(frozen=True)
class Observation:
    episode: int
    sequence: int
    stamp_ns: int
    position: tuple
    velocity: tuple
    quaternion: tuple  # x, y, z, w
    frame: str = "world"


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    require(q.shape == (4,) and np.all(np.isfinite(q)), "Invalid quaternion")
    require(abs(np.linalg.norm(q)-1.0) < 0.001, "Quaternion must have unit norm")
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def orientation_error(target, current):
    matrix = rotation(target)@rotation(current).T
    # These checks use only small errors, away from a pi rotation.
    return 0.5*np.array([matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0], matrix[1, 0]-matrix[0, 1]])


def angle_error(a, b):
    return math.acos(float(np.clip((np.trace(rotation(a)@rotation(b).T)-1.0)/2.0, -1.0, 1.0)))


def handle_position(tray_position, tray_quaternion, offset):
    return np.asarray(tray_position, dtype=float)+rotation(tray_quaternion)@np.asarray(offset, dtype=float)


def validate_observation(observation, config):
    require(observation.frame == config["world_frame"], "Target frame must be world")
    require(observation.stamp_ns > 0 and observation.episode >= 0 and observation.sequence > 0, "Invalid target identity/time")
    p, v = np.asarray(observation.position), np.asarray(observation.velocity)
    require(p.shape == (3,) and v.shape == (3,) and np.all(np.isfinite(p)) and np.all(np.isfinite(v)), "Non-finite target state")
    r = rotation(observation.quaternion)
    require(r[2, 2] > math.cos(0.02) and abs(v[1]) < 0.002 and abs(v[2]) < 0.002,
            "Oracle predictor requires a flat tray on a straight belt")
    require(-0.002 <= v[0] <= 0.105, "Oracle target speed outside debug range")


class ObservationCache:
    def __init__(self, config): self.config, self.latest = config, None
    def accept(self, observation):
        validate_observation(observation, self.config)
        old = self.latest
        if old is not None:
            require(observation.episode >= old.episode, "Old episode")
            if observation.episode == old.episode:
                require(observation.sequence > old.sequence and observation.stamp_ns > old.stamp_ns, "Out-of-order observation")
        self.latest = observation


def predict(observation, now_ns, target_ns, config):
    validate_observation(observation, config)
    age = (now_ns-observation.stamp_ns)/1e9
    horizon = (target_ns-now_ns)/1e9
    require(-0.02 <= age < config["maximum_observation_age_s"], "Stale or future observation")
    require(0 <= horizon <= config["maximum_prediction_lead_s"], "Invalid prediction horizon")
    position = np.asarray(observation.position)+np.asarray(observation.velocity)*((target_ns-observation.stamp_ns)/1e9)
    entry, exit_ = config["safe_x_interval_m"]
    require(entry <= position[0] <= exit_ and abs(position[1]) <= config["safe_y_limit_m"], "Predicted target outside interception window")
    vx = observation.velocity[0]
    # Use -1 for an unbounded time in a stationary scene. JSON must stay finite.
    remaining = (exit_-position[0])/vx if vx > 1e-6 else -1.0
    return {"position": position.tolist(), "quaternion": list(observation.quaternion),
            "observation_age_s": age, "horizon_s": horizon, "remaining_window_s": remaining,
            "target_ns": target_ns, "observation_ns": observation.stamp_ns}


def predict_grasp(observation, now_ns, approach_duration_s, config):
    require(math.isfinite(approach_duration_s) and approach_duration_s >= 0, "Invalid approach duration")
    target = now_ns+round((approach_duration_s+config["gripper_closure_estimate_s"])*1e9)
    return predict(observation, now_ns, target, config)


def limit_vector(vector, limit):
    value = np.asarray(vector, dtype=float)
    require(value.shape == (3,) and np.all(np.isfinite(value)) and limit > 0, "Invalid bounded vector")
    length = np.linalg.norm(value)
    return value*(min(1.0, limit/length) if length else 1.0)


def tracking_twist(target_position, target_velocity, target_quaternion, position, quaternion, config):
    linear = np.asarray(target_velocity)+config["position_gain_per_s"]*(np.asarray(target_position)-np.asarray(position))
    angular = config["orientation_gain_per_s"]*orientation_error(target_quaternion, quaternion)
    return (limit_vector(linear, config["maximum_linear_speed_m_s"]),
            limit_vector(angular, config["maximum_angular_speed_rad_s"]))
