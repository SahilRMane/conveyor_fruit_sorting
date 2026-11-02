"""Closure proof, measured velocity and smooth grasp references; no ROS imports."""
from collections import deque
import math
import numpy as np
from ownership_core import require

class ConstraintViolation(RuntimeError): pass

def constraint_require(condition, message):
    if not condition: raise ConstraintViolation(message)

def hand_relative_position(state):
    """Tray origin in the physical hand frame; MuJoCo quaternions use w,x,y,z."""
    hand = np.asarray(state["hand_xyz"], dtype=float)
    tray = np.asarray(state["tray_xyz"], dtype=float)
    quat = np.asarray(state["hand_quat_wxyz"], dtype=float)
    require(hand.shape == (3,) and tray.shape == (3,) and quat.shape == (4,) and
            all(np.all(np.isfinite(v)) for v in (hand, tray, quat)), "Invalid physical grasp pose")
    length = float(np.linalg.norm(quat))
    require(length > 1e-12 and math.isfinite(length), "Invalid hand quaternion")
    w, x, y, z = quat/length
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    # The inverse rotation removes motion shared by the hand and tray.
    return rotation.T @ (tray-hand)


def grasp_slip_metrics(state, baseline):
    relative = hand_relative_position(state)
    delta = relative-hand_relative_position(baseline)
    world_delta = ((np.asarray(state["tray_xyz"])-np.asarray(state["hand_xyz"]))-
                   (np.asarray(baseline["tray_xyz"])-np.asarray(baseline["hand_xyz"])))
    return dict(slip_frame="panda_hand", slip_metric_version=2,
                tray_position_in_hand_m=relative.tolist(), relative_translation_delta_m=delta.tolist(),
                relative_slip_m=float(np.linalg.norm(delta)),
                world_offset_change_m=float(np.linalg.norm(world_delta)))


def require_grasp_slip(metrics, limit_m):
    require(math.isfinite(limit_m) and limit_m > 0, "Invalid physical slip limit")
    require(math.isfinite(metrics["relative_slip_m"]) and metrics["relative_slip_m"] < limit_m,
            f"Physical tray slip in hand frame exceeds {limit_m*1000:g} mm")

def summarize_trials(trials, requested, owner, minimum_successes):
    passed = sum(t["status"]=="PASS" for t in trials)
    aborted = sum(t["status"]=="ABORTED" for t in trials)
    complete = len(trials)==requested and sorted(t["trial_index"] for t in trials)==list(range(1,requested+1))
    final_reset = trials[-1].get("reset",{}).get("status","NOT_RUN") if trials else "NOT_RUN"
    violation = any(t.get("constraint_violation",False) for t in trials)
    accepted = (complete and passed>=minimum_successes and aborted==0 and not violation and final_reset=="PASS"
                and owner.get("owner")=="HOLD" and not owner.get("busy")
                and owner.get("stats",{}).get("overlap_violations",1)==0)
    return dict(successful_trials=passed,failed_trials=sum(t["status"]=="FAIL" for t in trials),aborted_trials=aborted,
                completed_trials=requested-aborted,final_reset_status=final_reset,acceptance_passed=accepted,
                status="PASS" if accepted else "FAIL",full_M2_gate_passed=bool(accepted and requested==20))


def smooth(t, duration):
    u = min(1.0, max(0.0, t/duration))
    return u*u*(3-2*u), 6*u*(1-u)/duration


def exit_deadline(observation, now_ns, settings, x_exit):
    if observation.velocity[0] > 1e-6:
        return observation.stamp_ns+round((x_exit-observation.position[0])/observation.velocity[0]*1e9)
    return now_ns+round(settings["motion_lease_s"]*1e9)


class VelocityWindow:
    def __init__(self): self.samples = deque(maxlen=5); self.velocity = None
    def observe(self, stamp_ns, position):
        p = np.asarray(position, dtype=float)
        require(p.shape == (3,) and np.all(np.isfinite(p)), "Invalid hand position")
        if self.samples and stamp_ns < self.samples[-1][0]:
            self.samples.clear(); self.velocity = None
        if not self.samples or stamp_ns > self.samples[-1][0]: self.samples.append((stamp_ns, p))
        if len(self.samples) >= 2:
            dt = (self.samples[-1][0]-self.samples[0][0])/1e9
            if 0.02 <= dt <= 0.3: self.velocity = (self.samples[-1][1]-self.samples[0][1])/dt
        return self.velocity


class ClosureWindow:
    def __init__(self, settings):
        self.settings = settings; self.first_stamp = None; self.last_stamp = 0; self.samples = 0; self.reason = []
    def observe(self, sample):
        c = self.settings; reasons = []
        error = np.asarray(sample["position_error_m"]); relative = np.asarray(sample["relative_velocity_m_s"])
        scalars = [sample["orientation_error_rad"], sample["observation_age_s"], sample["tf_age_s"], sample["remaining_s"]]
        if error.shape != (3,) or relative.shape != (3,) or not np.all(np.isfinite(error)) or not np.all(np.isfinite(relative)) or not all(math.isfinite(x) for x in scalars):
            reasons.append("nonfinite_or_missing")
        else:
            if np.linalg.norm(error[:2]) >= c["close_position_xy_m"]: reasons.append("position_xy")
            if abs(error[2]) >= c["close_position_z_m"]: reasons.append("position_z")
            if np.linalg.norm(relative) >= c["close_relative_speed_m_s"]: reasons.append("relative_velocity")
            if sample["orientation_error_rad"] >= c["close_orientation_rad"]: reasons.append("orientation")
            if not -0.02 <= sample["observation_age_s"] < c["maximum_target_age_s"]: reasons.append("observation_age")
            if not -0.02 <= sample["tf_age_s"] < c["maximum_tf_age_s"]: reasons.append("transform_age")
            reserve = c["close_action_estimate_s"]+c["clear_lift_estimate_s"]+c["deadline_margin_s"]
            if sample["remaining_s"] <= reserve: reasons.append("deadline_margin")
        if not sample["oracle_only"]: reasons.append("not_explicit_oracle_debug")
        if not sample["at_grasp_height"]: reasons.append("descent_not_finished")
        self.reason = reasons
        stamp = sample["tf_stamp_ns"]
        if reasons or stamp < self.last_stamp:
            self.first_stamp = None; self.samples = 0
        if stamp > self.last_stamp and not reasons:
            if self.first_stamp is None: self.first_stamp = stamp
            self.samples += 1
        self.last_stamp = max(self.last_stamp, stamp)
        return self.first_stamp is not None and self.samples >= c["close_minimum_samples"] and (stamp-self.first_stamp)/1e9 >= c["close_stable_span_s"]


def braking_x(start_x, speed, elapsed, duration):
    b = min(1.0, max(0.0, elapsed/duration))
    return start_x+speed*duration*(b-b*b+b*b*b/3), speed*(1-b)*(1-b)


def lower_x(start_x, speed, elapsed, duration):
    # Match belt speed before touchdown without carrying the tray to belt exit.
    ramp_duration = 0.6; start = duration-1.0
    u = min(1.0, max(0.0, (elapsed-start)/ramp_duration))
    travel = speed*(0.5*ramp_duration*u*u+max(0.0, elapsed-start-ramp_duration))
    return start_x+travel, speed*u
