#!/usr/bin/env python3
"""Test MoveIt pregrasp planning, Cartesian frames and moving oracle prediction."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from std_srvs.srv import SetBool
from moveit_msgs.srv import GetPositionFK
from ament_index_python.packages import get_package_share_directory
from rosidl_runtime_py.convert import message_to_ordereddict
from m2b_common import xyz, xyzw
from m2b_runtime import M2BProbe, robot_state
from prediction_core import angle_error, predict_grasp
from ownership_core import NAMES, require, ns


def axis_checks(probe):
    settings = probe.settings; results = []
    probe.acquire_mode("SERVO", seconds=20.0, kind=1)
    position, quaternion = probe.hand_pose()
    probe.command_function = lambda now: (position, [0.0]*3, quaternion)
    probe.input_enabled = True; probe.wait_sim(0.5)
    for axis in range(3):
        position, quaternion = probe.hand_pose(); start = probe.now_ns()
        direction = np.eye(3)[axis]; speed = settings["servo_axis_speed_m_s"]; duration = settings["servo_axis_duration_s"]
        def reference(now, p=position.copy(), q=quaternion[:], t=start, d=direction):
            elapsed = min(duration, max(0.0, (now-t)/1e9))
            return p+d*speed*elapsed, d*speed if elapsed < duration else [0.0]*3, q
        probe.command_function = reference
        forwarded = probe.owner["stats"]["servo_output_forwarded"]
        probe.wait_sim(duration+0.3)
        current, rotation = probe.hand_pose(); displacement = current-position
        expected = direction*speed*duration
        require(probe.owner["owner"] == "SERVO", "Owner changed during Cartesian axis check: "+probe.owner.get("reason", ""))
        require(displacement[axis] > 0.004, "Cartesian motion did not have the required positive world direction")
        require(np.linalg.norm(displacement-expected) < 0.006, "Cartesian displacement error exceeded 6 mm")
        require(angle_error(rotation, quaternion) < 0.04, "Cartesian translation changed orientation excessively")
        count = probe.owner["stats"]["servo_output_forwarded"]-forwarded
        require(count > 10, "Cartesian check had too few forwarded Servo outputs")
        results.append({"world_axis": "xyz"[axis], "expected_displacement_m": expected.tolist(),
                        "measured_displacement_m": displacement.tolist(), "output_forwarded": count})
    # Verify a world-z angular command as well as linear directions.
    position, quaternion = probe.hand_pose(); start = probe.now_ns()
    omega = settings["servo_yaw_speed_rad_s"]; duration = settings["servo_axis_duration_s"]
    def yaw_reference(now):
        a = min(duration, max(0.0, (now-start)/1e9))*omega
        x, y, z, w = quaternion; c, s = math.cos(a/2), math.sin(a/2)
        target = [c*x-s*y, c*y+s*x, c*z+s*w, c*w-s*z]
        return position, [0.0]*3, target
    probe.command_function = yaw_reference; probe.wait_sim(duration+0.4)
    current, rotated = probe.hand_pose(); target_quat = yaw_reference(probe.now_ns())[2]
    from prediction_core import orientation_error
    observed_yaw = float(orientation_error(rotated, quaternion)[2])
    require(observed_yaw > 0.003, "Angular motion did not have the positive world-z direction")
    require(angle_error(rotated, target_quat) < 0.015 and np.linalg.norm(current-position) < 0.006,
            "Angular frame check exceeded its pose tolerance")
    results.append({"world_axis": "yaw_z", "expected_angle_rad": omega*duration, "observed_angle_rad": observed_yaw})
    probe.input_enabled = False; probe.acquire_mode("HOLD")
    return results


def prediction_checks(probe, folder):
    require(probe.owner["owner"] == "HOLD", "Prediction test requires HOLD")
    probe.allow_belt = True; probe.belt_speed = probe.settings["belt_speed_m_s"]; probe.send_belt()
    probe.wait_sim(1.5)
    require(0.028 < probe.oracle.velocity[0] < 0.032, "Conveyor did not reach the 0.03 m/s test speed")
    forecasts = []
    for _ in range(probe.settings["prediction_trials"]):
        m = copy.deepcopy(probe.prediction)
        require(m and m.oracle_only and m.valid, "Oracle forecast is invalid: "+(m.reason if m else "missing"))
        require(m.observation_age_s < probe.settings["maximum_observation_age_s"] and m.horizon_s > 0,
                "Forecast did not retain capture age and future horizon")
        forecasts.append(m)
        probe.wait_sim(probe.settings["prediction_trial_spacing_s"])
    last_target = max(ns(m.target_stamp) for m in forecasts)
    probe.until(lambda: probe.oracle.stamp_ns > last_target)
    samples = probe.oracle_history
    measured = []
    for forecast in forecasts:
        stamp = ns(forecast.target_stamp)
        pairs = [(a, b) for a, b in zip(samples, samples[1:]) if a.stamp_ns <= stamp <= b.stamp_ns
                 and a.episode == b.episode == forecast.episode_id]
        require(pairs, "No future truth samples bracket the forecast timestamp")
        a, b = pairs[0]; fraction = (stamp-a.stamp_ns)/(b.stamp_ns-a.stamp_ns)
        actual = np.asarray(a.position)+fraction*(np.asarray(b.position)-np.asarray(a.position))
        error = float(np.linalg.norm(actual-np.asarray(xyz(forecast.handle_pose.position))))
        require(error < probe.settings["prediction_error_tolerance_m"], "Oracle future-position error exceeded 3 mm")
        measured.append({"observation_stamp_ns": ns(forecast.observation_stamp), "decision_stamp_ns": ns(forecast.header.stamp),
                         "target_stamp_ns": stamp, "position_error_m": error, "actual_handle_xyz": actual.tolist()})
    (folder/"forecasts.json").write_text(json.dumps([message_to_ordereddict(m) for m in forecasts], indent=2)+"\n")
    probe.belt_speed = 0.0; probe.send_belt(); probe.wait_sim(0.8)
    require(abs(probe.oracle.velocity[0]) < 0.002 and abs(probe.contact["external_force_x_N"]) < 1e-6, "Conveyor stop was not verified")
    probe.reference = probe.contact.copy(); probe.allow_belt = False
    probe.scene_check()
    source = probe.node.create_client(SetBool, "/sorting/debug/oracle_source/set_enabled")
    require(probe.call(source, SetBool.Request(data=False)).success, "Could not pause debug observations")
    probe.until(lambda: probe.prediction is not None and not probe.prediction.valid)
    stale_reason = probe.prediction.reason
    require("Stale" in stale_reason or "timeout" in stale_reason, "Missing source did not invalidate prediction")
    require(probe.owner["owner"] == "HOLD", "Arm moved during stale-source check")
    require(probe.call(source, SetBool.Request(data=True)).success, "Could not resume debug observations")
    probe.until(lambda: probe.prediction.valid)
    return {"speed_m_s": probe.settings["belt_speed_m_s"], "trials": measured,
            "maximum_prediction_error_m": max(m["position_error_m"] for m in measured),
            "stale_source_reason": stale_reason, "source_recovered": True, "arm_owner": probe.owner["owner"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-oracle", action="store_true", help="Explicitly permit simulator-truth debug input")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.allow_oracle: parser.error("M2B is a truth-only debug checkpoint; pass --allow-oracle")
    folder = args.output_dir or Path("results")/f"M2B-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True, exist_ok=False)
    share = Path(get_package_share_directory("sorting_control")); source = Path(__file__).resolve().parent
    files = [share/"config"/n for n in ("m2b.yaml", "servo.yaml", "ownership.yaml", "ownership_controllers.yaml", "fixtures.yaml")]
    files += [source/n for n in ("m2b_runtime.py", "prediction_core.py", "test_m2b.py", "arm_command_gate.py", "ownership_core.py",
                                "oracle_target_source.py", "oracle_predictor.py", "publish_oracle_scene.py")]
    files += [share/"launch/panda_m2b.launch.py", Path(get_package_share_directory("sorting_description"))/"mujoco/static_grasp/scene_tracking.xml"]
    description = Path(get_package_share_directory("sorting_description"))
    files += [description/"mujoco/static_grasp"/n for n in ("panda_static.xml", "props.xml", "bins.xml")]
    files += [description/"urdf/panda_mujoco.urdf.xacro"]
    (folder/"manifest.json").write_text(json.dumps({"started_at": datetime.now().astimezone().isoformat(),
         "oracle_only": True, "physical_grasp": False, "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}, indent=2)+"\n")
    report = {"status": "FAIL", "acceptance_passed": False, "oracle_only": True,
              "moving_grasp": False, "mode": "pregrasp_cartesian_frames_and_oracle_prediction", "checks": {}}
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("test_m2b", parameter_overrides=[Parameter("use_sim_time", value=True)])
    probe = M2BProbe(node)
    try:
        probe.startup(); report["checks"]["planner_execution_disabled"] = "PASS"
        report["initial_frames"] = probe.frame_check(); report["checks"]["frame_agreement"] = "PASS"
        target = np.asarray(probe.oracle.position)+np.array([0.0, 0.0, probe.settings["hand_above_handle_m"]+probe.settings["pregrasp_clearance_m"]])
        quaternion = probe.settings["grasp_quaternion_xyzw"]
        report["pregrasp_plan"] = probe.execute_plan(target, quaternion, folder, "pregrasp")
        report["checks"]["moveit_pregrasp_plan_and_execution"] = "PASS"
        prediction = predict_grasp(probe.oracle, probe.now_ns(), report["pregrasp_plan"]["duration_sim_s"], probe.settings)
        report["approach_and_closure_prediction"] = prediction
        report["pregrasp_refinement"] = probe.refine_pose(target, quaternion)
        report["checks"]["pregrasp_pose_refinement"] = "PASS"
        report["cartesian_checks"] = axis_checks(probe); report["checks"]["cartesian_world_axes"] = "PASS"
        report["after_cartesian_frames"] = probe.frame_check()
        report["moving_prediction"] = prediction_checks(probe, folder)
        report["checks"]["future_oracle_prediction"] = "PASS"
        report["checks"]["stale_oracle_rejected_and_recovered"] = "PASS"
        home = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
        q = probe.q.copy(); q.update(dict(zip(NAMES, home)))
        request = GetPositionFK.Request(); request.header.frame_id = "world"; request.fk_link_names = ["panda_hand"]
        request.robot_state = robot_state(q)
        fk = probe.call(probe.fk_client, request); require(fk.error_code.val == 1 and fk.pose_stamped, "Home FK failed")
        report["return_home_plan"] = probe.execute_plan(xyz(fk.pose_stamped[0].pose.position), xyzw(fk.pose_stamped[0].pose.orientation), folder, "home", joint_goal=home)
        require(probe.owner["owner"] == "HOLD" and not probe.owner["busy"], "Final arm owner is not HOLD")
        require(probe.owner["stats"]["overlap_violations"] == 0, "Command-source overlap was detected")
        require(abs(probe.contact["external_force_x_N"]) < 1e-6, "Final conveyor force is not zero")
        report["checks"]["planned_home_and_final_hold"] = "PASS"
        require(len(report["checks"]) == 8 and all(v == "PASS" for v in report["checks"].values()), "M2B checks are incomplete")
        report.update(status="PASS", acceptance_passed=True, passed_checks=8, final_owner_state=probe.owner)
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = str(error) or "Interrupted by user"; report["owner_at_failure"] = probe.owner.copy()
        probe.input_enabled = False; probe.guarded = False; probe.belt_speed = 0.0; probe.send_belt()
        try: probe.safe_hold(); report["failure_hold"] = "PASS"
        except Exception as stop_error: report["failure_hold_error"] = str(stop_error)
    finally:
        probe.input_enabled = False; probe.guarded = False; probe.belt_speed = 0.0
        end = time.monotonic()+0.6
        while time.monotonic() < end:
            probe.send_belt(); rclpy.spin_once(node, timeout_sec=0.02)
        report.update(output_directory=str(folder.resolve()), last_owner_state=probe.owner, last_contact_state=probe.contact)
        (folder/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        (folder/"events.json").write_text(json.dumps(probe.events, indent=2, allow_nan=False)+"\n")
        node.destroy_node(); rclpy.try_shutdown()
    print(json.dumps(report, indent=2)); return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
