#!/usr/bin/env python3
"""Isolated live camera/plane calibration test; no camera-driven grasp claim."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from ament_index_python.packages import get_package_share_directory,get_package_prefix
from m3a_probe import CameraProbe
from camera_core import require

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-dir",type=Path,required=True);parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--summary-only",action="store_true",help="Print a short result; keep complete JSON evidence files")
    args=parser.parse_args();preflight=json.loads((args.preflight_dir/"validation.json").read_text())
    require(preflight["status"]=="PASS","Run the M3A native preflight first")
    motion=json.loads((args.preflight_dir/"camera-motion-plan.json").read_text())
    share=Path(get_package_share_directory("sorting_perception"))
    digest=hashlib.sha256((share/"config/cameras.yaml").read_bytes()+(share/"config/m3a.yaml").read_bytes()).hexdigest()
    require(motion["config_sha256"]==digest,"Camera settings changed; repeat M3A preflight")
    folder=(args.output_dir or Path("results")/f"M3A-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}").resolve();folder.mkdir(parents=True,exist_ok=False)
    files=list((share/"config").glob("*.yaml"))+list((Path(__file__).resolve().parent).glob("*.py"))
    d=Path(get_package_share_directory("sorting_description"));s=Path(get_package_share_directory("sorting_simulation"));control=Path(get_package_share_directory("sorting_control"))
    files += [share/"launch/panda_m3a.launch.py",d/"urdf/panda_cameras.urdf.xacro",d/"urdf/panda_mujoco.urdf.xacro",
              s/"config/camera_plugins.yaml",Path(get_package_prefix("sorting_simulation"))/"lib/libsorting_camera_evidence.so"]
    files += [d/"mujoco/static_grasp"/n for n in ("scene_camera.xml","panda_camera.xml","props_camera.xml","bins.xml")]
    files += [control/"config"/n for n in ("servo.yaml","ownership.yaml","ownership_controllers.yaml","fixtures.yaml")]
    write(folder/"manifest.json",dict(started_at=datetime.now().astimezone().isoformat(),
          protocol="Camera calibration fixture, supported image localization and reset",preflight=str(args.preflight_dir.resolve()),
          camera_driven_grasp=False,classifier=False,ground_truth_usage="Evaluator only; not passed to handle_localizer",
          file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    report=dict(status="FAIL",acceptance_passed=False,full_M3_gate_passed=False,camera_driven_grasp=False,
                oracle_controller=False,checkpoint="M3A",final_reset_status="NOT_RUN",checks={})
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node=Node("test_m3a",parameter_overrides=[Parameter("use_sim_time",value=True)]);probe=CameraProbe(node)
    begin_wall=time.monotonic();begin_sim=None
    try:
        probe.until(lambda:probe.owner.get("owner")=="HOLD" and probe.owner.get("ready") and not probe.owner.get("busy") and
                    probe.epoch is not None and probe.contact and len(probe.q)>=7 and probe.camera_status and
                    probe.action.server_is_ready() and probe.service.service_is_ready())
        probe.require_fresh_launch();probe.confirm_controller();begin_sim=probe.now_ns();home=[probe.q[n] for n in probe.names]
        probe.plan["initial_state"]=probe.contact.copy();report["isolation_audit"]=probe.audit();report["checks"]["evaluator_isolation"]="PASS"
        probe.begin("stationary");stationary=probe.collect("overhead",probe.c["minimum_stationary_samples"])
        probe.collect("wrist",probe.c["minimum_wrist_samples"])
        report["stationary_maximum_planar_error_m"]=max(s["planar_error_m"] for s in stationary)
        report["checks"]["stationary_overhead_localization"]="PASS";probe.save_images(folder,"stationary")
        for view in ("overhead","wrist"):
            frames=[f for f in probe.frames if f["phase"]=="stationary" and f["view_id"]==view and f["capture_ns"]>probe.phase_begin_ns]
            stamps=sorted(set(f["capture_ns"] for f in frames));require(len(stamps)>=10,"Camera did not publish distinct capture times")
            rate=(len(stamps)-1)/((stamps[-1]-stamps[0])/1e9)
            require(rate>=probe.c["minimum_camera_rate_hz"],"Camera capture rate below 7 Hz")
            require(all(f["width"]==probe.camera_config["width"] and f["height"]==probe.camera_config["height"] and
                        f["image_std"]>5 and -0.02<=f["age_s"]<probe.c["maximum_image_age_s"] for f in frames),"Invalid, blank or stale camera frame")
            report.setdefault("camera_streams",{})[view]=dict(rate_sim_hz=rate,frames=len(stamps),maximum_age_sim_s=max(f["age_s"] for f in frames))
        report["checks"]["two_capture_time_rgb_streams"]="PASS"
        probe.begin("camera_motion");before=len(probe.samples)
        probe.move(motion["inspection_joint_positions"],probe.c["inspection_motion_duration_s"])
        moving_frames=[s for s in probe.samples[before:] if s["view_id"]=="wrist"]
        require(len(moving_frames)>=10,"No wrist frame evidence during encoder motion")
        report["checks"]["wrist_capture_time_transform"]="PASS"
        probe.begin("wrist_inspection");wrist=probe.collect("wrist",probe.c["minimum_wrist_samples"])
        report["wrist_maximum_planar_error_m"]=max(s["planar_error_m"] for s in wrist)
        report["checks"]["wrist_plane_localization"]="PASS";probe.save_images(folder,"inspection")
        probe.begin("return_home");probe.move(home,probe.c["inspection_motion_duration_s"])
        probe.begin("moving_belt");probe.belt_speed=probe.c["belt_speed_m_s"];probe.send_belt();probe.wait_sim(probe.c["belt_duration_s"])
        moving=probe.phase_samples("overhead");require(len(moving)>=probe.c["minimum_moving_samples"],"Not enough moving image-localization samples")
        seconds=np.array([s["capture_ns"] for s in moving],dtype=float);seconds=(seconds-seconds[0])/1e9
        image_speed=float(np.polyfit(seconds,np.array([s["position"][0] for s in moving]),1)[0])
        require(abs(image_speed-probe.c["belt_speed_m_s"])<0.002,"Image-derived motion does not match the 0.03 m/s scenario")
        report["moving_localization"]=dict(samples=len(moving),maximum_planar_error_m=max(s["planar_error_m"] for s in moving),
             fitted_image_velocity_m_s=image_speed,scope="Evaluator slope check; not a deployed motion filter")
        report["checks"]["supported_moving_localization"]="PASS";probe.save_images(folder,"moving")
        probe.belt_speed=0.;probe.send_belt();probe.wait_sim(0.3)
        report["final_reset_status"]="IN_PROGRESS"
        report["reset"]=probe.reset_episode();probe.collect("overhead",10);probe.collect("wrist",10)
        barrier=probe.epoch.header.stamp.sec*1_000_000_000+probe.epoch.header.stamp.nanosec
        require(all(s["episode_id"]==probe.epoch.episode_id and s["capture_ns"]>barrier for s in probe.phase_samples("overhead")+probe.phase_samples("wrist")),"Reset reused old camera evidence")
        report["checks"]["guarded_reset_and_new_camera_epoch"]="PASS"
        report["final_reset_status"]="PASS";probe.reset_stage="complete"
        report["final_isolation_audit"]=probe.audit()
        require(probe.owner["owner"]=="HOLD" and not probe.owner["busy"] and probe.owner["stats"]["overlap_violations"]==0,"Final ownership check failed")
        report["checks"]["exclusive_final_hold"]="PASS"
        report.update(status="PASS",acceptance_passed=True,final_reset_status="PASS",passed_checks=len(report["checks"]))
    except (Exception,KeyboardInterrupt) as error:
        if report["final_reset_status"]=="IN_PROGRESS":report["final_reset_status"]="FAIL"
        report["error"]=str(error) or "Interrupted by user";report["failure_handling"]=probe.stop_after_failure()
        report["passed_checks"]=sum(v=="PASS" for v in report["checks"].values())
    finally:
        probe.guarded=False;probe.belt_speed=0.;probe.send_belt()
        report.update(output_directory=str(folder),elapsed_wall_s=time.monotonic()-begin_wall,
                      elapsed_sim_s=(probe.now_ns()-begin_sim)/1e9 if begin_sim is not None else None,
                      final_owner_state=probe.owner,camera_status=probe.camera_status,
                      localization_samples=len(probe.samples),rejected_observations=len(probe.rejections))
        report["evaluator_metadata_pairs_skipped"]=sum(r["reason"]=="evaluator_camera_info_unavailable" for r in probe.rejections)
        report["real_time_factor"]=report["elapsed_sim_s"]/report["elapsed_wall_s"] if report["elapsed_sim_s"] is not None else None
        write(folder/"camera-samples.json",probe.samples);write(folder/"camera-frames.json",probe.frames)
        write(folder/"rejected-observations.json",probe.rejections);write(folder/"ownership-events.json",probe.events)
        write(folder/"pairing-diagnostics.json",probe.pairing_diagnostics)
        if probe.camera_wait_diagnostic is not None:
            report["camera_wait_diagnostic"]=probe.camera_wait_diagnostic
            write(folder/"camera-wait-diagnostic.json",probe.camera_wait_diagnostic)
        report["reset_stage"]=probe.reset_stage
        if probe.reset_report is not None:
            report.setdefault("reset",probe.reset_report)
            report["physical_reset_status"]=probe.reset_report["status"]
        if probe.failed_camera_sample is not None:
            report["camera_validation_failure"]=probe.failed_camera_sample
            write(folder/"camera-validation-failure.json",probe.failed_camera_sample)
        write(folder/"summary.json",report);node.destroy_node();rclpy.try_shutdown()
    printed=report
    if args.summary_only:
        printed={k:report[k] for k in ("checkpoint","status","acceptance_passed","passed_checks","final_reset_status",
                    "full_M3_gate_passed","checks","error","output_directory","camera_streams","evaluator_metadata_pairs_skipped") if k in report}
        if report.get("camera_wait_diagnostic") is not None:printed["camera_wait_diagnostic"]=report["camera_wait_diagnostic"]
        if report.get("camera_validation_failure") is not None:printed["camera_validation_failure"]=report["camera_validation_failure"]
        printed["reset_stage"]=report["reset_stage"]
        printed["final_owner"]=report["final_owner_state"].get("owner")
        printed["overlap_violations"]=report["final_owner_state"].get("stats",{}).get("overlap_violations")
        unresolved=[d for d in probe.pairing_diagnostics if not d.get("resolved",False) and d["phase"]==probe.phase and
                    d.get("status") in {"WAITING","TIMEOUT"}]
        if report["status"]!="PASS" and unresolved:printed["last_pairing_diagnostic"]=unresolved[-1]
    print(json.dumps(printed,indent=2));return 0 if report["status"]=="PASS" else 1

if __name__=="__main__":raise SystemExit(main())
