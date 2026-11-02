#!/usr/bin/env python3
"""Camera-only moving pickup with separate physical verification and reset."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import uuid
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sorting_interfaces.srv import StartCameraPickup
from m3b_evaluator import Evaluator
from m3b_acceptance import accepted_batch

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--trials",type=int,choices=(1,20),default=1)
    parser.add_argument("--preflight-dir",type=Path,required=True);parser.add_argument("--output-dir",type=Path);parser.add_argument("--summary-only",action="store_true")
    args=parser.parse_args();preflight=json.loads((args.preflight_dir/"validation.json").read_text())
    if preflight.get("status")!="PASS" or preflight.get("checkpoint")!="M3B":parser.error("Run the M3B preflight first; an M3A report is insufficient")
    from m3b_manifest import installed_manifest
    if preflight.get("installed_manifest")!=installed_manifest():parser.error("Installed M3B files changed. Run the preflight again")
    folder=(args.output_dir or Path("results")/f"M3B-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}").resolve();folder.mkdir(parents=True,exist_ok=False)
    report=dict(checkpoint="M3B",status="FAIL",camera_only=True,prepositioned_camera_checkpoint=True,
                active_rgb_views=preflight["render_budget"]["active_views"],classifier_ready=False,
                full_M3_gate_passed=False,trials=[],final_reset_status="NOT_RUN")
    rclpy.init();node=Node("test_m3b",parameter_overrides=[Parameter("use_sim_time",value=True)]);e=Evaluator(node)
    started=time.monotonic();sim_start=None
    try:
        e.until(lambda:e.owner.get("owner")=="HOLD" and e.owner.get("ready") and e.current() is not None and e.contact and e.scene.get("valid"),45.)
        require_fresh=e.owner.get("stats",{}).get("planned_accepted",0)==0 and e.owner.get("stats",{}).get("servo_input_accepted",0)==0
        if not require_fresh:raise RuntimeError("Start a fresh M3B launch before the smoke or20-trial run")
        e.plan["initial_state"]=e.contact.copy();report["isolation_audit"]=e.audit();sim_start=e.now_ns()
        for index in range(args.trials):
            trial=dict(trial_index=index+1,status="FAIL");trial_folder=folder/f"trial-{index+1:02d}";trial_folder.mkdir()
            e.prepare_trial()
            try:
                e.until(lambda:e.current() is not None)
                b=e.current();request=StartCameraPickup.Request(episode_id=b.episode_id,track_id=b.track_id,trial_id=index+1)
                response=e.call(e.start,request)
                if not response.accepted:raise RuntimeError(response.reason)
                e.until(lambda:e.control_result is not None and e.control_result.get("trial_id")==index+1,90.)
                policy=e.control_result;write(trial_folder/"controller.json",policy)
                physical=e.verify_physical(policy);trial["physical"]=physical
                write(trial_folder/"physical-trace.json",e.history);write(trial_folder/"grip-metrics.json",e.grip_trace)
                trial["checks"]={"camera_policy_complete":"PASS","bilateral_contact_lift_hold":"PASS","exclusive_hold":"PASS"}
                report["final_reset_status"]="IN_PROGRESS";reset=e.reset_episode();trial["reset"]=reset
                trial["checks"]["fresh_camera_reset"]="PASS";report["isolation_audit"]=e.audit();trial["status"]="PASS";report["final_reset_status"]="PASS"
            except Exception as error:
                trial["error"]=str(error);trial["scene_at_failure"]=e.scene.copy();trial["filter_at_failure"]=e.filter_status.copy();trial["camera_at_failure"]=e.camera_status.copy()
                write(trial_folder/"physical-trace.json",e.history);write(trial_folder/"grip-metrics.json",e.grip_trace)
                # Only a supported miss can be reset and counted as a failure.
                # A lifted load, constraint breach or gate fault stops the batch.
                supported=(not e.control_state.get("active") and e.owner.get("owner")=="HOLD" and e.owner.get("ready") and
                    e.contact.get("support_force_N",0)>4. and e.maximum_rise<.003 and
                    all(s.get("forbidden_contacts",1)==0 and max(s["left_force_N"],s["right_force_N"])<15 for s in e.history) and
                    e.maximum_joint_speed<.5 and e.owner.get("stats",{}).get("overlap_violations",1)==0)
                if supported and report["final_reset_status"]!="IN_PROGRESS":
                    try:
                        report["final_reset_status"]="IN_PROGRESS";trial["reset"]=e.reset_episode();report["final_reset_status"]="PASS"
                        trial["recovery"]="supported_miss_reset";report["isolation_audit"]=e.audit()
                    except Exception as recovery:
                        trial["recovery_error"]=str(recovery);trial["partial_reset_report"]=e.last_reset_report;report["final_reset_status"]="FAIL";report["stopped_early"]=True
                        trial["failure_handling"]=e.stop_after_failure()
                else:
                    trial["failure_handling"]=e.stop_after_failure();report["stopped_early"]=True
                    if report["final_reset_status"]=="IN_PROGRESS":report["final_reset_status"]="FAIL"
            finally:
                if e.control_result is not None and e.control_result.get("trial_id")==index+1:
                    write(trial_folder/"controller.json",e.control_result)
                if not (trial_folder/"physical-trace.json").exists():
                    write(trial_folder/"physical-trace.json",e.history);write(trial_folder/"grip-metrics.json",e.grip_trace)
                write(trial_folder/"summary.json",trial)
                report["trials"].append(trial);print(f"Trial {index+1}/{args.trials}: {trial['status']}",flush=True)
            if report.get("stopped_early"):
                for rest in range(index+1,args.trials):report["trials"].append(dict(trial_index=rest+1,status="ABORTED",reason="failure_requires_review"))
                break
        passed=sum(t["status"]=="PASS" for t in report["trials"]);aborted=sum(t["status"]=="ABORTED" for t in report["trials"])
        accepted=accepted_batch(report["trials"],args.trials,e.c["minimum_successful_trials"],report["final_reset_status"],
                                e.owner.get("owner"),e.owner.get("stats",{}).get("overlap_violations"),report.get("stopped_early",False))
        report.update(status="PASS" if accepted else "FAIL",acceptance_passed=accepted,successful_trials=passed,
                      completed_trials=args.trials-aborted,aborted_trials=aborted,full_camera_pickup_gate_passed=bool(accepted and args.trials==20))
    except Exception as error:report["error"]=str(error);report["failure_handling"]=e.stop_after_failure()
    finally:
        report.update(output_directory=str(folder),elapsed_wall_s=time.monotonic()-started,elapsed_sim_s=(e.now_ns()-sim_start)/1e9 if sim_start else None,
                      final_owner=e.owner.get("owner"),overlap_violations=e.owner.get("stats",{}).get("overlap_violations"))
        write(folder/"reset-scene-diagnostics.json",e.reset_diagnostics)
        write(folder/"filter-beliefs.json",e.belief_log);write(folder/"filter-errors.json",e.filter_errors);write(folder/"summary.json",report)
        node.destroy_node();rclpy.try_shutdown()
    printed={k:v for k,v in report.items() if k not in {"trials","isolation_audit","failure_handling"}} if args.summary_only else report
    print(json.dumps(printed,indent=2));return 0 if report["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
