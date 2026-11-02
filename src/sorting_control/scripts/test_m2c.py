#!/usr/bin/env python3
"""Run one moving physical pickup or the 20-trial nominal M2 oracle gate."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import uuid
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from ament_index_python.packages import get_package_share_directory
from m2c_runtime import MovingGrasp
from ownership_core import require
from grasp_core import ConstraintViolation,summarize_trials


def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-oracle",action="store_true")
    parser.add_argument("--trials",type=int,choices=(1,20),default=1)
    parser.add_argument("--output-dir",type=Path)
    args = parser.parse_args()
    if not args.allow_oracle: parser.error("This is a named oracle debug test; pass --allow-oracle")
    folder = args.output_dir or Path("results")/f"M2C-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True,exist_ok=False)
    control = Path(get_package_share_directory("sorting_control")); source = Path(__file__).resolve().parent
    sim = Path(get_package_share_directory("sorting_simulation")); description = Path(get_package_share_directory("sorting_description"))
    files = [control/"config"/n for n in ("m2c.yaml","m2b.yaml","servo.yaml","ownership.yaml","ownership_controllers.yaml","fixtures.yaml")]
    files += [source/n for n in ("test_m2c.py","m2c_runtime.py","m2c_reset.py","grasp_core.py","grasp_scene.py","arm_command_gate.py","ownership_core.py","m2b_runtime.py","prediction_core.py")]
    files += [control/"launch/panda_m2c.launch.py",description/"mujoco/static_grasp/scene_tracking.xml"]
    write(folder/"manifest.json",{"started_at":datetime.now().astimezone().isoformat(),"oracle_only":True,
          "prepositioned_oracle_debug":True,"nominal_reset_repeats":True,"requested_trials":args.trials,
          "slip_frame":"panda_hand","slip_metric_version":2,
          "forced_attachment":False,"file_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    report = {"status":"FAIL","acceptance_passed":False,"full_M2_gate_passed":False,
              "oracle_only":True,"moving_grasp":True,"prepositioned_oracle_debug":True,
              "slip_frame":"panda_hand","slip_metric_version":2,
              "requested_trials":args.trials,"trials":[]}
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("test_m2c",parameter_overrides=[Parameter("use_sim_time",value=True)])
    probe = MovingGrasp(node); started_wall = time.monotonic(); started_sim = None
    try:
        probe.startup(); probe.until(lambda:probe.grasp_scene.get("valid") and probe.hand.server_is_ready())
        started_sim = probe.now_ns(); probe.plan["initial_state"] = probe.contact.copy()
        for index in range(args.trials):
            trial_folder = folder/f"trial-{index+1:02d}"; trial_folder.mkdir()
            trial = {"trial_index":index+1,"scenario":"nominal_reset_repeat","status":"FAIL"}
            probe.attempt_record = None
            try:
                require(probe.owner["owner"]=="HOLD" and probe.owner["ready"],"Trial requires ready HOLD")
                pickup = probe.run_pickup(trial_folder)
                trial.update(pickup); trial["trial_index"] = index+1
                write(trial_folder/"pickup.json",pickup)
                reset = probe.reset_episode(); trial["reset"] = reset
                trial["checks"]["guarded_reset"] = "PASS"; trial["passed_checks"] = 7
                trial["final_owner_state"] = copy.deepcopy(probe.owner)
                trial["status"] = "PASS"
            except (Exception,KeyboardInterrupt) as error:
                if probe.attempt_record: trial.update(copy.deepcopy(probe.attempt_record))
                trial.update(status="FAIL",acceptance_passed=False,error=str(error) or "Interrupted by user",
                             constraint_violation=isinstance(error,ConstraintViolation),
                             owner_at_failure=copy.deepcopy(probe.owner),contact_at_failure=copy.deepcopy(probe.contact))
                trial["failure_handling"] = probe.stop_after_failure()
                # Supported failures can reset. A tray in the air keeps its grip
                # and the paused state; the remaining trials are recorded aborted.
                recoverable = (not isinstance(error,ConstraintViolation) and probe.owner.get("owner")=="HOLD" and probe.owner.get("ready") and
                               probe.contact.get("support_force_N",0)>0.5 and
                               probe.contact.get("forbidden_contacts",1)==0)
                if recoverable and not isinstance(error,KeyboardInterrupt):
                    try: trial["reset"] = probe.reset_episode()
                    except Exception as reset_error: trial["reset_error"] = str(reset_error); recoverable = False
                if not recoverable or isinstance(error,KeyboardInterrupt):
                    report["stopped_early"] = True
            finally:
                trial["maximum_relative_slip_m"] = max(trial.get("maximum_relative_slip_m",0),probe.maximum_slip)
                trial["maximum_world_offset_change_m"] = probe.maximum_world_offset_change
                trial["passed_checks"] = sum(value=="PASS" for value in trial.get("checks",{}).values())
                write(trial_folder/"tracking.json",probe.trace)
                write(trial_folder/"grip-metrics.json",probe.grip_trace)
                write(trial_folder/"summary.json",trial)
                report["trials"].append(trial)
                print(f"Trial {index+1}/{args.trials}: {trial['status']}",flush=True)
            if report.get("stopped_early"):
                for rest in range(index+1,args.trials):
                    report["trials"].append({"trial_index":rest+1,"status":"ABORTED","reason":"unsafe_recovery_or_reset_failure"})
                break
        report["successful_grasps"] = sum(t.get("checks",{}).get("physical_hold")=="PASS" for t in report["trials"])
        success_needed = 1 if args.trials==1 else probe.c["minimum_successful_trials"]
        report.update(summarize_trials(report["trials"],args.trials,probe.owner,success_needed))
        report["final_owner_state"] = copy.deepcopy(probe.owner)
        if args.trials==1 and report["trials"]:
            report["checks"] = report["trials"][0].get("checks",{}); report["passed_checks"] = report["trials"][0].get("passed_checks",0)
    except (Exception,KeyboardInterrupt) as error:
        report["error"] = str(error) or "Interrupted by user"; report["failure_handling"] = probe.stop_after_failure()
    finally:
        probe.input_enabled = False; probe.guarded = False; probe.belt_speed = 0.0; probe.send_belt()
        report.update(output_directory=str(folder.resolve()),elapsed_wall_s=time.monotonic()-started_wall,
                      elapsed_sim_s=(probe.now_ns()-started_sim)/1e9 if started_sim is not None else None,
                      last_owner_state=probe.owner,last_contact_state=probe.contact)
        write(folder/"events.json",probe.events); write(folder/"summary.json",report)
        node.destroy_node(); rclpy.try_shutdown()
    print(json.dumps(report,indent=2));return 0 if report["status"]=="PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
