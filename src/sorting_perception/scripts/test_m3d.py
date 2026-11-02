#!/usr/bin/env python3
"""Evaluate fixed-view class-driven moving pickup/placement or covered pass."""
import argparse,json,time,uuid
from datetime import datetime
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sorting_interfaces.srv import StartCameraPickup
from m3d_evaluator import SortEvaluator
from m3d_manifest import installed_manifest
from ownership_core import require
from m3d_components import runtime_configuration
from carried_checks import planning_configuration

def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight-dir',type=Path,required=True);parser.add_argument('--planning-probe-dir',type=Path);parser.add_argument('--expected-class',choices=('red_fruit','green_fruit'),required=True);parser.add_argument('--expect-pass',action='store_true');parser.add_argument('--trials',type=int,choices=(1,10),default=1);parser.add_argument('--summary-only',action='store_true');parser.add_argument('--output-dir',type=Path);a=parser.parse_args()
    pre=json.loads((a.preflight_dir/'validation.json').read_text())
    if pre.get('checkpoint')!='M3D' or pre.get('status')!='PASS' or pre.get('installed_manifest')!=installed_manifest():parser.error('Run a fresh installed M3D preflight')
    if not a.expect_pass:
        if a.planning_probe_dir is None:parser.error('Run the planning-only probe and supply --planning-probe-dir')
        proof=json.loads((a.planning_probe_dir/'validation.json').read_text())
        if proof.get('checkpoint')!='M3D-carried-planning' or proof.get('status')!='PASS' or proof.get('successful_plans')!=40 or proof.get('scene_context')!='private_fixture_scenes' or proof.get('installed_manifest')!=pre['installed_manifest']:parser.error('Run a fresh installed planning-only probe')
    folder=(a.output_dir or Path('results')/f'M3D-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}').resolve()
    if folder.exists():parser.error('Use a new output directory to preserve the previous evidence')
    folder.mkdir(parents=True)
    rclpy.init();node=Node('test_m3d',parameter_overrides=[Parameter('use_sim_time',value=True)]);e=SortEvaluator(node);report=dict(checkpoint='M3D',status='FAIL',camera_only=True,placement_feedback='physical_wrist_wrench',classifier_ready=True,fixed_view=True,installed_manifest=pre['installed_manifest'],expected_class=a.expected_class,expected_pass=a.expect_pass,trials=[],final_reset_status='NOT_RUN',full_M3_gate_passed=False)
    started=time.monotonic();started_sim=None
    try:
        e.until(lambda:e.owner.get('owner')=='HOLD' and e.owner.get('ready') and e.current() is not None and e.contact and e.scene.get('valid'),45.)
        started_sim=e.now_ns()
        require(e.owner['stats']['planned_accepted']==0 and e.owner['stats']['servo_input_accepted']==0,'Start a fresh M3D launch')
        report['runtime_configuration']=runtime_configuration(e)
        report['planning_configuration']=planning_configuration(e)
        report['isolation_audit']=e.audit();e.plan['initial_state']=e.contact.copy()
        for index in range(a.trials):
            trial=dict(index=index+1,status='FAIL');output=folder/f'trial-{index+1:02d}';output.mkdir();e.prepare_trial();report['final_reset_status']='NOT_RUN'
            try:
                e.until(lambda:e.current() is not None);b=e.current();response=e.call(e.start,StartCameraPickup.Request(episode_id=b.episode_id,track_id=b.track_id,trial_id=index+1));require(response.accepted,response.reason)
                e.until(lambda:e.control_result and e.control_result.get('trial_id')==index+1,240.)
                write(output/'controller.json',e.control_result);require(e.control_result.get('status')=='PASS',e.control_result.get('error','Controller did not complete the operation'));require(e.control_result.get('class_decision',{}).get('model_sha256')==pre['model_sha256'],'Live classifier model differs from preflight');trial['physical']=e.verify_sort(e.control_result,a.expected_class,a.expect_pass)
                trial['decision']=e.control_result['class_decision'];trial['reset']=e.reset_episode();report['final_reset_status']='PASS';trial['status']='PASS'
            except Exception as error:
                trial['error']=str(error)
                failure=e.control_result if e.control_result and e.control_result.get('trial_id')==index+1 else {}
                phase=failure.get('failure_diagnostic',{}).get('phase',e.phase)
                report['last_failure']=dict(trial=index+1,phase=phase,reason=str(error))
                # A supported miss may reset. A held load or physical violation stops the batch.
                supported=(not e.control_state.get('active') and e.owner.get('owner')=='HOLD' and e.owner.get('ready') and e.contact.get('support_force_N',0)>4. and e.maximum_rise<.003 and
                    all(v.get('forbidden_contacts',1)==0 and max(v['left_force_N'],v['right_force_N'])<15 for v in e.history))
                if supported:
                    try:trial['reset']=e.reset_episode();report['final_reset_status']='PASS'
                    except Exception as reset:trial['reset_error']=str(reset);report['final_reset_status']='FAIL';report['stopped_early']=True;trial['failure_handling']=e.stop_after_failure()
                else:report['stopped_early']=True;trial['failure_handling']=e.stop_after_failure()
            finally:
                if e.control_result and e.control_result.get('trial_id')==index+1:write(output/'controller.json',e.control_result)
                write(output/'physical-trace.json',e.history);write(output/'grip-metrics.json',e.grip_trace);write(output/'summary.json',trial);report['trials'].append(trial);print(f'Trial {index+1}/{a.trials}: {trial["status"]}',flush=True)
            if report.get('stopped_early'):break
        successful=sum(v['status']=='PASS' for v in report['trials']);completed=len(report['trials'])
        accepted=(completed==a.trials and successful>=(a.trials if a.trials==1 or a.expect_pass else 8) and report['final_reset_status']=='PASS' and not report.get('stopped_early') and all(t.get('reset',{}).get('status')=='PASS' for t in report['trials']) and e.owner.get('owner')=='HOLD' and e.owner.get('stats',{}).get('overlap_violations')==0)
        report.update(status='PASS' if accepted else 'FAIL',acceptance_passed=accepted,successful_trials=successful,completed_trials=completed,aborted_trials=a.trials-completed,
            classified_bin_batch_passed=bool(accepted and a.trials==10 and not a.expect_pass))
    except Exception as error:
        report['error']=str(error)
        if e.audit_diagnostic is not None:report['startup_audit_diagnostic']=e.audit_diagnostic
        report['failure_handling']=e.stop_after_failure()
    finally:
        report.update(output_directory=str(folder),elapsed_wall_s=time.monotonic()-started,elapsed_sim_s=(e.now_ns()-started_sim)/1e9 if started_sim is not None else None,final_owner=e.owner.get('owner'),overlap_violations=e.owner.get('stats',{}).get('overlap_violations'))
        write(folder/'summary.json',report);node.destroy_node();rclpy.try_shutdown()
    printed={k:v for k,v in report.items() if k not in ('trials','isolation_audit','failure_handling','installed_manifest')} if a.summary_only else report;print(json.dumps(printed,indent=2));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
