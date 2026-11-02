#!/usr/bin/env python3
"""Live moving inspection. PICK_READY is a time-stamped proposal, never a pickup command."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import rclpy
from m4c_common import checked_preflight,config
from m4c_core import MovingPolicy,bind_capture,DeadlinePass
from m4c_runtime import MovingInspectionExecutor, FeedbackExecutor
from m4c_evaluator import MovingEvaluator
from classifier_core import require
from m4c_timing import CollectionProbe,FullCollectionWindow


def finish_transport(control,evaluator):
    # Confirm the physics timestamp reached the cutoff before publishing belt zero.
    evaluator.window_end_ns=control.now()
    control.until(lambda:evaluator.latest_transport_ns is not None
        and evaluator.latest_transport_ns>=evaluator.window_end_ns,timeout=2.)
    require(evaluator.failure is None,evaluator.failure)
    control.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-class',choices=['red_fruit','green_fruit'],required=True)
    parser.add_argument('--expected-path',choices=['direct','inspect','unresolved','deadline'],required=True)
    parser.add_argument('--timing-only',action='store_true',help='Measure stationary delivery. Keep HOLD and belt zero.')
    args=parser.parse_args();folder,preflight=checked_preflight(args.preflight_dir);out=args.output_dir.expanduser().resolve()
    if out.exists():parser.error('Use a new output directory to preserve the previous evidence')
    out.mkdir(parents=True);c=config();rclpy.init();executor=FeedbackExecutor(c['callback_budget_s'])
    control,evaluator=MovingInspectionExecutor(folder,preflight,c),MovingEvaluator(c)
    executor.startup_clock_pending=control.startup_clock_pending
    executor.diagnostic_context=control.timing_context
    collections=CollectionProbe(control.timing_context);collections.start();executor.collection_probe=collections
    full_gc=FullCollectionWindow();control.full_collection_window=full_gc
    executor.add_node(control);executor.add_node(evaluator)
    report=dict(checkpoint='M4C',status='FAIL',acceptance_passed=False,full_M4_gate_passed=False,
                moving_conveyor=not args.timing_only,belt_speed_m_s=0. if args.timing_only else c['belt_speed_m_s'],physical_grasp='NOT_RUN',
                physical_pickup_permitted=False,physical_pickup_timing_validated=False,proposal_only=True,
                output_directory=str(out),final_hold_status='NOT_RUN')
    samples,checks,policy,decision=[],[],None,None;wall=time.monotonic();begin_sim=None
    try:
        control.startup();control.until(lambda:evaluator.contact is not None,timeout=5.)
        require(evaluator.failure is None,evaluator.failure)
        identity=next(k for k,v in preflight['scene_paths'].items() if v==control.running_scene_path)
        fixture,arrival=identity.split(':');report.update(fixture=fixture,arrival=arrival)
        require(args.expected_class==fixture.split('_')[0]+'_fruit','Evaluator arguments do not match the launched class')
        required_path='deadline' if arrival=='late' else ('direct' if fixture.endswith('_clear') else 'unresolved' if fixture.endswith('_covered') else 'inspect')
        require(args.expected_path==required_path,'Evaluator path does not match the launched fixture')
        require(control.owner.get('owner')=='HOLD' and not control.owner.get('busy') and control.belt_speed==0.,
                'Full-collection window requires exclusive HOLD and a stopped belt')
        full_gc.begin()
        control.prime_timing()
        if args.timing_only:
            control.stop();require(evaluator.failure is None,evaluator.failure)
            require(control.owner.get('owner')=='HOLD' and control.owner.get('stats',{}).get('overlap_violations')==0,
                    'Final exclusive HOLD failed')
            report.update(checkpoint='M4C-timing',status='PASS',acceptance_passed=True,
                          moving_conveyor=False,belt_speed_m_s=0.,policy_decisions='NOT_RUN',final_hold_status='PASS')
        else:
            control.start_transport();begin_sim=control.now();evaluator.window_start_ns=begin_sim
            control.wait_sim(2.)
            try:
                control.until(lambda:control.target()['predicted_x_m']>=c['assess_trigger_x_m'],
                              timeout=c['entry_wait_wall_s'],sim_timeout=c['entry_wait_sim_s'])
                policy=MovingPolicy(control.episode,control.bound_track,control.now(),c,control.costs)
                for pose in ('assess','inspect_left'):
                    if pose=='inspect_left':policy.start_inspection(control.now())
                    name='initial_assess' if pose=='assess' else 'inspect_left'
                    control.begin_operation(name,capture=True,returning=pose=='inspect_left')
                    control.move(pose);captured=control.capture(pose,out)
                    for sample in captured:
                        sample['class_evidence']=sample.pop('classifier_diagnostic_only')
                        sample['track_binding']=bind_capture(sample,list(control.history),control.episode,control.bound_track,c,control.filter_config)
                    control.operation=None;evidence=captured[-1]
                    lo,hi=c['classification_x_ranges'][pose]
                    require(lo<=evidence['handle_xyz'][0]<=hi,'Wrist evidence is outside the validated moving domain')
                    decision=policy.observe(track_id=control.bound_track,horizon=control.target()['remaining_s'],
                        episode=control.episode,pose=pose,capture_ns=evidence['capture_ns'],now_ns=control.now(),
                        probabilities=evidence['class_evidence']['probabilities'])
                    decision['decision_time_ns']=control.now();control.publish_decision(decision)
                    print(json.dumps(decision),flush=True)
                    control.until(lambda:evaluator.have_pairs(captured),timeout=2.)
                    checks+=evaluator.assess(captured,c['maximum_localization_error_m']);samples+=captured
                    if decision['action']!='INSPECT':break
                if pose=='inspect_left':
                    control.begin_operation('return_assess',reserve=decision['action']=='PICK_READY')
                    control.move('assess');control.operation=None
            except DeadlinePass as error:
                decision=dict(control.deadline_diagnostic)
                decision.update(action='PASS',reason=str(error),track_id=control.bound_track,episode_id=control.episode,
                              decision_time_ns=control.now(),proposal_only=True,physical_pickup_permitted=False,
                              physical_pickup_timing_validated=False)
                control.publish_decision(decision);print(json.dumps(decision),flush=True)
            finish_transport(control,evaluator);report['final_hold_status']='PASS'
            control.until(lambda:not evaluator.pending,timeout=2.,check=False)
            physical=evaluator.finish(control.events)
            inspections=policy.inspections if policy else 0
            path=('direct' if inspections==0 else 'inspect') if decision['action']=='PICK_READY' else (
                'unresolved' if decision['reason']=='uncertain_after_one_extra_view' else 'deadline')
            require(path==args.expected_path,'Evaluator: unexpected policy path')
            if decision['action']=='PICK_READY':require(decision['class_id']==args.expected_class,'Evaluator: wrong proposed class')
            require(control.owner.get('owner')=='HOLD' and control.owner.get('stats',{}).get('overlap_violations')==0,'Final exclusive HOLD failed')
            report.update(status='PASS',acceptance_passed=True,decision=decision,policy_path=path,inspections=inspections,
                          checked_samples=len(samples),tracking_and_transport=physical,
                          decision_is_current_pickup_authorization=False)
    except (Exception,KeyboardInterrupt) as error:
        report['error']=str(error) or 'Interrupted'
        report['callback_batch_at_failure']=dict(executor.callback_diagnostic)
        report['timing_at_failure']=deepcopy(control.timing_diagnostic())
        if evaluator.window_end_ns is None:evaluator.window_end_ns=control.now()
        try:control.stop();report['final_hold_status']='PASS'
        except Exception as cleanup:report['cleanup_error']=str(cleanup)
    finally:
        full_gc.restore('finalization_fallback')
        report.update(final_owner=control.owner.get('owner'),overlap_violations=control.owner.get('stats',{}).get('overlap_violations'),
            elapsed_wall_s=time.monotonic()-wall,elapsed_sim_s=(control.now()-begin_sim)/1e9 if begin_sim is not None else None,
            joint_feedback_diagnostic=control.feedback_diagnostic,feedback_watchdog_armed=control.feedback_armed,
            deadline_diagnostic=control.deadline_diagnostic,capture_rejections=dict(control.rejections),
            overhead_feedback_diagnostic=getattr(control,'overhead_diagnostic',{}),
            transport_window=dict(start_ns=evaluator.window_start_ns,end_ns=evaluator.window_end_ns,
                latest_sample_ns=evaluator.latest_transport_ns,failure=evaluator.transport_failure),
            callback_batch_diagnostic=executor.callback_diagnostic,timing_diagnostic=control.timing_diagnostic(),
            collection_diagnostic=collections.report(),full_collection_policy=full_gc.report(),
            last_state=control.events[-1] if control.events else None,model_sha256=control.model.sha,
            installed_manifest=preflight['installed_manifest'])
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'timing-diagnosis.json').write_text(json.dumps(dict(checkpoint='M4C-timing-diagnosis',
            diagnostic_only=True,functional_status=report['status'],error=report.get('error'),
            timing_diagnostic=report['timing_diagnostic'],callback_batch_diagnostic=report['callback_batch_diagnostic'],
            transport_window=report['transport_window'],last_state=report['last_state'],
            collection_diagnostic=report['collection_diagnostic'],timing_at_failure=report.get('timing_at_failure'),
            full_collection_policy=report['full_collection_policy']),indent=2)+'\n')
        (out/'events.json').write_text(json.dumps(control.events,indent=2)+'\n')
        (out/'evidence.json').write_text(json.dumps(dict(samples=samples,checks=checks,
            policy=policy.history if policy else [],overhead_grades=evaluator.grades,transport=evaluator.transport),indent=2)+'\n')
        executor.remove_node(control);executor.remove_node(evaluator);control.destroy_node();evaluator.destroy_node()
        collections.close();executor.shutdown();rclpy.try_shutdown()
    print(json.dumps({k:v for k,v in report.items() if k!='installed_manifest'},indent=2))
    raise SystemExit(0 if report['status']=='PASS' else 1)


if __name__=='__main__':main()
