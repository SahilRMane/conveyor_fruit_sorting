#!/usr/bin/env python3
"""One complete camera inspection, moving physical pickup, placement and reset."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time
import xml.etree.ElementTree as ET
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from m4d_common import checked_preflight, config
from m4c_common import config as moving_config
from m4c_timing import FullCollectionWindow
from m4d_runtime import IntegratedInspector, IntegratedExecutor, inspect, wrist_timing_report
from m4d_pickup import InspectedSortController
from m4d_evaluator import IntegratedEvaluator
from m3d_components import runtime_configuration
from carried_checks import planning_configuration
from ownership_core import require


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-dir',type=Path,required=True)
    parser.add_argument('--expected-class',choices=('red_fruit','green_fruit'),required=True)
    parser.add_argument('--expected-path',choices=('direct','inspect'),default='inspect')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();folder,preflight=checked_preflight(args.preflight_dir)
    out=args.output_dir.expanduser().resolve()
    if out.exists():parser.error('Use a new output directory to preserve previous evidence')
    out.mkdir(parents=True);rclpy.init();mc=moving_config();integration=config()
    executor=IntegratedExecutor(mc['callback_budget_s'])
    inspector=IntegratedInspector(folder/'inspection',json.loads((folder/'inspection/preflight.json').read_text()),mc)
    executor.add_node(inspector)
    executor.inspector=inspector
    evaluator_node=Node('m4d_evaluator',parameter_overrides=[Parameter('use_sim_time',value=True)],
        cli_args=['--ros-args','-r','/sorting/targets:=/sorting/m4c/targets',
                  '-r','/sorting/perception/filter_status:=/sorting/m4c/filter_status'])
    evaluator=IntegratedEvaluator(evaluator_node,executor);executor.add_node(evaluator_node)
    executor.startup_clock_pending=inspector.startup_clock_pending
    executor.diagnostic_context=inspector.timing_context
    gc_window=FullCollectionWindow();controller=None;controller_node=None
    started=time.monotonic();start_sim=None
    report=dict(checkpoint='M4D.2',status='FAIL',acceptance_passed=False,full_M4_gate_passed=False,
        camera_only=True,physical_pickup_permitted=True,physical_grasp='NOT_RUN',
        final_reset_status='NOT_RUN',final_hold_status='NOT_RUN',output_directory=str(out),
        installed_manifest=preflight['installed_manifest'])
    inspection=None;physical_history=None;grip_history=None
    try:
        inspector.startup();start_sim=inspector.now()
        controller_node=Node('m4d_sort_controller',parameter_overrides=[Parameter('use_sim_time',value=True)],
            cli_args=['--ros-args','-r','/sorting/targets:=/sorting/m4c/targets'])
        controller=InspectedSortController(controller_node,executor,inspector,None,
            mc,inspector.filter_config,integration,started+integration['maximum_episode_wall_s'],None)
        executor.add_node(controller_node);executor.physical=controller
        controller.until(lambda:controller.epoch is not None and controller.belief is not None
            and len(controller.q)>=9 and controller.owner.get('owner')=='HOLD'
            and controller.scene.get('valid') and controller.buffer.can_transform('world','panda_hand',rclpy.time.Time()),5.)
        inspector.until(lambda:bool(evaluator.contact),timeout=5.)
        evaluator.plan['initial_state']=deepcopy(evaluator.contact)
        report['runtime_configuration']=runtime_configuration(evaluator)
        report['planning_configuration']=planning_configuration(evaluator)
        report['inspection_isolation']=evaluator.isolation(inspector.get_name())
        report['pickup_isolation']=evaluator.isolation(controller_node.get_name())
        identity=next(k for k,v in json.loads((folder/'inspection/preflight.json').read_text())['scene_paths'].items()
                      if v==inspector.running_scene_path)
        fixture,arrival=identity.split(':');report.update(fixture=fixture,arrival=arrival)
        require(args.expected_class==fixture.split('_')[0]+'_fruit','Evaluator expected class differs from the scene')
        require(arrival=='upstream' and fixture.endswith(('_partial','_clear')),'This pilot requires an upstream clear or partial fixture')
        controller.wrist.calibrate()
        report['stationary_wrist_calibration']=dict(tare_world_z_N=controller.wrist.tare)
        gc_window.begin();inspection=inspect(inspector,out,integration)
        report['inspection']=inspection
        report['wrist_checks']=evaluator.wrist_checks(inspection['samples'])
        decision=inspection['decision']
        require(decision['action']=='PICK_READY','Inspection did not propose pickup: '+decision['reason'])
        require(inspector.timing.failure is None,'Camera delivery failed before pickup handoff')
        path='inspect' if inspection['inspections'] else 'direct'
        report['policy_path']=path
        require(inspector.owner.get('owner')=='HOLD' and not inspector.owner.get('busy')
                and not inspector.refinement_active and inspector.active_goal is None,'Inspection command source is still active')
        handoff_wall=time.monotonic();inspector.handed_off=True;inspector.tracking_active=False
        inspector.operation=None;executor.startup_clock_pending=None
        controller.enable_pickup(inspector.route_lease,controller.wrist.tare)
        require(time.monotonic()-handoff_wall<integration['maximum_handoff_wall_s'],'Controller handoff exceeded its wall time bound')
        controller.current_route()
        controller.pending=SimpleNamespace(episode_id=controller.epoch.episode_id,
            track_id=controller.lease.route['track_id'],trial_id=1)
        physical=out/'physical';physical.mkdir()
        print('Inspection complete. Physical pickup and placement started.',flush=True)
        controller.run_pending(physical)
        report['controller']=controller.result
        require(controller.result.get('status')=='PASS',controller.result.get('error','Physical controller failed'))
        require(controller.result['class_decision']['model_sha256']==preflight['model_sha256'],'Physical class model differs from preflight')
        evaluator.until(lambda:evaluator.control_result is not None and not evaluator.control_state.get('active'),5.)
        report['physical']=evaluator.verify_sort(controller.result,args.expected_class)
        require(path==args.expected_path,'Evaluator: inspection path differs')
        report['continuous_transport']=evaluator.transport_checks(inspection['transport_start_ns'],
            next(event['time_ns'] for event in controller.result['events'] if event['phase']=='hold'),mc['belt_speed_m_s'],
            next(event['time_ns'] for event in controller.result['events'] if event['phase']=='close'))
        report['physical_grasp']='PASS';report['placement_status']='PASS'
        require(inspector.timing.report()['status']=='PASS','Camera delivery timing failed before physical grip')
        require(inspector.pickup_timing is not None and inspector.pickup_timing.report()['status']=='PASS',
                'Wrist delivery timing failed during physical pickup')
        physical_history=deepcopy(evaluator.history);grip_history=deepcopy(evaluator.grip_trace)
        key=ET.parse(Path(inspector.running_scene_path).parent/'panda.xml').find(".//key[@name='home']")
        evaluator.reset_manager.expected={name:[float(v) for v in key.get(name,' '.join(['0']*15)).split()]
                                          for name in ('qpos','qvel','ctrl')}
        report['reset']=evaluator.reset_episode();report['final_reset_status']='PASS'
        report.update(status='PASS',acceptance_passed=True,final_hold_status='PASS')
    except (Exception,KeyboardInterrupt) as error:
        report['error']=str(error) or 'Interrupted'
        if controller is not None:
            controller.input_enabled=False;controller.target_required=False;controller.cleanup=True
            controller.belt_speed=0.;controller.send_belt()
        else:
            try:inspector.stop()
            except Exception as cleanup:report['inspection_stop_error']=str(cleanup)
        report['failure_handling']=evaluator.stop_after_failure()
    finally:
        inspector.capture_monitoring=False;inspector.belt_speed=0.
        if controller is not None:controller.input_enabled=False;controller.belt_speed=0.;controller.send_belt()
        try:
            report['image_evidence']=inspector.flush_capture_images()
        except Exception as error:
            report['image_evidence']=dict(status='FAIL',error=str(error))
            report.update(status='FAIL',acceptance_passed=False)
            report.setdefault('error','Capture evidence was not saved: '+str(error))
        gc_window.restore('episode_finished')
        report.update(final_owner=evaluator.owner.get('owner'),overlap_violations=evaluator.owner.get('stats',{}).get('overlap_violations'),
            elapsed_wall_s=time.monotonic()-started,elapsed_sim_s=(evaluator.now_ns()-start_sim)/1e9 if start_sim else None,
            timing=inspector.timing.report() if inspector.timing else None,
            wrist_timing=wrist_timing_report(inspector),
            last_controller_phase=controller.phase if controller else None,
            class_continuity=controller.continuity_log if controller else [])
        write(out/'summary.json',report);write(out/'inspection-events.json',inspector.events)
        write(out/'physical-trace.json',physical_history if physical_history is not None else evaluator.history)
        write(out/'grip-metrics.json',grip_history if grip_history is not None else evaluator.grip_trace)
        write(out/'transport.json',evaluator.transport)
        write(out/'camera-targets.json',evaluator.belief_log)
        if controller_node is not None:executor.remove_node(controller_node);controller_node.destroy_node()
        executor.remove_node(inspector);executor.remove_node(evaluator_node)
        inspector.destroy_node();evaluator_node.destroy_node();executor.shutdown();rclpy.try_shutdown()
    print(json.dumps({k:v for k,v in report.items() if k not in ('installed_manifest','inspection','controller','timing','class_continuity','wrist_checks')},indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
