#!/usr/bin/env python3
"""Run one seeded episode with the frozen learned view selector."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace
PROJECTS = Path(__file__).resolve().parents[1]
# The archive runs next to the M4 and M5 packages already in Projects.
sys.path[:0] = [str(PROJECTS / name) for name in ('M5E-learned-sorting', 'M4D-seeded-batch-v2')]
from run_m5e import model_for, panda_model, pose_targets, reset_reference, sha
from prepare_m5d import load_case, read
from batch_core import write_json, verify_pass, pass_runout_s
from m4d_common import config
from m4c_common import config as moving_config
from m4d_runtime import IntegratedExecutor, wrist_timing_report
from learned_inspection import inspect
from m4d_pickup import InspectedSortController
from m4d_evaluator import IntegratedEvaluator
from m4c_timing import FullCollectionWindow
from m3d_components import runtime_configuration
from carried_checks import planning_configuration
from ownership_core import require
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


def inspection_targets(suite, scene, arrival):
    if arrival == 'late':
        upstream = Path(suite['cases']['left']['scene'])
        models = [ET.parse(panda_model(path)).getroot() for path in (scene, upstream)]
        for model in models:
            home = model.find(".//key[@name='home']")
            qpos = home.attrib['qpos'].split()
            # The late scene changes the tray's home X, independent of the arm.
            qpos[9] = 'tray_x'
            home.set('qpos', ' '.join(qpos))
        require(ET.tostring(models[0]) == ET.tostring(models[1]), 'Late scene changed the Panda model')
        scene = upstream
    return pose_targets(suite, scene)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir',type=Path,required=True)
    parser.add_argument('--case-file',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();case=json.loads(args.case_file.read_text())
    suite, _ = load_case(args.suite_dir, 'right' if case['fixture'] == 'green_right_reveal' else 'left')
    model = model_for(suite)
    folder = Path(suite['preflight']) / 'inspection'
    preflight = read(folder / 'preflight.json')
    scene = Path(case['scene'])
    require(sha(scene) == case['scene_sha256'], 'Episode scene differs from the batch manifest')
    preflight['scene_paths']['m5f'] = str(scene)
    expected_reset = reset_reference(scene)
    joints = inspection_targets(suite, scene, case['arrival'])
    helper = Path(suite['m5b_helper'])
    require(sha(helper) == suite['m5b_helper_sha256'], 'M5B motion helper differs')
    sys.path.insert(0, str(helper.parent))
    from probe_m5b import Inspector
    out=args.output_dir.resolve()
    if out.exists():parser.error('Use a new episode output directory')
    out.mkdir(parents=True);rclpy.init();mc=moving_config();limits=config()
    mc.update(assess_trigger_x_m=.532, entry_wait_sim_s=85.)
    executor=IntegratedExecutor(mc['callback_budget_s'])
    control=Inspector(folder, preflight, mc, joints)
    control.c['samples_per_pose'] = 1
    executor.add_node(control);executor.inspector=control
    evaluator_node=Node('m4d_evaluator',parameter_overrides=[Parameter('use_sim_time',value=True)],
        cli_args=['--ros-args','-r','/sorting/targets:=/sorting/m4c/targets',
                  '-r','/sorting/perception/filter_status:=/sorting/m4c/filter_status'])
    evaluator=IntegratedEvaluator(evaluator_node,executor);executor.add_node(evaluator_node)
    executor.startup_clock_pending=control.startup_clock_pending;executor.diagnostic_context=control.timing_context
    gc_window=FullCollectionWindow();controller=None;node=None;detached=False;history=[];transport=[]
    started=time.monotonic();begin_sim=None
    report=dict(checkpoint='M5F-seeded-episode',status='FAIL',acceptance_passed=False,
        selector_sha256=suite['selector_sha256'], suite_sha256=sha(args.suite_dir / 'suite.json'),
        scene=str(scene), scene_sha256=case['scene_sha256'],
        scenario_selection_seed=case['scenario_selection_seed'],case_index=case['index'],camera_only=True,physical_grasp='NOT_RUN',
        placement_status='NOT_RUN',final_reset_status='NOT_RUN',final_hold_status='NOT_RUN',
        installed_manifest=read(Path(suite['preflight']) / 'preflight.json')['installed_manifest'])
    try:
        control.startup();begin_sim=control.now()
        node=Node('m4d_sort_controller',parameter_overrides=[Parameter('use_sim_time',value=True)],
            cli_args=['--ros-args','-r','/sorting/targets:=/sorting/m4c/targets'])
        controller=InspectedSortController(node,executor,control,None,mc,control.filter_config,limits,
            started+limits['maximum_episode_wall_s'],None)
        executor.add_node(node);executor.physical=controller
        controller.until(lambda:controller.epoch is not None and controller.belief is not None and len(controller.q)>=9
            and controller.owner.get('owner')=='HOLD' and controller.scene.get('valid')
            and controller.buffer.can_transform('world','panda_hand',rclpy.time.Time()),5.)
        control.until(lambda:bool(evaluator.contact),timeout=5.)
        evaluator.plan['initial_state']=deepcopy(evaluator.contact)
        report['runtime_configuration']=runtime_configuration(evaluator)
        report['planning_configuration']=planning_configuration(evaluator)
        report['inspection_isolation']=evaluator.isolation(control.get_name())
        report['pickup_isolation']=evaluator.isolation(node.get_name())
        require(control.running_scene_path == str(scene) and control.model.sha == suite['model_sha256'],
                'Launched scene or classifier differs from the batch manifest')
        report.update(fixture=case['fixture'], arrival=case['arrival'])
        controller.wrist.calibrate()
        report['stationary_wrist_calibration'] = dict(tare_world_z_N=controller.wrist.tare)
        before_inspection = deepcopy(control.owner['stats'])
        gc_window.begin()
        inspection = inspect(control, out, limits, suite, model)
        report['inspection'] = inspection
        report['wrist_checks']=evaluator.wrist_checks(inspection['samples']) if inspection['samples'] else []
        decision=inspection['decision']
        require(control.timing.failure is None,'Camera timing failed before execution')
        require(control.owner.get('owner')=='HOLD' and not control.owner.get('busy')
                and not control.refinement_active and control.active_goal is None,'Inspection command source remains active')
        if decision['action']=='PICK_READY':
            report.update(action='PICK_PLACE',policy_path=inspection['path'])
            handoff=time.monotonic();control.handed_off=True;control.tracking_active=False;control.operation=None
            executor.startup_clock_pending=None
            controller.enable_pickup(control.route_lease,controller.wrist.tare)
            require(time.monotonic()-handoff<limits['maximum_handoff_wall_s'],'Handoff exceeded its bound')
            controller.current_route();controller.pending=SimpleNamespace(episode_id=controller.epoch.episode_id,
                track_id=controller.lease.route['track_id'],trial_id=case['index'])
            physical=out/'physical';physical.mkdir();controller.run_pending(physical)
            report['controller']=controller.result
            require(controller.result.get('status')=='PASS',controller.result.get('error','Physical controller failed'))
            evaluator.until(lambda:evaluator.control_result is not None and not evaluator.control_state.get('active'),5.)
            report['physical']=evaluator.verify_sort(controller.result,case['expected_class'])
            events=controller.result['events']
            report['continuous_transport']=evaluator.transport_checks(inspection['transport_start_ns'],
                next(e['time_ns'] for e in events if e['phase']=='hold'),mc['belt_speed_m_s'],
                next(e['time_ns'] for e in events if e['phase']=='close'))
            require(control.pickup_timing is not None and control.pickup_timing.report()['status']=='PASS','Wrist timing failed')
            report.update(physical_grasp='PASS',placement_status='PASS')
        else:
            require(decision['action']=='PASS','Unknown policy action')
            report.update(action='PASS',policy_path=inspection['path'])
            control.publish_decision(decision)
            target=control.target();runout=pass_runout_s(target,mc)
            report['pass_runout']=dict(duration_sim_s=runout,start_camera_target=target,
                timing_scope='Camera timing remains strict until the final PASS decision')
            control.tracking_active=False;control.capture_monitoring=False;control.operation=None
            # Inspection has finished. Runout has its own simulation and wall deadlines.
            control.policy_started = None
            executor.startup_clock_pending = None
            pass_begin = control.now()
            stats = before_inspection if case['expected_path'] == 'deadline' else deepcopy(control.owner['stats'])
            initial_z = evaluator.plan['initial_state']['tray_xyz'][2]
            def exited():
                require(time.monotonic()-started<limits['maximum_episode_wall_s'],'PASS wall bound exceeded')
                require(control.owner.get('owner')=='HOLD' and not control.owner.get('busy'),'PASS lost exclusive HOLD')
                require(not control.refinement_active and control.active_goal is None,'PASS issued an arm goal')
                require(all(control.q.get(n,0)>.039 for n in ('panda_finger_joint1','panda_finger_joint2')),'PASS closed fingers')
                return control.now()-pass_begin>=round(runout*1e9)
            control.until(exited,timeout=max(.01,limits['maximum_episode_wall_s']-(time.monotonic()-started)),sim_timeout=runout+2.)
            cutoff=control.now()
            control.until(lambda:round(evaluator.contact['sim_time']*1e9)>=cutoff
                and evaluator.transport and round(evaluator.transport[-1]['sim_time']*1e9)>=cutoff,timeout=2.)
            control.stop()
            report['pass_physical']=verify_pass(evaluator.history,evaluator.transport,start_ns=inspection['transport_start_ns'],
                decision_ns=pass_begin,end_ns=cutoff,initial_z=initial_z,exit_x=mc['exit_x_m'],speed=mc['belt_speed_m_s'],
                before_stats=stats,after_stats=control.owner['stats'])
        selection = inspection['selection'] or {}
        report['selected_view'] = selection.get('selected_view')
        require(control.timing.report()['status']=='PASS','Overhead timing did not pass')
        history=deepcopy(evaluator.history);transport=deepcopy(evaluator.transport)
        control.capture_monitoring=False;control.tracking_active=False;control.belt_speed=0.;control.stop()
        controller.input_enabled=False;controller.belt_speed=0.;controller.send_belt()
        executor.remove_node(control);executor.remove_node(node);executor.inspector=None;executor.physical=None;detached=True
        evaluator.reset_manager.expected = expected_reset
        report['reset'] = evaluator.reset_episode()
        require(evaluator.owner.get('owner') == 'HOLD' and not evaluator.owner.get('busy')
                and evaluator.owner['stats']['overlap_violations'] == 0, 'Final exclusive HOLD failed')
        report.update(status='PASS',acceptance_passed=True,final_reset_status='PASS',final_hold_status='PASS')
    except (Exception,KeyboardInterrupt) as error:
        report['error']=str(error) or 'Interrupted'
        if controller is not None:
            controller.input_enabled=False;controller.cleanup=True;controller.target_required=False;controller.belt_speed=0.;controller.send_belt()
        try:control.stop()
        except Exception as stop_error:report['stop_error']=str(stop_error)
        report['failure_handling']=evaluator.stop_after_failure()
    finally:
        control.capture_monitoring=False;control.belt_speed=0.
        if controller is not None:controller.input_enabled=False;controller.belt_speed=0.;controller.send_belt()
        try:report['image_evidence']=control.flush_capture_images()
        except Exception as error:
            report.update(status='FAIL', acceptance_passed=False, error=str(error))
            report['image_evidence'] = dict(status='FAIL', error=str(error))
        gc_window.restore('episode_finished')
        report.update(final_owner=evaluator.owner.get('owner'),overlap_violations=evaluator.owner.get('stats',{}).get('overlap_violations'),
            elapsed_wall_s=time.monotonic()-started,elapsed_sim_s=(evaluator.now_ns()-begin_sim)/1e9 if begin_sim else None,
            timing=control.timing.report() if control.timing else None,wrist_timing=wrist_timing_report(control))
        write_json(out/'summary.json',report);write_json(out/'inspection-events.json',control.events)
        write_json(out/'physical-trace.json',history or evaluator.history);write_json(out/'transport.json',transport or evaluator.transport)
        if not detached:
            executor.remove_node(control)
            if node is not None:executor.remove_node(node)
        if node is not None:node.destroy_node()
        executor.remove_node(evaluator_node);control.destroy_node();evaluator_node.destroy_node();executor.shutdown();rclpy.try_shutdown()
    print(json.dumps({k:report.get(k) for k in ('status','fixture','action','policy_path','physical_grasp','placement_status','final_reset_status','final_hold_status','error')},indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
