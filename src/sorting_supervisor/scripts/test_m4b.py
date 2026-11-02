#!/usr/bin/env python3
"""Live stationary M4B inspection proposals. No grasp or sorting action is sent."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rcl_interfaces.srv import GetParameters
from m4a_runtime import InspectionExecutor
from test_m4a import Evaluator
from m4b_common import checked_preflight, classifier_config, policy_config, model_directory
from classifier_core import Classifier, require
from m4b_core import SingleViewPolicy


class ClassInspectionExecutor(InspectionExecutor):
    def __init__(self, folder, report):
        self.policy_started = None
        super().__init__(folder)
        self.scene_paths = set(report['scene_paths'].values())
        self.cc = classifier_config(); self.model = Classifier(model_directory(), self.cc)
        self.policy_config = policy_config()

    def startup(self):
        self.until(self.model_parameters.service_is_ready, timeout=60.)
        value = self.future(self.model_parameters.call_async(GetParameters.Request(names=['robot_description']))).values[0]
        root = ET.fromstring(value.string_value)
        path = root.find(".//ros2_control/hardware/param[@name='mujoco_model']").text
        require(path in self.scene_paths, 'The running scene does not match this M4B preflight')
        self.running_scene_path=path
        # The path is used only to bind the running model, never to classify or select a view.
        self.preflight_directory = Path(path).parent
        super().startup()

    def until(self, predicate, **kwargs):
        def bounded():
            if kwargs.get('check', True) and self.policy_started is not None:
                sim, wall = self.policy_started
                require(0 <= self.now()-sim < self.policy_config['maximum_policy_sim_s']*1e9, 'M4B simulation budget exceeded')
                require(time.monotonic()-wall < self.policy_config['maximum_policy_wall_s'], 'M4B wall budget exceeded')
            return predicate()
        return super().until(bounded, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight-dir', type=Path, required=True); p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--expected-class', choices=['red_fruit','green_fruit'], required=True)
    p.add_argument('--expected-action', choices=['PICK_READY','PASS'], required=True)
    p.add_argument('--expected-inspections', type=int, choices=[0,1], required=True)
    a = p.parse_args(); folder, preflight = checked_preflight(a.preflight_dir); out=a.output_dir.resolve()
    if out.exists(): p.error('Use a new output directory')
    out.mkdir(parents=True); rclpy.init(); executor = SingleThreadedExecutor()
    control, evaluator = ClassInspectionExecutor(folder, preflight), Evaluator()
    executor.add_node(control); executor.add_node(evaluator)
    report = dict(checkpoint='M4B', status='FAIL', full_M4_gate_passed=False, moving_conveyor=False,
                  physical_grasp='NOT_RUN', physical_pickup_permitted=False, proposal_only=True,
                  output_directory=str(out), final_hold_status='NOT_RUN', classifier_new_views_validated=True)
    samples, checks, policy = [], [], None
    try:
        control.startup(); control.until(lambda: evaluator.contact is not None, timeout=5.)
        # Evaluator-only fixture identity checks. The policy below gets no fixture or label.
        fixture=next(key for key,path in preflight['scene_paths'].items() if path==control.running_scene_path)
        report['fixture']=fixture
        label,condition=fixture.split('_')
        require(a.expected_class==label+'_fruit', 'Evaluator arguments do not match the launched class fixture')
        require(a.expected_action==('PASS' if condition=='covered' else 'PICK_READY') and
                a.expected_inspections==(0 if condition=='clear' else 1), 'Evaluator arguments do not match the launched cover fixture')
        control.policy_started = (control.now(), time.monotonic())
        policy = SingleViewPolicy(control.episode, control.now(), policy_config())
        for pose in ('assess', policy_config()['inspection_view']):
            if pose != 'assess': policy.start_inspection(control.now())
            control.move(pose); captured = control.capture(pose, out)
            for sample in captured:
                sample['class_evidence']=sample.pop('classifier_diagnostic_only')
            # Use one newest capture. Do not multiply correlated frame probabilities.
            evidence = captured[-1]; score = evidence['class_evidence']
            decision = policy.observe(episode=evidence['episode_id'], pose=pose, capture_ns=evidence['capture_ns'],
                                      now_ns=control.now(), probabilities=score['probabilities'])
            print(json.dumps(decision), flush=True)
            # Form the decision before waiting for evaluator-only truth packets.
            control.until(lambda: evaluator.have_pairs(captured), timeout=2.)
            checks += evaluator.assess(captured, control.c['maximum_localization_error_m'])
            samples += captured
            if decision['action'] != 'INSPECT': break
        if pose != 'assess': control.move('assess')
        control.stop()
        require(evaluator.failure is None, evaluator.failure)
        require(decision['action'] == a.expected_action, 'Evaluator: unexpected terminal decision')
        require(policy.inspections == a.expected_inspections, 'Evaluator: unexpected inspection count')
        if decision['action'] == 'PICK_READY':
            require(decision['class_id'] == a.expected_class, 'Evaluator: wrong proposed class')
        report.update(status='PASS', acceptance_passed=True, decision=decision, inspections=policy.inspections,
                      checked_samples=len(samples), final_hold_status='PASS', return_to_assess='PASS',
                      maximum_capture_age_sim_s=max(s['age_sim_s'] for s in samples),
                      maximum_localization_error_m=max(c['localization_error_m'] for c in checks))
    except (Exception, KeyboardInterrupt) as error:
        report.update(error=str(error) or 'Interrupted', acceptance_passed=False)
        try: control.stop(); report['final_hold_status']='PASS'
        except Exception as cleanup: report['cleanup_error']=str(cleanup)
    finally:
        report.update(final_owner=control.owner.get('owner'), overlap_violations=control.owner.get('stats',{}).get('overlap_violations'),
                      capture_rejections=dict(control.rejections), model_sha256=control.model.sha,
                      installed_manifest=preflight['installed_manifest'],
                      joint_feedback_diagnostic=getattr(control, 'feedback_diagnostic', {}),
                      feedback_watchdog_armed=getattr(control, 'feedback_armed', False),
                      last_state=control.events[-1] if control.events else None)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'evidence.json').write_text(json.dumps(dict(samples=samples,checks=checks,policy=policy.history if policy else []),indent=2)+'\n')
        (out/'events.json').write_text(json.dumps(control.events,indent=2)+'\n')
        executor.remove_node(control); executor.remove_node(evaluator); control.destroy_node(); evaluator.destroy_node()
        executor.shutdown(); rclpy.try_shutdown()
    print(json.dumps({k:v for k,v in report.items() if k!='installed_manifest'},indent=2))
    raise SystemExit(0 if report['status']=='PASS' else 1)


if __name__ == '__main__': main()
