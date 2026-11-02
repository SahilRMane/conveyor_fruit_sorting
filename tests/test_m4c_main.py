"""Run every M4C CLI branch with simulated external I/O. No ROS nodes."""
from contextlib import ExitStack,redirect_stdout
import io
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import Mock,patch
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
import test_m4c as live
from m4c_core import DeadlinePass


class M4CMainTests(unittest.TestCase):
    def run_case(self,label,condition,arrival='upstream',*,capture_error=False,wrong_class=False,timing_only=False,timing_error=False):
        c=yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4c.yaml').read_text())
        fc=yaml.safe_load((ROOT/'src/sorting_perception/config/m3b_filter.yaml').read_text())
        path='deadline' if arrival=='late' else 'direct' if condition=='clear' else 'unresolved' if condition=='covered' else 'inspect'
        fixture=label+'_'+condition;scene='/simulated/'+fixture+'/'+arrival+'/scene.xml'
        n=S(belt_speed=0.,episode=7,bound_track=3,clock=10_000_000_000,events=[],history=[],operation=None,
            model=S(sha='same-model'),owner=dict(owner='HOLD',stats=dict(overlap_violations=0)),rejections={},
            feedback_diagnostic=dict(fresh=True),feedback_armed=True,deadline_diagnostic=dict(episode_id=7,track_id=3,remaining_s=.1),
            running_scene_path=scene,costs=dict(initial_assess=11.,inspect_left=14.,return_assess=15.),filter_config=fc,
            startup=Mock(),stop=Mock(),publish_decision=Mock(),destroy_node=Mock(),
            prime_timing=Mock(side_effect=ValueError('Injected timing failure') if timing_error else None),
            timing_diagnostic=Mock(return_value={'admission':{'status':'PASS'}}),
            startup_clock_pending=Mock(return_value=False),timing_context=Mock(return_value={'phase':'startup'}))
        n.now=lambda:n.clock
        n.target=lambda:dict(predicted_x_m=.54,remaining_s=60.)
        n.until=lambda predicate,**kwargs:self.assertTrue(predicate())
        n.start_transport=Mock()
        n.wait_sim=lambda seconds:setattr(n,'clock',n.clock+round(seconds*1e9))
        def begin_operation(*args,**kwargs):
            if arrival=='late':raise DeadlinePass('insufficient_slack_before_initial_assess')
        n.begin_operation=begin_operation
        def move(pose):
            n.events.append(dict(state='PLAN',pose=pose,enter_ns=n.clock));n.clock+=8_000_000_000
            n.events.append(dict(state='SETTLED',pose=pose,enter_ns=n.clock))
        n.move=move
        def capture(pose,out):
            if capture_error:raise ValueError('Injected camera failure')
            result=[];x=.54 if pose=='assess' else .56
            visible=condition=='clear' or (condition=='partial' and pose=='inspect_left')
            probabilities=([.99,.01] if label=='red' else [.01,.99]) if visible else [.5,.5]
            for _ in range(5):
                n.clock+=100_000_000
                n.history.append(dict(valid=True,camera_only=True,view_id='overhead',frame_id='world',episode_id=7,track_id=3,
                    capture_ns=n.clock,state=[x,0.,.0015,0.],covariance=(np.eye(4)*1e-7).reshape(-1).tolist()))
                result.append(dict(episode_id=7,capture_ns=n.clock,age_sim_s=.01,handle_xyz=[x,0.,.4229],
                    classifier_diagnostic_only={'probabilities':probabilities}))
            return result
        n.capture=capture
        e=S(contact=object(),failure=None,pending={},grades=[],transport=[],destroy_node=Mock(),
            window_start_ns=None,window_end_ns=None,latest_transport_ns=10**15,transport_failure=None,
            have_pairs=Mock(return_value=True),assess=Mock(return_value=[dict(localization_error_m=.0004)]),
            finish=Mock(return_value=dict(status='PASS')))
        executor=Mock();executor.callback_diagnostic={}
        preflight=dict(scene_paths={fixture+':'+arrival:scene},installed_manifest={'fixture':'binding'})
        with tempfile.TemporaryDirectory(prefix='m4c-main-') as temp:
            out=Path(temp)/'new-evidence';expected=('green' if label=='red' else 'red') if wrong_class else label
            argv=['test_m4c.py','--preflight-dir',temp,'--output-dir',str(out),
                  '--expected-class',expected+'_fruit','--expected-path',path]
            if timing_only:argv.append('--timing-only')
            stdout=io.StringIO();gc_callbacks_before=list(gc.callbacks);gc_thresholds_before=gc.get_threshold();gc_enabled_before=gc.isenabled()
            with ExitStack() as stack:
                for target,value in [('checked_preflight',(Path(temp),preflight)),('config',c),
                    ('MovingInspectionExecutor',n),('MovingEvaluator',e),('FeedbackExecutor',executor)]:
                    stack.enter_context(patch.object(live,target,return_value=value))
                stack.enter_context(patch.object(live.rclpy,'init'));stack.enter_context(patch.object(live.rclpy,'try_shutdown'))
                stack.enter_context(patch.object(sys,'argv',argv));stack.enter_context(redirect_stdout(stdout))
                with self.assertRaises(SystemExit) as outcome:live.main()
            self.assertEqual(gc.callbacks,gc_callbacks_before)
            self.assertEqual(gc.get_threshold(),gc_thresholds_before)
            self.assertEqual(gc.isenabled(),gc_enabled_before)
            summary=json.loads((out/'summary.json').read_text());evidence=json.loads((out/'evidence.json').read_text())
            self.assertEqual(json.loads((out/'events.json').read_text()),n.events)
            diagnosis=json.loads((out/'timing-diagnosis.json').read_text())
            self.assertTrue(diagnosis['diagnostic_only'])
            self.assertEqual(diagnosis['functional_status'],summary['status'])
            self.assertEqual(diagnosis['timing_diagnostic'],summary['timing_diagnostic'])
            self.assertEqual(diagnosis['collection_diagnostic'],summary['collection_diagnostic'])
            self.assertEqual(diagnosis['full_collection_policy'],summary['full_collection_policy'])
            if summary['full_collection_policy']['started']:self.assertTrue(summary['full_collection_policy']['restored'])
            if summary['status']=='FAIL':self.assertIn('timing_at_failure',diagnosis)
            records=[];text=stdout.getvalue().strip()
            while text:
                item,end=json.JSONDecoder().raw_decode(text);records.append(item);text=text[end:].lstrip()
            self.assertEqual(records[-1],{k:v for k,v in summary.items() if k!='installed_manifest'})
            self.assertEqual(summary['final_hold_status'],'PASS');self.assertFalse(summary['physical_pickup_permitted'])
            self.assertEqual(summary['final_owner'],'HOLD');self.assertEqual(summary['overlap_violations'],0)
            self.assertFalse(summary['full_M4_gate_passed'])
            n.destroy_node.assert_called_once();e.destroy_node.assert_called_once();executor.shutdown.assert_called_once()
            if capture_error or wrong_class or timing_error:
                if timing_error:
                    self.assertIn('timing failure',summary['error']);self.assertEqual(outcome.exception.code,1)
                    n.start_transport.assert_not_called();return

                self.assertEqual(outcome.exception.code,1);self.assertEqual(summary['status'],'FAIL')
                self.assertIn('Injected camera failure' if capture_error else 'launched class',summary['error']);return
            self.assertEqual(outcome.exception.code,0);self.assertEqual(summary['status'],'PASS')
            n.prime_timing.assert_called_once()
            if timing_only:
                self.assertEqual(summary['checkpoint'],'M4C-timing');self.assertFalse(summary['moving_conveyor'])
                n.start_transport.assert_not_called();n.publish_decision.assert_not_called()
                self.assertEqual(n.events,[]);self.assertEqual(evidence['samples'],[]);return
            self.assertEqual(summary['policy_path'],path)
            samples=0 if arrival=='late' else 5 if condition=='clear' else 10
            self.assertEqual(summary['checked_samples'],samples);self.assertEqual(len(evidence['samples']),samples)
            self.assertEqual(summary['inspections'],1 if path in ('inspect','unresolved') else 0)
            self.assertEqual(summary['decision']['action'],'PASS' if path in ('deadline','unresolved') else 'PICK_READY')
            if samples:self.assertTrue(all(s['track_binding']['track_id']==3 for s in evidence['samples']))
            if path in ('inspect','direct'):
                self.assertEqual(summary['decision']['class_id'],label+'_fruit')
                self.assertEqual(summary['decision']['proposed_bin'],'bin_a' if label=='red' else 'bin_b')
            n.stop.assert_called_once()

    def test_timing_only_never_starts_transport_or_motion(self):self.run_case('green','partial',timing_only=True)
    def test_timing_failure_never_starts_transport(self):self.run_case('green','partial',timing_error=True)

    def test_red_partial(self):self.run_case('red','partial')
    def test_green_partial(self):self.run_case('green','partial')
    def test_red_clear(self):self.run_case('red','clear')
    def test_green_clear(self):self.run_case('green','clear')
    def test_red_covered(self):self.run_case('red','covered')
    def test_green_covered(self):self.run_case('green','covered')
    def test_late_confident_target_declines_without_motion(self):self.run_case('red','clear','late')
    def test_camera_failure_preserves_hold_and_evidence(self):self.run_case('red','partial',capture_error=True)
    def test_wrong_evaluator_class_stops_before_motion(self):self.run_case('red','partial',wrong_class=True)


if __name__=='__main__':unittest.main()
