"""Run the real M4B CLI flow with simulated I/O; create no ROS nodes."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/sorting_supervisor/scripts'))
import test_m4b as live


class M4BMainTests(unittest.TestCase):
    def run_case(self, label, condition, *, evaluator_error=False, wrong_expected_class=False):
        config = yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4b.yaml').read_text())
        fixture = label+'_'+condition
        action = 'PASS' if condition == 'covered' else 'PICK_READY'
        inspections = 0 if condition == 'clear' else 1
        scene = '/simulated/'+fixture+'.xml'
        control = SimpleNamespace(
            episode=7, clock=1_000_000_000, running_scene_path=scene,
            c={'maximum_localization_error_m': .002}, rejections={},
            owner={'owner': 'HOLD', 'stats': {'overlap_violations': 0}},
            model=SimpleNamespace(sha='unchanged-model'), events=[], moves=[],
            feedback_armed=True, feedback_diagnostic={'fresh': True, 'violations': []},
            startup=Mock(), stop=Mock(), destroy_node=Mock())
        control.now = lambda: control.clock

        def until(predicate, **kwargs):
            self.assertTrue(predicate(), 'Simulated input must be ready')

        def move(pose):
            control.clock += 1_000_000_000
            control.moves.append(pose)

        def capture(pose, out):
            confident = condition == 'clear' or (condition == 'partial' and pose == 'inspect_left')
            probabilities = ([.99, .01] if label == 'red' else [.01, .99]) if confident else [.5, .5]
            samples = []
            for _ in range(5):
                control.clock += 10_000_000
                samples.append(dict(episode_id=7, capture_ns=control.clock, age_sim_s=.01,
                                    classifier_diagnostic_only={'probabilities': probabilities}))
            control.events.append(dict(state='CAPTURE_COMPLETE', pose=pose, frames=5))
            return samples

        control.until, control.move, control.capture = until, move, capture
        evaluator = SimpleNamespace(contact=object(), failure=None, destroy_node=Mock(),
                                    have_pairs=Mock(return_value=True))
        evaluator.assess = Mock(side_effect=ValueError('Injected evaluator failure')) if evaluator_error else Mock(
            side_effect=lambda samples, limit: [{'localization_error_m': .0004} for _ in samples])
        executor = Mock()
        preflight = dict(scene_paths={fixture: scene}, installed_manifest={'test': 'source-binding'})
        with tempfile.TemporaryDirectory(prefix='m4b-main-') as temp:
            out = Path(temp)/'new-output'
            expected_class = ('green' if label == 'red' else 'red') if wrong_expected_class else label
            argv = ['test_m4b.py', '--preflight-dir', temp, '--output-dir', str(out),
                    '--expected-class', expected_class+'_fruit', '--expected-action', action,
                    '--expected-inspections', str(inspections)]
            output = io.StringIO()
            # Mock only external I/O. Keep main(), SingleViewPolicy and JSON output real.
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, 'argv', argv))
                stack.enter_context(patch.object(live, 'checked_preflight', return_value=(Path(temp), preflight)))
                stack.enter_context(patch.object(live, 'policy_config', return_value=config))
                stack.enter_context(patch.object(live, 'ClassInspectionExecutor', return_value=control))
                stack.enter_context(patch.object(live, 'Evaluator', return_value=evaluator))
                stack.enter_context(patch.object(live, 'SingleThreadedExecutor', return_value=executor))
                init = stack.enter_context(patch.object(live.rclpy, 'init'))
                shutdown = stack.enter_context(patch.object(live.rclpy, 'try_shutdown'))
                stack.enter_context(redirect_stdout(output))
                with self.assertRaises(SystemExit) as exit_result:
                    live.main()
            summary = json.loads((out/'summary.json').read_text())
            evidence = json.loads((out/'evidence.json').read_text())
            self.assertEqual(json.loads((out/'events.json').read_text()), control.events)
            # Read the actual progress objects and final multiline summary from stdout.
            records, remaining = [], output.getvalue().strip()
            while remaining:
                value, end = json.JSONDecoder().raw_decode(remaining)
                records.append(value)
                remaining = remaining[end:].lstrip()
            self.assertEqual(records[-1], {k: v for k, v in summary.items() if k != 'installed_manifest'})
            self.assertEqual(summary['final_hold_status'], 'PASS')
            self.assertEqual(summary['final_owner'], 'HOLD')
            self.assertEqual(summary['overlap_violations'], 0)
            self.assertFalse(summary['physical_pickup_permitted'])
            self.assertTrue(summary['feedback_watchdog_armed'])
            self.assertEqual(summary['joint_feedback_diagnostic'], control.feedback_diagnostic)
            control.stop.assert_called_once()
            control.destroy_node.assert_called_once()
            evaluator.destroy_node.assert_called_once()
            executor.shutdown.assert_called_once()
            self.assertEqual(executor.remove_node.call_count, 2)
            init.assert_called_once(); shutdown.assert_called_once()
            if evaluator_error or wrong_expected_class:
                self.assertEqual(exit_result.exception.code, 1)
                self.assertEqual(summary['status'], 'FAIL')
                self.assertFalse(summary['acceptance_passed'])
                self.assertIn('Injected evaluator failure' if evaluator_error else 'launched class fixture', summary['error'])
                self.assertEqual(control.moves, ['assess'] if evaluator_error else [])
                return
            self.assertEqual(exit_result.exception.code, 0)
            self.assertEqual(summary['status'], 'PASS')
            self.assertTrue(summary['acceptance_passed'])
            self.assertEqual(summary['decision']['action'], action)
            self.assertEqual(summary['inspections'], inspections)
            self.assertEqual(summary['checked_samples'], 5*(1+inspections))
            self.assertEqual(len(evidence['checks']), summary['checked_samples'])
            self.assertTrue(all('class_evidence' in s and 'classifier_diagnostic_only' not in s
                                for s in evidence['samples']))
            self.assertEqual(control.moves, ['assess', 'inspect_left', 'assess'] if inspections else ['assess'])
            self.assertEqual([r['action'] for r in records[:-1]], ['INSPECT', action] if inspections else [action])
            self.assertEqual([r['pose'] for r in records[:-1]], ['assess', 'inspect_left'] if inspections else ['assess'])
            self.assertEqual(len(evidence['policy']), 1+inspections)
            if action == 'PICK_READY':
                self.assertEqual(summary['decision']['class_id'], label+'_fruit')
                self.assertEqual(summary['decision']['proposed_bin'], 'bin_a' if label == 'red' else 'bin_b')

    def test_red_partial_inspect_then_ready(self): self.run_case('red', 'partial')
    def test_green_partial_inspect_then_ready(self): self.run_case('green', 'partial')
    def test_red_clear_ready_without_inspection(self): self.run_case('red', 'clear')
    def test_green_clear_ready_without_inspection(self): self.run_case('green', 'clear')
    def test_red_covered_inspect_then_pass(self): self.run_case('red', 'covered')
    def test_green_covered_inspect_then_pass(self): self.run_case('green', 'covered')
    def test_evaluator_failure_confirms_hold_and_writes_evidence(self):
        self.run_case('red', 'partial', evaluator_error=True)
    def test_wrong_fixture_arguments_stop_before_motion(self):
        self.run_case('red', 'partial', wrong_expected_class=True)


if __name__ == '__main__': unittest.main()
