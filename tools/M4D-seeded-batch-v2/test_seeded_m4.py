"""Scenario seeds, strict outcomes and supported PASS evidence."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from batch_core import aggregate, assess_episode, schedule, verify_pass, write_json, pass_runout_s
import run_seeded_m4


class ScenarioTests(unittest.TestCase):
    def test_same_seed_repeats_selection_and_different_seed_changes_it(self):
        self.assertEqual(schedule(42),schedule(42));self.assertNotEqual(schedule(42),schedule(43))
        self.assertEqual(len(schedule(42)),10)
        self.assertEqual(len({(c['fixture'],c['arrival']) for c in schedule(42)}),10)

    def test_gate_covers_both_bins_inspection_and_supported_pass(self):
        cases=schedule(20261104)
        self.assertEqual(sum(c['expected_action']=='PICK_PLACE' for c in cases),4)
        self.assertEqual(sum(c['expected_action']=='PASS' for c in cases),6)
        self.assertEqual({c['expected_path'] for c in cases},{'direct','inspect','unresolved','deadline'})
        self.assertEqual({c['expected_class'] for c in cases},{'red_fruit','green_fruit'})

    def test_bad_seeds_are_rejected(self):
        for seed in (-1,2**32,True):
            with self.subTest(seed=seed),self.assertRaises(ValueError):schedule(seed)

    def test_smoke_cannot_pass_the_full_gate(self):
        cases=schedule(42,'smoke');rows=[dict(c,status='PASS') for c in cases]
        result=aggregate(dict(profile='smoke',cases=cases),rows)
        self.assertEqual(result['status'],'PASS');self.assertFalse(result['full_M4_gate_passed'])

    def test_case_four_probe_selects_red_covered_and_cannot_pass_the_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'preflight.json').write_text('{}')
            out = root/'probe'
            argv = ['run_seeded_m4.py', '--preflight-dir', folder, '--results-root', folder,
                    '--output-dir', str(out), '--seed', '42', '--case-index', '4']
            def passed(case, *_):
                return dict(case, status='PASS')
            with (patch('sys.argv', argv), patch.dict('os.environ', ROS_DOMAIN_ID='173'),
                  patch.object(run_seeded_m4, 'precheck', return_value=(root, {'installed_manifest': {}}, [])),
                  patch.object(run_seeded_m4, 'run_case', side_effect=passed), redirect_stdout(io.StringIO())):
                self.assertEqual(run_seeded_m4.main(), 0)
            result = json.loads((out/'summary.json').read_text())
            self.assertEqual(result['profile'], 'probe')
            self.assertEqual(result['successful_episodes'], 1)
            self.assertFalse(result['full_M4_gate_passed'])
            case, = result['cases']
            self.assertEqual((case['index'], case['fixture'], case['arrival']), (4, 'red_covered', 'upstream'))

    def test_failure_and_unrun_cases_remain_in_the_denominator(self):
        cases=schedule(42);rows=[dict(c,status='NOT_RUN') for c in cases]
        rows[0]['status']='PASS';rows[1]['status']='FAIL'
        result=aggregate(dict(profile='gate',cases=cases),rows)
        self.assertEqual((result['planned_episodes'],result['successful_episodes'],result['failed_episodes'],result['not_run_episodes']),(10,1,1,8))
        self.assertFalse(result['full_M4_gate_passed'])

    def test_atomic_json_preserves_real_newlines_and_rejects_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'progress.json';write_json(path,dict(value='line\nnext'))
            self.assertEqual(json.loads(path.read_text())['value'],'line\nnext')
            with self.assertRaises(ValueError):write_json(path,dict(value=float('nan')))
            self.assertEqual(json.loads(path.read_text())['value'],'line\nnext')


class EpisodeTests(unittest.TestCase):
    def report(self,case):
        result=dict(checkpoint='M4-seeded-episode',status='PASS',acceptance_passed=True,
            installed_manifest={'source':'frozen'},case_index=case['index'],scenario_selection_seed=case['scenario_selection_seed'],
            runtime_configuration=dict(status='PASS'),planning_configuration=dict(status='PASS'),fixture=case['fixture'],arrival=case['arrival'],
            action=case['expected_action'],policy_path=case['expected_path'],final_reset_status='PASS',
            final_hold_status='PASS',final_owner='HOLD',overlap_violations=0,timing=dict(status='PASS'),
            image_evidence=dict(status='PASS'))
        if case['expected_action']=='PICK_PLACE':
            result.update(physical_grasp='PASS',placement_status='PASS',wrist_timing=dict(status='PASS'),
                physical=dict(selected_bin='bin_a' if case['expected_class']=='red_fruit' else 'bin_b'))
        else:result.update(physical_grasp='NOT_RUN',placement_status='NOT_RUN',
            pass_physical=dict(status='PASS',exit_verified=True,no_arm_commands_after_decision=True))
        return result

    def test_each_expected_case_has_strict_acceptance(self):
        for case in schedule(42):assess_episode(self.report(case),case,{'source':'frozen'})

    def test_missing_exit_reset_or_matching_source_fails(self):
        case=next(c for c in schedule(42) if c['expected_action']=='PASS');report=self.report(case)
        for update in (dict(pass_physical={'status':'PASS','exit_verified':False}),dict(final_reset_status='NOT_RUN'),
                       dict(installed_manifest={'source':'changed'}),dict(overlap_violations=1)):
            with self.subTest(update=update),self.assertRaises(ValueError):
                assess_episode(dict(report,**update),case,{'source':'frozen'})


class PassTests(unittest.TestCase):
    def test_runout_uses_conservative_camera_position_and_fixed_speed(self):
        cfg=dict(uncertainty_sigma=3.,exit_x_m=.75,belt_speed_m_s=.001)
        self.assertAlmostEqual(pass_runout_s(dict(predicted_x_m=.73,position_sigma_m=.001),cfg),27.)
        with self.assertRaises(ValueError):pass_runout_s(dict(predicted_x_m=.3,position_sigma_m=.001),cfg)

    def samples(self):
        history=[];transport=[]
        for index in range(30):
            stamp=2.+index*.1
            history.append(dict(sim_time=stamp,tray_xyz=[.73+index*.001,0.,.306],finger_positions_m=[.04,.04],
                support_force_N=5.1,left_force_N=0.,right_force_N=0.,forbidden_contacts=0,bin_floor_force_N=[0.,0.]))
            transport.append(dict(sim_time=stamp,command_fresh=True,requested_speed_m_s=.001,force_x_N=.1,tray_vx_m_s=.001))
        return history,transport

    def verify(self,history,transport,**kwargs):
        options=dict(start_ns=0,decision_ns=1_000_000_000,end_ns=5_000_000_000,
            initial_z=.306,exit_x=.75,speed=.001,before_stats={'planned_accepted':2,'servo_input_accepted':3,'servo_output_forwarded':4},
            after_stats={'planned_accepted':2,'servo_input_accepted':3,'servo_output_forwarded':4})
        options.update(kwargs);return verify_pass(history,transport,**options)

    def test_supported_exit_with_open_fingers_passes(self):
        history,transport=self.samples();self.assertTrue(self.verify(history,transport)['exit_verified'])

    def test_contact_lift_missing_exit_or_early_belt_stop_fails(self):
        for field,value in (('left_force_N',.2),('tray_xyz',[.75,0.,.32]),('finger_positions_m',[.02,.02])):
            history,transport=self.samples();history[5][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.verify(history,transport)
        history,transport=self.samples()
        with self.assertRaises(ValueError):self.verify(history,transport,exit_x=.8)
        transport[5]['requested_speed_m_s']=0.
        with self.assertRaises(ValueError):self.verify(history,transport)

    def test_new_arm_command_after_pass_fails(self):
        history,transport=self.samples()
        with self.assertRaises(ValueError):self.verify(history,transport,after_stats={'planned_accepted':3,'servo_input_accepted':3,'servo_output_forwarded':4})


if __name__=='__main__':unittest.main()
