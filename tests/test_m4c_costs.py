"""Require complete live evidence before constructing M4C movement budgets."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4c_common import measured_costs


class M4CCostTests(unittest.TestCase):
    def fixtures(self, folder):
        for label in ('red','green'):
            for cover in ('clear','partial','covered'):
                fixture=label+'_'+cover;out=folder/('M4B-'+fixture);out.mkdir()
                poses=['assess'] if cover=='clear' else ['assess','inspect_left','assess']
                events=[];at=1_000_000_000
                for pose in poses:
                    events.append(dict(state='PLAN',pose=pose,enter_ns=at));at+=8_000_000_000
                    events.append(dict(state='SETTLED',pose=pose,enter_ns=at));at+=1_000_000_000
                summary=dict(checkpoint='M4B',status='PASS',acceptance_passed=True,fixture=fixture,final_hold_status='PASS',
                    final_owner='HOLD',overlap_violations=0,model_sha256='model',installed_manifest={},
                    checked_samples=5 if cover=='clear' else 10)
                (out/'summary.json').write_text(json.dumps(summary));(out/'events.json').write_text(json.dumps(events))

    def costs(self, folder):
        with patch('m4c_common.config',return_value=dict(cost_scale=1.25,cost_addition_s=.5,approach_allocation_s=8.)):
            return measured_costs(folder,'model')

    def test_all_six_live_sources_are_hashed_and_costs_have_margin(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.fixtures(folder);r=self.costs(folder)
            self.assertEqual(len(r['sources']),6);self.assertTrue(all(s['events_sha256'] for s in r['sources']))
            self.assertEqual(r['upper_sim_s']['inspect_left'],10.5)
            self.assertFalse(r['physical_pickup_timing_validated'])

    def test_missing_fixture_prevents_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.fixtures(folder);(folder/'M4B-red_partial/summary.json').unlink()
            with self.assertRaisesRegex(ValueError,'Six passing'):self.costs(folder)

    def test_incomplete_motion_evidence_prevents_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.fixtures(folder);(folder/'M4B-red_partial/events.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'does not match'):self.costs(folder)

    def test_model_change_does_not_reuse_old_pilot_costs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.fixtures(folder)
            with self.assertRaisesRegex(ValueError,'Six passing'):measured_costs(folder,'other-model')


if __name__=='__main__':unittest.main()
