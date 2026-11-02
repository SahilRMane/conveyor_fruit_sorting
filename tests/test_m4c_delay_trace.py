"""Retain rare delay samples after recent records roll over. No ROS nodes."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4c_timing import CaptureTiming
import test_m4c_callbacks as callbacks
from rclpy.task import Task


class M4CDelayTraceTests(unittest.TestCase):
    def setUp(self):
        self.c=yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4c.yaml').read_text())
        self.t=CaptureTiming(self.c,0);self.wall=100.
        self.patcher=patch('m4c_timing.time.monotonic',side_effect=lambda:self.wall)
        self.patcher.start();self.addCleanup(self.patcher.stop)

    def record(self,at,age=.034,*,pub_age=.03,receipt_age=.032,phase='moving',extra=None):
        b=dict(capture_ns=at,episode_id=0,track_id=1,valid=True)
        context=dict(phase=phase,state='PLAN',pose='inspect_left')
        self.t.note(b,at+round(pub_age*1e9),at+round(receipt_age*1e9),context=context)
        self.wall+=.004
        self.t.sample(at+round(age*1e9),enforce=False,context=context,
            batch=dict(batch_wall_s=.004),pipeline_status=extra or {'camera':{'valid':100}})

    def test_first_and_worst_survive_recent_rollover(self):
        self.record(40_000_000,.06);self.record(80_000_000,.08)
        for i in range(3,30):self.record(i*40_000_000)
        r=self.t.report()
        self.assertEqual(r['first_delivery_violation']['capture_ns'],40_000_000)
        self.assertEqual(r['worst_delivery_sample']['capture_ns'],80_000_000)
        self.assertEqual(len(r['recent']),8)
        self.assertNotIn(80_000_000,[s['capture_ns'] for s in r['recent']])
        self.assertEqual(r['violation_counts']['delivery_budget_exceeded'],2)

    def test_ties_retain_first_worst_sample(self):
        self.record(40_000_000,.08);self.record(80_000_000,.08)
        self.assertEqual(self.t.worst_delivery_sample['capture_ns'],40_000_000)

    def test_stage_clock_differences_sum_to_original_age(self):
        self.record(40_000_000,.08,pub_age=.05,receipt_age=.06)
        r=self.t.worst_delivery_sample
        self.assertAlmostEqual(r['capture_to_publication_sim_s'],.05)
        self.assertAlmostEqual(r['publication_to_receipt_clock_delta_s'],.01)
        self.assertAlmostEqual(r['receipt_to_check_clock_delta_s'],.02)
        self.assertAlmostEqual(sum(r[k] for k in ('capture_to_publication_sim_s',
            'publication_to_receipt_clock_delta_s','receipt_to_check_clock_delta_s')),r['delivery_age_s'])
        self.assertAlmostEqual(r['receipt_to_check_wall_s'],.004)

    def test_negative_local_clock_difference_is_not_clamped(self):
        self.record(40_000_000,pub_age=.035,receipt_age=.032)
        self.assertLess(self.t.worst_delivery_sample['publication_to_receipt_clock_delta_s'],0.)

    def test_retained_context_does_not_change_with_later_status(self):
        pipeline={'camera':{'valid':100}}
        self.record(40_000_000,.08,extra=pipeline);pipeline['camera']['valid']=999
        self.assertEqual(self.t.worst_delivery_sample['pipeline_status']['camera']['valid'],100)
        self.assertEqual(self.t.worst_delivery_sample['check_context']['phase'],'moving')
        self.assertEqual(self.t.worst_delivery_sample['callback_batch']['batch_wall_s'],.004)

    def test_cleanup_and_moving_samples_have_distinct_phase(self):
        self.record(40_000_000,.06,phase='moving');self.record(80_000_000,.08,phase='cleanup')
        self.assertEqual(self.t.first_delivery_violation['receipt_context']['phase'],'moving')
        self.assertEqual(self.t.worst_delivery_sample['check_context']['phase'],'cleanup')

    def test_no_delay_violation_keeps_first_delay_null(self):
        self.record(40_000_000)
        self.assertIsNone(self.t.first_delivery_violation)
        self.assertIsNotNone(self.t.worst_delivery_sample)
        self.assertEqual(self.t.violation_counts,{})

    def test_gap_fault_and_first_delay_are_separate(self):
        self.record(40_000_000);self.record(160_000_000);self.record(200_000_000,.08)
        self.assertIn('capture_gap_exceeded',self.t.first_violation['violations'])
        self.assertEqual(self.t.first_delivery_violation['capture_ns'],200_000_000)
        self.assertAlmostEqual(self.t.worst_capture_gap_sample['capture_gap_s'],.12)

    def test_sample_before_admission_epoch_has_no_trace(self):
        self.record(0,.08);self.assertIsNone(self.t.worst_delivery_sample)
        self.assertIsNone(self.t.first_delivery_violation);self.assertEqual(self.t.frames,0)

    def test_report_is_json_serializable(self):
        self.record(40_000_000,.08)
        self.assertEqual(json.loads(json.dumps(self.t.report()))['worst_delivery_sample']['delivery_age_s'],.08)

    def test_admission_snapshot_stays_unchanged_after_moving_fault(self):
        for i in range(1,251):self.record(i*40_000_000)
        self.t.ready(10_034_000_000)
        self.record(10_040_000_000,.08)
        self.assertEqual(self.t.admission['status'],'PASS')
        self.assertIsNone(self.t.admission['first_delivery_violation'])
        self.assertEqual(self.t.report()['status'],'FAIL')


class M4CBatchTraceTests(unittest.TestCase):
    def setUp(self):
        self.h=callbacks.M4CCallbackTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.e=self.h.executor;self.wall=self.h.replay

    def test_worst_and_first_over_budget_batches_are_retained(self):
        self.h.queue.append(lambda:setattr(self.wall,'wall',self.wall.wall+.02))
        self.e.spin_once(timeout_sec=0.)
        first=self.e.callback_diagnostic['first_over_budget_batch']
        self.h.queue.append(lambda:setattr(self.wall,'wall',self.wall.wall+.03))
        self.e.spin_once(timeout_sec=0.)
        self.assertEqual(self.e.callback_diagnostic['first_over_budget_batch'],first)
        self.assertGreater(self.e.callback_diagnostic['maximum_batch']['batch_wall_s'],first['batch_wall_s'])
        self.e.spin_once(timeout_sec=0.)
        self.assertEqual(self.e.callback_diagnostic['maximum_batch']['callbacks'],1)

    def test_dispatch_delay_is_separate_from_callback_execution(self):
        self.h.queue.extend([lambda:None,lambda:None]);original=self.h.ready;calls=0
        def ready(**kw):
            nonlocal calls
            calls+=1
            if calls==2:self.wall.wall+=.02
            return original(**kw)
        self.e.wait_for_ready_callbacks.side_effect=ready
        self.e.spin_once(timeout_sec=0.)
        b=self.e.callback_diagnostic['maximum_batch']
        self.assertGreaterEqual(b['outside_callback_wall_s'],.019)
        self.assertEqual(b['callback_execution_wall_s'],0.)
        self.assertEqual(b['callbacks'],2)
        self.assertFalse(b['startup_clock_alignment'])

    def test_callback_execution_duration_is_reported(self):
        self.h.queue.append(lambda:setattr(self.wall,'wall',self.wall.wall+.002))
        self.e.spin_once(timeout_sec=0.)
        b=self.e.callback_diagnostic['current_batch']
        self.assertAlmostEqual(b['callback_execution_wall_s'],.002)
        self.assertAlmostEqual(b['outside_callback_wall_s'],0.)
        self.assertAlmostEqual(b['slowest_callback']['duration_wall_s'],.002)


if __name__=='__main__':unittest.main()
