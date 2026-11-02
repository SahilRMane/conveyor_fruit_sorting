"""Check the loss allowance and stationary admission without ROS nodes."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace as S
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4c_timing import CaptureTiming,timing_budget
from m4c_checks import replay_capture_delivery
import test_m4c_delivery as delivery


class M4CTimingTests(unittest.TestCase):
    def setUp(self):
        self.c=yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4c.yaml').read_text())
        self.t=CaptureTiming(self.c,0)

    def note(self,at,now=None,**changes):
        b=dict(capture_ns=at,episode_id=0,track_id=1,valid=True);b.update(changes)
        self.t.note(b,at+10_000_000,now if now is not None else at+20_000_000)
        self.t.sample(now if now is not None else at+20_000_000)

    def test_one_missing_capture_budget_is_below_150_ms(self):
        self.assertAlmostEqual(timing_budget(self.c)['age_with_guard_tick_s'],.147)

    def test_old_rate_cannot_reserve_one_missing_capture(self):
        c=dict(self.c,camera_rate_hz=15.,capture_period_budget_s=1/15)
        with self.assertRaisesRegex(ValueError,'missing capture'):timing_budget(c)

    def test_render_must_fit_period(self):
        with self.assertRaisesRegex(ValueError,'Rendering'):timing_budget(dict(self.c,maximum_pair_render_p95_wall_s=.04))

    def test_exact_delivery_allocation_passes(self):self.note(40_000_000,90_000_000)
    def test_delivery_above_allocation_fails(self):
        with self.assertRaisesRegex(ValueError,'delivery_budget'):self.note(40_000_000,90_000_001)
    def test_future_stamp_fails(self):
        with self.assertRaisesRegex(ValueError,'future_capture'):self.note(40_000_000,39_000_000)
    def test_invalid_target_fails(self):
        with self.assertRaisesRegex(ValueError,'invalid_target'):self.note(40_000_000,valid=False)
    def test_changed_identity_fails(self):
        self.note(40_000_000)
        with self.assertRaisesRegex(ValueError,'identity'):self.note(80_000_000,track_id=2)
    def test_one_missing_capture_with_jitter_passes(self):
        self.note(40_000_000);self.note(125_000_000)
    def test_two_missing_captures_fail_admission(self):
        self.note(40_000_000)
        with self.assertRaisesRegex(ValueError,'gap'):self.note(160_000_000)
    def test_duplicate_does_not_satisfy_admission(self):
        self.note(40_000_000)
        with self.assertRaisesRegex(ValueError,'gap'):self.note(40_000_000)
    def test_old_capture_does_not_count(self):
        self.note(-1);self.assertEqual(self.t.frames,0)
    def test_full_ten_second_stream_passes(self):
        for i in range(1,251):self.note(i*40_000_000)
        self.assertTrue(self.t.ready(10_020_000_000));self.assertEqual(self.t.admission['status'],'PASS')
        self.assertEqual(self.t.frames,250)
    def test_too_short_stream_does_not_pass(self):
        self.note(40_000_000);self.assertFalse(self.t.ready(1_000_000_000))
    def test_too_few_updates_fail(self):
        self.note(40_000_000)
        with self.assertRaisesRegex(ValueError,'Too few'):self.t.ready(10_000_000_000)
    def test_admission_report_stays_separate_from_later_diagnostics(self):
        for i in range(1,251):self.note(i*40_000_000)
        self.t.ready(10_020_000_000)
        self.t.note(dict(capture_ns=10_200_000_000,episode_id=0,track_id=1,valid=True),10_210_000_000,10_220_000_000)
        self.t.sample(10_220_000_000,enforce=False)
        self.assertEqual(self.t.admission['status'],'PASS');self.assertEqual(self.t.report()['status'],'FAIL')
    def test_pending_queue_is_bounded(self):
        for i in range(150):self.t.note(dict(capture_ns=i+1,episode_id=0,track_id=1,valid=True),i,i)
        self.assertEqual(len(self.t.pending),128)
        with self.assertRaisesRegex(ValueError,'queue'):self.t.sample(200)
    def test_clock_first_replay_includes_isolated_missing_updates(self):
        for hz,passed in [(15,False),(25,True)]:
            captures=[round(i/hz*1e9) for i in range(300) if i%31!=15]
            r=replay_capture_delivery(captures,.15,.055,.01)
            self.assertEqual(r['stale_checks']==0,passed)
    def test_two_consecutive_missing_updates_are_not_guaranteed(self):
        captures=[i*40_000_000 for i in range(300) if i%31 not in (15,16)]
        self.assertGreater(replay_capture_delivery(captures,.15,.055,.01)['stale_checks'],0)


class M4CTimingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.h=delivery.M4CDeliveryTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.n=self.h.n;self.n.timing=None;self.n.pipeline_status={};self.n.publisher_diagnostic={}
        self.n.get_publishers_info_by_topic=lambda topic:[S(node_name='expected',node_namespace='/')]

    def test_no_transport_before_admission(self):
        with self.assertRaisesRegex(ValueError,'admission'):self.n.start_transport()
        self.assertEqual(self.h.commands,[])
    def test_real_admission_passes_without_conveyor_motion(self):
        self.h.stream(hz=25.,delay=.02,seconds=12.)
        self.n.prime_timing()
        self.assertEqual(self.n.timing.admission['status'],'PASS')
        self.assertEqual(self.h.commands,[]);self.assertFalse(self.n.tracking_active)
        self.n.start_transport();self.assertEqual(self.h.commands[-1][1],.001)
    def test_real_admission_rejects_late_delivery_before_motion(self):
        self.h.stream(hz=25.,delay=.06,seconds=12.)
        with self.assertRaisesRegex(ValueError,'delivery_budget'):self.n.prime_timing()
        self.assertEqual(self.h.commands,[])
    def test_duplicate_publishers_fail_before_motion(self):
        self.n.get_publishers_info_by_topic=lambda topic:[S(node_name='a',node_namespace='/')]*2
        with self.assertRaisesRegex(ValueError,'one publisher'):self.n.check_publishers()
        self.assertEqual(self.h.commands,[])
    def test_missing_publisher_fails_before_motion(self):
        self.n.get_publishers_info_by_topic=lambda topic:[]
        with self.assertRaisesRegex(ValueError,'one publisher'):self.n.check_publishers()
    def test_pipeline_status_is_saved_without_modifying_targets(self):
        self.n.receive_pipeline_status('/sorting/m4c/filter_status',S(data='{"accepted":117,"rejected":0}'))
        self.assertEqual(self.n.timing_diagnostic()['pipeline_status']['/sorting/m4c/filter_status']['data']['accepted'],117)

if __name__=='__main__':unittest.main()
