"""Exercise exact capture joins and both real producer callbacks without ROS nodes."""
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace as S
from rclpy.serialization import serialize_message
from tf2_ros import TransformException
ROOT=Path(__file__).resolve().parents[1]
for package in ('sorting_supervisor','sorting_perception'):
    sys.path.insert(0,str(ROOT/'src'/package/'scripts'))
from m4c_upstream import UpstreamTrace
from m4c_timing import CaptureTiming
from m4c_runtime import MovingInspectionExecutor
from handle_localizer import HandleLocalizer
from motion_filter import MotionFilter
import test_camera_ros_contracts as camera_contracts
import test_m3b_core as m3b_core
import yaml


def records(capture=40_000_000,episode=0,sequence=7):
    identity=dict(capture_ns=capture,episode_id=episode,view_id='overhead',version=1,observation_sequence=sequence)
    local=dict(identity,stage='localizer',image_arrival_wall_ns=1_000_000_000,image_arrival_sim_ns=capture+25_000_000,
        first_attempt_wall_ns=1_005_000_000,metadata_ready_wall_ns=1_006_000_000,tf_ready_wall_ns=1_016_000_000,
        localization_start_wall_ns=1_016_000_000,localization_end_wall_ns=1_017_000_000,
        image_prepare_start_wall_ns=1_006_000_000,image_prepare_end_wall_ns=1_007_000_000,tf_lookup_start_wall_ns=1_015_000_000,
        observation_publish_enter_wall_ns=1_018_000_000)
    filtered=dict(identity,stage='filter',track_id=1,filter_receive_wall_ns=1_020_000_000,
        filter_select_wall_ns=1_024_000_000,filter_update_end_wall_ns=1_025_000_000,filter_publish_enter_wall_ns=1_026_000_000)
    sample=dict(episode_id=episode,track_id=1,capture_ns=capture,receipt_wall_s=1.028)
    return local,filtered,sample


class JoinTests(unittest.TestCase):
    def setUp(self):self.trace=UpstreamTrace();self.l,self.f,self.sample=records()
    def feed(self):
        self.trace.note(self.l,'localizer',0);self.trace.note(self.f,'filter',0)
    def test_complete_join_and_stage_intervals(self):
        self.feed();r=self.trace.capture(self.sample)
        self.assertEqual(r['status'],'COMPLETE');self.assertAlmostEqual(r['intervals']['metadata_ready_to_tf_ready_wall_s'],.010)
        self.assertAlmostEqual(r['intervals']['image_to_executor_receive_wall_s'],.028)
        self.assertAlmostEqual(r['image_arrival_capture_age_sim_s'],.025)
    def test_filter_may_arrive_before_localizer_trace(self):
        self.trace.note(self.f,'filter',0);self.trace.note(self.l,'localizer',0)
        self.assertEqual(self.trace.capture(self.sample)['status'],'COMPLETE')
    def test_late_trace_completes_a_retained_sample(self):
        self.assertEqual(self.trace.decorate(self.sample)['upstream_trace']['status'],'MISSING')
        self.feed();self.assertEqual(self.trace.decorate(self.sample)['upstream_trace']['status'],'COMPLETE')
    def test_different_capture_never_substitutes(self):
        self.f['capture_ns']+=1;self.feed()
        self.assertEqual(self.trace.capture(self.sample)['missing'],['filter'])
    def test_previous_episode_is_rejected(self):
        self.trace.note(self.l,'localizer',1)
        self.assertEqual(self.trace.counts['rejected'],1);self.assertEqual(len(self.trace.rows),0)
    def test_different_sequence_is_a_mismatch(self):
        self.f['observation_sequence']+=1;self.feed()
        self.assertEqual(self.trace.capture(self.sample)['status'],'IDENTITY_MISMATCH')
    def test_different_track_is_a_mismatch(self):
        self.f['track_id']=2;self.feed();self.assertEqual(self.trace.capture(self.sample)['status'],'IDENTITY_MISMATCH')
    def test_missing_timestamp_is_explicit(self):
        del self.l['tf_ready_wall_ns'];self.feed()
        self.assertEqual(self.trace.capture(self.sample)['status'],'MISSING_TIMESTAMPS')
    def test_negative_wall_interval_is_not_clamped(self):
        self.f['filter_receive_wall_ns']=1;self.feed();r=self.trace.capture(self.sample)
        self.assertEqual(r['status'],'CLOCK_OR_ORDER_MISMATCH');self.assertTrue(r['negative_wall_intervals'])
    def test_local_ros_clock_age_remains_signed(self):
        self.l['image_arrival_sim_ns']=self.l['capture_ns']-2_000_000;self.feed()
        self.assertEqual(self.trace.capture(self.sample)['image_arrival_capture_age_sim_s'],-.002)
    def test_cache_is_bounded_and_first_failure_is_pinned(self):
        self.feed();self.trace.pin([self.sample])
        for i in range(500):
            local,filtered,sample=records(capture=100_000_000+i)
            self.trace.note(local,'localizer',0);self.trace.note(filtered,'filter',0)
        self.assertLessEqual(len(self.trace.rows),129)
        self.assertEqual(self.trace.capture(self.sample)['status'],'COMPLETE')
    def test_duplicate_does_not_overwrite_first_record(self):
        self.feed();self.trace.note(dict(self.l,image_arrival_wall_ns=2),'localizer',0)
        self.assertEqual(self.trace.capture(self.sample)['stages']['localizer'],self.l)
    def test_bad_types_are_diagnostic_only(self):
        for row in [None,[],{},dict(self.l,capture_ns='40')]:self.trace.note(row,'localizer',0)
        self.assertEqual(self.trace.counts['rejected'],4)
    def test_malformed_json_cannot_change_control_failure(self):
        n=S(timing=S(upstream=self.trace),episode=0,failure=None)
        MovingInspectionExecutor.receive_upstream_trace(n,'localizer',S(data='bad json'))
        self.assertEqual(self.trace.counts['malformed_json'],1);self.assertIsNone(n.failure)
    def test_retained_timing_sample_joins_a_later_trace(self):
        c=yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4c.yaml').read_text());t=CaptureTiming(c,0)
        with patch('m4c_timing.time.monotonic',return_value=1.028):
            t.note(dict(self.sample,valid=True),80_000_000,94_000_000)
        t.sample(94_000_000,enforce=False)
        t.upstream.note(self.l,'localizer',0);t.upstream.note(self.f,'filter',0)
        self.assertEqual(t.report()['first_delivery_violation']['upstream_trace']['status'],'COMPLETE')


class ProducerTests(unittest.TestCase):
    def localizer(self):
        p=camera_contracts.CallbackTests().probe();p.timing_trace=Mock();p.image_trace={};p.info_times={}
        p.stats['overhead'].update(received=0,last_received_ns=0,pending_replaced=0,capture_gap_estimate=0)
        p.cameras['publish_rate_hz']=25.;image=p.pending['overhead'][0];info=p.info['overhead'][1_000_000_000]
        HandleLocalizer.camera_info(p,'overhead',info);HandleLocalizer.image(p,'overhead',image)
        return p
    def test_localizer_trace_preserves_observation_fields(self):
        plain=camera_contracts.CallbackTests().probe();HandleLocalizer.process(plain)
        p=self.localizer();HandleLocalizer.process(p)
        # compute_wall_s is intentionally measured; compare all non-timing fields.
        a=p.observation_publishers['overhead'].publish.call_args.args[0];b=plain.observation_publishers['overhead'].publish.call_args.args[0]
        a.compute_wall_s=b.compute_wall_s;self.assertEqual(a,b)
        serialize_message(a);serialize_message(b)
        trace=json.loads(p.timing_trace.publish.call_args.args[0].data)
        self.assertEqual(trace['capture_ns'],1_000_000_000);self.assertEqual(trace['observation_sequence'],a.sequence)
        self.assertLessEqual(trace['image_arrival_wall_ns'],trace['localization_end_wall_ns'])
    def test_metadata_and_tf_retries_remain_in_one_capture(self):
        p=self.localizer();info=p.info['overhead'].pop(1_000_000_000)
        HandleLocalizer.process(p);self.assertEqual(p.image_trace['overhead']['metadata_waits'],1)
        HandleLocalizer.camera_info(p,'overhead',info)
        transform=p.buffer.lookup_transform.return_value
        p.buffer.lookup_transform.side_effect=[TransformException('pending'),transform]
        HandleLocalizer.process(p);HandleLocalizer.process(p)
        trace=json.loads(p.timing_trace.publish.call_args.args[0].data)
        self.assertEqual((trace['attempts'],trace['metadata_waits'],trace['tf_waits']),(3,1,1))
        self.assertEqual(trace['capture_ns'],1_000_000_000);self.assertEqual(p.image_trace,{})
    def test_new_image_replaces_its_trace_and_reset_clears_it(self):
        p=self.localizer();image=p.pending['overhead'][0];image.header.stamp.nanosec=40_000_000
        HandleLocalizer.image(p,'overhead',image);self.assertEqual(len(p.image_trace),1)
        p.last_clock_ns=2_000_000_000;HandleLocalizer.process(p)
        self.assertEqual(p.image_trace,{});self.assertEqual(p.info_times,{})
    def test_filter_trace_preserves_target_fields(self):
        h=m3b_core.ViewSelectionTests();plain=h.fixture();traced=h.fixture()
        for n in (plain,traced):n.c=dict(n.c,tracking_view='overhead',view_selection_wait_wall_s=0.)
        traced.timing_trace=Mock();m=h.observation('overhead');m.sequence=7
        for n in (plain,traced):MotionFilter.receive(n,m);MotionFilter.tick(n)
        self.assertEqual(plain.publisher.publish.call_args.args[0],traced.publisher.publish.call_args.args[0])
        serialize_message(traced.publisher.publish.call_args.args[0])
        row=json.loads(traced.timing_trace.publish.call_args.args[0].data)
        self.assertEqual(row['observation_sequence'],7);self.assertEqual(row['capture_ns'],1_000_000_000)
        self.assertLessEqual(row['filter_receive_wall_ns'],row['filter_publish_return_wall_ns'])
        MotionFilter.tick(traced);self.assertEqual(traced.timing_trace.publish.call_count,1)
    def test_filter_trace_does_not_use_wrong_view(self):
        h=m3b_core.ViewSelectionTests();n=h.fixture();n.timing_trace=Mock();n.c=dict(n.c,tracking_view='overhead')
        MotionFilter.receive(n,h.observation('wrist'));MotionFilter.tick(n)
        n.timing_trace.publish.assert_not_called()
    def test_m4c_launch_enables_both_producers(self):
        text=(ROOT/'src/sorting_supervisor/launch/panda_m4c.launch.py').read_text()
        self.assertEqual(text.count("'timing_trace':True"),2)
        self.assertIn("('/sorting/perception/filter_timing','/sorting/m4c/filter_timing')",text)


if __name__=='__main__':unittest.main()
