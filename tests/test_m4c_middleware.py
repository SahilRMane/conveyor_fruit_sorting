"""Check Jazzy's actual callback selector and system-time diagnostics without nodes."""
from functools import partial
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace as S,MethodType
import unittest
from unittest.mock import Mock,patch
from rclpy.subscription import Subscription
ROOT=Path(__file__).resolve().parents[1]
for package in ('sorting_supervisor','sorting_perception'):
    sys.path.insert(0,str(ROOT/'src'/package/'scripts'))
from m4c_upstream import camera_delivery_timing,UpstreamTrace
from handle_localizer import HandleLocalizer
import test_m4c_upstream as upstream


def metadata(**changes):
    epoch=1_790_000_000_000_000_000
    return dict(rmw_implementation='rmw_fastrtps_cpp',source_timestamp_system_ns=epoch+10_000_000,
        received_timestamp_system_ns=epoch+12_000_000,callback_entry_system_ns=epoch+40_000_000,
        callback_entry_monotonic_before_ns=900_000_000,callback_entry_monotonic_after_ns=900_000_400,
        publication_sequence_number=17,reception_sequence_number=None,
        system_monotonic_delta_ns=0,offset_comparison_uncertainty_ns=400,**changes)


class MiddlewareIntervalsTests(unittest.TestCase):
    def measure(self,**changes):
        m=metadata();m.update(changes);return camera_delivery_timing(dict(camera_delivery=m))
    def test_complete_system_clock_split(self):
        r=self.measure();self.assertEqual(r['status'],'COMPLETE')
        self.assertAlmostEqual(r['intervals']['publication_to_middleware_receipt_system_s'],.002)
        self.assertAlmostEqual(r['intervals']['middleware_receipt_to_callback_system_s'],.028)
        self.assertAlmostEqual(r['intervals']['publication_to_callback_system_s'],.030)
    def test_missing_receipt_keeps_only_available_interval(self):
        r=self.measure(received_timestamp_system_ns=0)
        self.assertEqual(r['status'],'PARTIAL');self.assertEqual(r['missing'],['received_timestamp_system_ns'])
        self.assertEqual(set(r['intervals']),{'publication_to_callback_system_s'})
    def test_zero_source_is_not_replaced_with_capture_time(self):
        r=self.measure(source_timestamp_system_ns=0)
        self.assertEqual(r['status'],'PARTIAL');self.assertNotIn('publication_to_callback_system_s',r['intervals'])
    def test_negative_source_is_unavailable(self):
        self.assertIn('source_timestamp_system_ns',self.measure(source_timestamp_system_ns=-1)['missing'])
    def test_string_timestamp_is_not_converted(self):
        r=self.measure(source_timestamp_system_ns='1790000000000000000')
        self.assertIn('source_timestamp_system_ns',r['missing'])
    def test_old_trace_has_explicit_unavailable_status(self):
        self.assertEqual(camera_delivery_timing({})['status'],'UNAVAILABLE')
    def test_malformed_metadata_stays_diagnostic(self):
        self.assertEqual(camera_delivery_timing({'camera_delivery':[]})['status'],'MALFORMED_METADATA')
    def test_unknown_middleware_retains_raw_data_without_interpretation(self):
        r=self.measure(rmw_implementation='unknown_rmw')
        self.assertEqual(r['status'],'UNVALIDATED_RMW');self.assertNotIn('intervals',r)
        self.assertEqual(r['raw']['publication_sequence_number'],17)
    def test_unavailable_sequence_remains_null(self):
        self.assertIsNone(self.measure()['raw']['reception_sequence_number'])
    def test_negative_interval_is_visible(self):
        r=self.measure(received_timestamp_system_ns=1_790_000_000_050_000_000)
        self.assertEqual(r['status'],'CLOCK_OR_ORDER_MISMATCH')
        self.assertLess(r['intervals']['middleware_receipt_to_callback_system_s'],0)
    def test_wide_clock_pair_is_not_interpreted(self):
        r=self.measure(callback_entry_monotonic_after_ns=902_000_000)
        self.assertEqual(r['status'],'CLOCK_SAMPLE_SPAN_EXCEEDED');self.assertNotIn('intervals',r)
    def test_reversed_monotonic_bracket_is_invalid(self):
        self.assertEqual(self.measure(callback_entry_monotonic_after_ns=899_000_000)['status'],'INVALID_CLOCK_SAMPLE')
    def test_system_clock_step_is_not_called_transport_latency(self):
        for delta in (-40_000_000,40_000_000):
            r=self.measure(system_monotonic_delta_ns=delta)
            self.assertEqual(r['status'],'SYSTEM_CLOCK_DISCONTINUITY');self.assertNotIn('intervals',r)
    def test_first_callback_has_no_prior_clock_comparison(self):
        self.assertEqual(self.measure(system_monotonic_delta_ns=None,offset_comparison_uncertainty_ns=None)['status'],'COMPLETE')
    def test_clock_sampling_uncertainty_is_explicit(self):
        self.assertAlmostEqual(self.measure()['clock_pair_span_wall_s'],.0000004)
    def test_delivery_fields_attach_only_to_the_exact_capture(self):
        trace=UpstreamTrace();local,filtered,sample=upstream.records()
        local['camera_delivery']=metadata();trace.note(local,'localizer',0);trace.note(filtered,'filter',0)
        result=trace.capture(sample)
        self.assertEqual(result['status'],'COMPLETE');self.assertEqual(result['camera_delivery']['status'],'COMPLETE')
        wrong=dict(sample,capture_ns=sample['capture_ns']+1)
        self.assertEqual(trace.capture(wrong)['status'],'MISSING')


class JazzyCallbackTests(unittest.TestCase):
    def fixture(self):
        p=upstream.ProducerTests().localizer()
        p.image=MethodType(HandleLocalizer.image,p);p.trace_rmw='rmw_fastrtps_cpp';p.delivery_clock_pair=None
        return p
    def test_actual_jazzy_selector_requests_message_info(self):
        callback=partial(HandleLocalizer.image_with_info,self.fixture(),'overhead')
        subscription=Subscription.__new__(Subscription);subscription.callback=callback
        self.assertEqual(subscription._callback_type,Subscription.CallbackType.WithMessageInfo)
        with self.assertRaises(TypeError):callback(object())
    def test_message_info_dict_reaches_published_trace(self):
        p=self.fixture();image=p.pending['overhead'][0]
        info=dict(source_timestamp=1_790_000_000_000_000_000,received_timestamp=1_790_000_000_001_000_000,
                  publication_sequence_number=53,reception_sequence_number=None)
        HandleLocalizer.image_with_info(p,'overhead',image,info);HandleLocalizer.process(p)
        trace=json.loads(p.timing_trace.publish.call_args.args[0].data)
        self.assertEqual(trace['capture_ns'],1_000_000_000)
        self.assertEqual(trace['camera_delivery']['publication_sequence_number'],53)
        self.assertEqual(trace['camera_delivery']['source_timestamp_system_ns'],info['source_timestamp'])
        self.assertTrue(p.observation_publishers['overhead'].publish.call_args.args[0].valid)
    def test_entry_clock_pair_is_taken_before_image_handler(self):
        p=self.fixture();image=p.pending['overhead'][0]
        ticks=itertools.count(900_000_000,100)
        with patch('handle_localizer.time.monotonic_ns',side_effect=lambda:next(ticks)),patch('handle_localizer.time.time_ns',return_value=1_790_000_000_000_000_000):
            HandleLocalizer.image_with_info(p,'overhead',image,{})
        trace=p.image_trace['overhead'];m=trace['camera_delivery']
        self.assertEqual(trace['image_arrival_wall_ns'],900_000_000)
        self.assertEqual(m['callback_entry_monotonic_after_ns'],900_000_100)
        self.assertIsNone(m['system_monotonic_delta_ns'])
    def test_adjacent_callbacks_compare_system_and_monotonic_progress(self):
        p=self.fixture();p.delivery_clock_pair=(1_790_000_000_000_000_000,860_000_050,100)
        image=p.pending['overhead'][0];ticks=itertools.count(900_000_000,100)
        with patch('handle_localizer.time.monotonic_ns',side_effect=lambda:next(ticks)),patch('handle_localizer.time.time_ns',return_value=1_790_000_000_040_000_000):
            HandleLocalizer.image_with_info(p,'overhead',image,{})
        self.assertEqual(p.image_trace['overhead']['camera_delivery']['system_monotonic_delta_ns'],0)
    def test_missing_message_info_never_changes_pending_image(self):
        p=self.fixture();image=p.pending['overhead'][0]
        HandleLocalizer.image_with_info(p,'overhead',image,None)
        self.assertIs(p.pending['overhead'][0],image)
        self.assertIsNone(p.image_trace['overhead']['camera_delivery']['source_timestamp_system_ns'])
    def test_handler_failure_is_not_hidden_by_diagnostics(self):
        p=self.fixture();p.image=Mock(side_effect=ValueError('handler failed'))
        with self.assertRaisesRegex(ValueError,'handler failed'):HandleLocalizer.image_with_info(p,'overhead',object(),{})
    def test_real_constructor_uses_info_callback_only_for_traced_overhead(self):
        p=HandleLocalizer.__new__(HandleLocalizer)
        p.create_subscription=Mock();p.create_publisher=Mock();p.create_timer=Mock();p.declare_parameter=Mock()
        p.get_parameter=Mock(side_effect=lambda name:S(value={'timing_trace':True,'camera_config':'cameras.yaml','localization_config':'m3a.yaml','process_period_s':.01}[name]))
        with patch('handle_localizer.Node.__init__',return_value=None),patch('handle_localizer.Buffer'),patch('handle_localizer.TransformListener'),patch('handle_localizer.get_rmw_implementation_identifier',return_value='rmw_fastrtps_cpp'):
            HandleLocalizer.__init__(p)
        callbacks={call.args[1]:call.args[2] for call in p.create_subscription.call_args_list}
        for topic,kind in [('/camera/overhead/image_raw',Subscription.CallbackType.WithMessageInfo),('/camera/wrist/image_raw',Subscription.CallbackType.MessageOnly)]:
            s=Subscription.__new__(Subscription);s.callback=callbacks[topic];self.assertEqual(s._callback_type,kind)


if __name__=='__main__':unittest.main()
