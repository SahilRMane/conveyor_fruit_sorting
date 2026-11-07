"""Check camera-stage matching, stamp consistency, clock separation and rollout scope."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4c_upstream import UpstreamTrace,camera_side_timing
import test_m4c_upstream as upstream
import test_m4c_middleware as middleware


def camera_record(capture=40_000_000):
    return dict(stage='camera',version=1,view_id='overhead',episode_id=0,capture_ns=capture,
        source_data_ns=capture,render_data_begin_ns=capture,render_data_end_ns=capture,
        rate_clock_sim_ns=capture+2_000_000,image_publish_ros_ns=capture+26_000_000,
        snapshot_policy='worker_private_copy',snapshot_serial=3,pending_replacements=0,
        selected_count=2,camera_order=0,update_enter_wall_ns=900_000_000,
        snapshot_lock_acquired_wall_ns=900_001_000,snapshot_accept_wall_ns=900_002_000,
        snapshot_copy_end_wall_ns=900_010_000,worker_take_wall_ns=901_000_000,
        worker_copy_begin_wall_ns=901_001_000,worker_copy_end_wall_ns=901_010_000,
        render_pass_begin_wall_ns=901_020_000,camera_render_begin_wall_ns=901_030_000,
        scene_ready_wall_ns=902_000_000,draw_return_wall_ns=905_000_000,
        pixels_ready_wall_ns=925_000_000,format_ready_wall_ns=925_050_000,
        image_publish_enter_wall_ns=925_060_000,image_publish_return_wall_ns=925_100_000,
        metadata_publish_return_wall_ns=925_150_000,render_thread_cpu_begin_ns=3_000_000_000,
        render_thread_cpu_end_ns=3_001_000_000,image_publish_enter_system_ns=1_790_000_000_925_060_000)


class CameraSideTests(unittest.TestCase):
    def setUp(self):self.c=camera_record();self.l,self.f,self.sample=upstream.records()
    def test_camera_intervals_remain_separate_from_simulation_age(self):
        r=camera_side_timing(self.c,self.l)
        self.assertEqual(r['status'],'COMPLETE')
        self.assertAlmostEqual(r['intervals']['pixel_readback_wall_s'],.020)
        self.assertAlmostEqual(r['control_snapshot_age_at_rate_check_sim_s'],.002)
        self.assertAlmostEqual(r['capture_age_at_image_publish_sim_s'],.026)
        self.assertAlmostEqual(r['render_thread_cpu_s'],.001)
    def test_pending_queue_and_worker_copy_are_visible(self):
        r=camera_side_timing(self.c,self.l)
        self.assertAlmostEqual(r['intervals']['render_queue_wait_wall_s'],.000990)
        self.assertAlmostEqual(r['intervals']['worker_snapshot_copy_wall_s'],.000009)
    def test_absent_camera_stage_is_explicit(self):
        self.assertEqual(camera_side_timing(None,self.l)['status'],'MISSING')
    def test_source_stamp_mismatch_is_not_a_valid_timing_sample(self):
        self.c['source_data_ns']+=1
        self.assertEqual(camera_side_timing(self.c,self.l)['status'],'STAMP_MISMATCH')
    def test_render_data_cannot_change_timestamp(self):
        self.c['render_data_end_ns']+=1
        self.assertEqual(camera_side_timing(self.c,self.l)['status'],'STAMP_MISMATCH')
    def test_unknown_snapshot_policy_is_explicit(self):
        self.c['snapshot_policy']='unknown'
        self.assertEqual(camera_side_timing(self.c,self.l)['status'],'SNAPSHOT_POLICY_MISMATCH')
    def test_missing_stage_timestamp_is_explicit(self):
        del self.c['pixels_ready_wall_ns']
        self.assertEqual(camera_side_timing(self.c,self.l)['status'],'MISSING_TIMESTAMPS')
    def test_negative_wall_interval_is_retained(self):
        self.c['worker_take_wall_ns']=1;r=camera_side_timing(self.c,self.l)
        self.assertEqual(r['status'],'CLOCK_OR_ORDER_MISMATCH')
        self.assertLess(r['intervals']['render_queue_wait_wall_s'],0)
    def test_cpu_clock_unavailability_remains_null(self):
        self.c['render_thread_cpu_begin_ns']=0
        self.assertIsNone(camera_side_timing(self.c,self.l)['render_thread_cpu_s'])
    def test_middleware_bridge_uses_only_system_timestamps(self):
        self.l['camera_delivery']=middleware.metadata()
        start=self.c['image_publish_enter_system_ns'];self.l['camera_delivery'].update(source_timestamp_system_ns=start+10_000,
            received_timestamp_system_ns=start+20_000,callback_entry_system_ns=start+30_000)
        r=camera_side_timing(self.c,self.l)
        self.assertEqual(r['status'],'COMPLETE');self.assertAlmostEqual(r['publish_entry_to_middleware_source_system_s'],.00001)
    def test_unvalidated_middleware_is_not_used_for_a_bridge(self):
        self.l['camera_delivery']=dict(middleware.metadata(),rmw_implementation='unsupported')
        self.assertNotIn('publish_entry_to_middleware_source_system_s',camera_side_timing(self.c,self.l))
    def test_camera_trace_accepts_its_own_snapshot_serial(self):
        trace=UpstreamTrace()
        for row in (self.c,self.l,self.f):trace.note(row,row['stage'],0)
        r=trace.capture(self.sample)
        self.assertEqual(r['status'],'COMPLETE');self.assertEqual(r['camera_side']['status'],'COMPLETE')
        self.assertEqual(trace.counts['camera'],1)
    def test_camera_trace_can_arrive_after_the_target(self):
        trace=UpstreamTrace();trace.note(self.l,'localizer',0);trace.note(self.f,'filter',0);trace.pin([self.sample])
        self.assertEqual(trace.capture(self.sample)['camera_side']['status'],'MISSING')
        trace.note(self.c,'camera',0);self.assertEqual(trace.capture(self.sample)['camera_side']['status'],'COMPLETE')
    def test_neighbouring_capture_cannot_supply_camera_timing(self):
        trace=UpstreamTrace();self.c['capture_ns']+=1
        for row in (self.c,self.l,self.f):trace.note(row,row['stage'],0)
        self.assertEqual(trace.capture(self.sample)['camera_side']['status'],'MISSING')
    def test_old_episode_camera_trace_is_rejected(self):
        trace=UpstreamTrace();self.c['episode_id']=1;trace.note(self.c,'camera',0)
        self.assertEqual(trace.counts['rejected'],1)
    def test_retained_camera_trace_survives_cache_eviction(self):
        trace=UpstreamTrace()
        for row in (self.c,self.l,self.f):trace.note(row,row['stage'],0)
        trace.pin([self.sample])
        for i in range(300):trace.note(camera_record(80_000_000+i),'camera',0)
        self.assertLessEqual(len(trace.rows),129)
        self.assertEqual(trace.capture(self.sample)['camera_side']['status'],'COMPLETE')
    def test_unknown_trace_stage_is_rejected(self):
        trace=UpstreamTrace();trace.note(dict(self.c,stage='unknown'),'unknown',0)
        self.assertEqual(trace.counts['rejected'],1)
    def test_only_m4c_preflight_selects_the_project_camera(self):
        import yaml
        baseline=yaml.safe_load((ROOT/'src/sorting_simulation/config/camera_plugins_m3d.yaml').read_text())
        self.assertEqual(baseline['/**']['ros__parameters']['mujoco_plugins']['camera_plugin']['type'],'mujoco_ros2_control_plugins/CameraPlugin')
        preflight=(ROOT/'src/sorting_supervisor/scripts/validate_m4c.py').read_text()
        self.assertIn("camera['type']='sorting_simulation/TracedCameraPlugin'",preflight)
    def test_camera_binary_and_factory_xml_are_bound_to_preflight(self):
        text=(ROOT/'src/sorting_supervisor/scripts/m4c_common.py').read_text()
        self.assertIn('libsorting_traced_camera.so',text);self.assertIn('traced_camera_plugin.xml',text)
    def test_pinned_source_license_and_provenance_are_present(self):
        p=ROOT/'src/sorting_simulation';provenance=json.loads((p/'camera_trace_provenance.json').read_text())
        self.assertEqual(provenance['version'],'0.1.2');self.assertEqual(provenance['license'],'Apache-2.0')
        self.assertIn('Apache License',(p/'CAMERA-TRACE-LICENSE').read_text())


if __name__=='__main__':unittest.main()
