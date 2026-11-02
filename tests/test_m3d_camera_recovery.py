"""Replay the user's camera stall with real messages and mocked transport."""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import yaml
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/sorting_control/scripts'),str(R/'src/sorting_perception/scripts'),str(R/'tests')]
import test_m3b_core as baseline
from camera_sort import SortController
from camera_pickup import CameraCaptureWait
from m2b_common import timestamp

class RecoveryTests(unittest.TestCase):
    def fixture(self,updated=True):
        c=baseline.StationaryCameraWaitTests().fixture()
        c.scene.update(episode_id=2,track_id=7)
        if updated:
            c.__class__=SortController
            c.s=yaml.safe_load((R/'src/sorting_control/config/m3d.yaml').read_text())
            c.configure_stationary_camera_recovery()
        return c
    def begin(self,c):
        with patch('camera_pickup.time.monotonic',return_value=10.):c.send()
        c.command_function.assert_not_called()
        m=c.publisher.publish.call_args.args[0]
        self.assertEqual([m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z,
            m.twist.twist.angular.x,m.twist.twist.angular.y,m.twist.twist.angular.z],[0.]*6)
    def fresh(self,c,age_ns=116_000_000,elapsed_ns=102_000_000):
        c.now['ns']=1_154_000_000+elapsed_ns
        c.belief.header.stamp=timestamp(c.now_ns()-age_ns)
        c.received=10.+elapsed_ns/1e9
        c.scene.update(valid=True,camera_only=True,capture_age_s=age_ns/1e9,
            time_ns=c.now_ns()-4_000_000,model_capture_ns=c.now_ns()-age_ns,reason='camera_geometry')
    def replay(self,c):
        f=json.loads((R/'tests/fixtures/m3d-stationary-resume-failure.json').read_text())
        start=f['camera_wait_trace'][0];end=f['failure_diagnostic']
        c.now['ns']=start['time_ns'];c.input_deadline=c.now_ns()+20_000_000_000
        c.belief.header.stamp=timestamp(start['capture_ns']);self.begin(c)
        c.now['ns']=end['time_ns'];c.belief.header.stamp=timestamp(end['belief_capture_ns']);c.received=10.102
        c.scene=copy.deepcopy(end['scene']);c.scene.update(episode_id=2,track_id=7)
        return c
    def test_recorded_failure_is_reproduced_by_previous_policy(self):
        c=self.replay(self.fixture(False))
        with patch('camera_pickup.time.monotonic',return_value=10.102):
            with self.assertRaisesRegex(RuntimeError,'exceeded its bound'):c.send()
        c.command_function.assert_not_called()
    def test_recorded_116ms_capture_resumes_within_new_bounded_wait(self):
        c=self.replay(self.fixture())
        with patch('camera_pickup.time.monotonic',return_value=10.102):c.send()
        self.assertIsNone(c.camera_wait);c.command_function.assert_called_once()
        self.assertEqual(c.events[-1]['phase'],'stationary_camera_resume')
        self.assertAlmostEqual(c.camera_wait_total_wall_s,.102)
    def test_initial_start_still_requires_less_than_90ms(self):
        c=self.fixture();self.fresh(c)
        with patch('camera_pickup.time.monotonic',return_value=10.102):
            self.assertFalse(c.refinement_ready())
        self.assertEqual(c.c['pregrasp_start_max_capture_age_s'],.09)
        self.assertEqual(c.c['maximum_capture_age_s'],.15)
    def test_new_belief_requires_scene_from_that_capture_or_later(self):
        c=self.fixture();self.begin(c);self.fresh(c);c.scene['model_capture_ns']-=1
        with patch('camera_pickup.time.monotonic',return_value=10.102):c.send()
        self.assertIsNotNone(c.camera_wait);c.command_function.assert_not_called()
    def test_scene_episode_track_flags_and_age_block_resume(self):
        for key,value in [('episode_id',3),('track_id',8),('valid',False),('camera_only',False),('time_ns',1_156_000_000)]:
            with self.subTest(key=key):
                c=self.fixture();self.begin(c);self.fresh(c);c.scene[key]=value
                with patch('camera_pickup.time.monotonic',return_value=10.102):self.assertFalse(c.camera_resume_ready())
                c.command_function.assert_not_called()
    def test_stale_capture_cannot_resume_even_with_extended_wait(self):
        for age in (150_000_000,160_000_000):
            c=self.fixture();self.begin(c);self.fresh(c,age_ns=age)
            with patch('camera_pickup.time.monotonic',return_value=10.102):c.send()
            self.assertIsNotNone(c.camera_wait);c.command_function.assert_not_called()
    def test_capture_must_advance_after_braking(self):
        c=self.fixture();c.now['ns']=1_050_000_000;c.scene.update(valid=True,time_ns=c.now_ns())
        c.camera_wait=dict(capture_ns=1_000_000_000)
        with patch('camera_pickup.time.monotonic',return_value=10.):self.assertFalse(c.camera_resume_ready())
    def test_wait_sim_wall_and_total_limits_remain_hard_bounds(self):
        for mutation in ('sim','wall','total'):
            with self.subTest(mutation=mutation):
                c=self.fixture();self.begin(c);self.fresh(c)
                wall=10.102
                if mutation=='sim':self.fresh(c,elapsed_ns=200_000_000);wall=10.2
                elif mutation=='wall':wall=10.241
                else:c.camera_wait_total_wall_s=.30
                with patch('camera_pickup.time.monotonic',return_value=wall):
                    with self.assertRaisesRegex(RuntimeError,'exceeded its bound'):c.send()
                c.command_function.assert_not_called()
    def test_moving_phases_still_abort_on_stale_images(self):
        for phase in ('track','close','lift'):
            c=self.fixture();c.phase=phase
            with patch('camera_pickup.time.monotonic',return_value=10.):
                with self.assertRaises(CameraCaptureWait):c.send()
            c.publisher.publish.assert_not_called()
    def test_stationary_resume_requires_open_empty_gripper_and_unchanged_owner(self):
        for mutation in ('belt','finger','payload','owner','lease'):
            c=self.fixture();self.begin(c);self.fresh(c)
            if mutation=='belt':c.belt_speed=.03
            elif mutation=='finger':c.q['panda_finger_joint1']=.01
            elif mutation=='payload':c.scene['estimated_attachment']=True
            elif mutation=='owner':c.owner['epoch']+=1
            else:c.input_deadline=c.now_ns()
            with patch('camera_pickup.time.monotonic',return_value=10.102):
                with self.assertRaises(RuntimeError):c.send()
            c.command_function.assert_not_called()
    def test_invalid_resume_configuration_cannot_increase_image_age(self):
        for value in (.151,float('nan'),0.,.08):
            c=self.fixture();c.s['stationary_camera_recovery']['resume_max_capture_age_s']=value
            with self.assertRaises(RuntimeError):c.configure_stationary_camera_recovery()
    def test_invalid_wait_configuration_is_rejected(self):
        for key,value in [('maximum_wait_sim_s',0.),('maximum_wait_wall_s',float('inf')),('maximum_total_wait_wall_s',.1)]:
            c=self.fixture();c.s['stationary_camera_recovery'][key]=value
            with self.assertRaises(RuntimeError):c.configure_stationary_camera_recovery()
    def test_earlier_checkpoint_policy_retains_original_limits(self):
        c=self.fixture(False);self.begin(c);self.fresh(c,age_ns=116_000_000,elapsed_ns=80_000_000)
        with patch('camera_pickup.time.monotonic',return_value=10.08):c.send()
        self.assertIsNotNone(c.camera_wait);c.command_function.assert_not_called()
        self.assertEqual(c.c['maximum_stationary_camera_wait_sim_s'],.10)

if __name__=='__main__':unittest.main()
