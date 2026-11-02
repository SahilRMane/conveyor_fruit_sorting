"""M4C identity, capture binding and conservative deadline decisions."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4c_core import MovingPolicy,remaining,required_time,bind_capture


class M4CCoreTests(unittest.TestCase):
    def setUp(self):
        self.c=yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4c.yaml').read_text())
        self.fc=yaml.safe_load((ROOT/'src/sorting_perception/config/m3b_filter.yaml').read_text())
        self.costs=dict(initial_assess=11.,inspect_left=14.,return_assess=15.)
        self.b=dict(valid=True,camera_only=True,view_id='overhead',frame_id='world',episode_id=7,track_id=3,
                    capture_ns=1_000_000_000,state=[.54,0.,.0015,0.],covariance=(np.eye(4)*1e-7).reshape(-1).tolist())
        self.policy=MovingPolicy(7,3,1_000_000_000,self.c,self.costs)

    def observe(self, probabilities, **kwargs):
        args=dict(episode=7,track_id=3,pose='assess',capture_ns=1_100_000_000,
                  now_ns=1_120_000_000,probabilities=probabilities,horizon=60.)
        args.update(kwargs);return self.policy.observe(**args)

    def test_clear_red_routes_to_a(self):
        r=self.observe([.99,.01]);self.assertEqual((r['action'],r['proposed_bin']),('PICK_READY','bin_a'))
        self.assertFalse(r['physical_pickup_permitted']);self.assertFalse(r['physical_pickup_timing_validated'])

    def test_clear_green_routes_to_b(self):
        self.assertEqual(self.observe([.01,.99])['proposed_bin'],'bin_b')

    def test_uncertain_requests_one_view(self): self.assertEqual(self.observe([.5,.5])['action'],'INSPECT')

    def test_inspection_resolves_green_on_same_track(self):
        self.observe([.5,.5]);self.policy.start_inspection(1_150_000_000)
        r=self.observe([.01,.99],pose='inspect_left',capture_ns=1_200_000_000,now_ns=1_220_000_000)
        self.assertEqual((r['action'],r['track_id']),('PICK_READY',3))

    def test_unresolved_view_declines_pickup(self):
        self.observe([.5,.5]);self.policy.start_inspection(1_150_000_000)
        r=self.observe([.5,.5],pose='inspect_left',capture_ns=1_200_000_000,now_ns=1_220_000_000)
        self.assertEqual((r['action'],r['reason']),('PASS','uncertain_after_one_extra_view'))

    def test_inspection_cost_includes_return_and_pick_allocation(self):
        total=14.+self.c['capture_budget_s']+15.+self.c['approach_allocation_s']+self.c['close_s']+self.c['margin_s']
        self.assertAlmostEqual(required_time('INSPECT','assess',self.costs,self.c),total)

    def test_exact_deadline_is_not_feasible(self):
        needed=required_time('INSPECT','assess',self.costs,self.c)
        r=self.observe([.5,.5],horizon=needed)
        self.assertEqual(r['action'],'PASS');self.assertNotIn('selected_view',r)
        self.assertEqual(self.policy.state,'PASS');self.assertEqual(self.policy.history[-1]['to_state'],'PASS')

    def test_confident_target_still_requires_pick_slack(self):
        r=self.observe([.99,.01],horizon=1.)
        self.assertEqual(r['action'],'PASS');self.assertNotIn('proposed_bin',r);self.assertNotIn('class_id',r)

    def test_no_second_view(self):
        self.observe([.5,.5]);self.policy.start_inspection(1_150_000_000)
        with self.assertRaisesRegex(ValueError,'Only one'):self.policy.start_inspection(1_160_000_000)

    def test_wrong_track_fails(self):
        with self.assertRaisesRegex(ValueError,'identity changed'):self.observe([.99,.01],track_id=4)

    def test_wrong_episode_fails(self):
        with self.assertRaisesRegex(ValueError,'Episode changed'):self.observe([.99,.01],episode=8)

    def test_stale_class_score_declines_pickup(self):
        self.assertEqual(self.observe([.99,.01],now_ns=1_300_000_000)['action'],'PASS')

    def test_uncertainty_reduces_exit_horizon(self):
        baseline=remaining(self.b,1_020_000_000,self.c,self.fc)['remaining_s']
        other=copy.deepcopy(self.b);other['covariance']=(np.eye(4)*1e-5).reshape(-1).tolist()
        self.assertLess(remaining(other,1_020_000_000,self.c,self.fc)['remaining_s'],baseline)

    def test_zero_speed_has_capped_horizon(self):
        self.b['state'][2]=0;self.b['covariance']=[0.]*16
        r=remaining(self.b,1_000_000_000,self.c,self.fc)
        self.assertEqual(r['remaining_s'],self.c['maximum_horizon_s'])

    def test_late_target_has_short_horizon(self):
        self.b['state'][0]=.749
        self.assertLess(remaining(self.b,1_020_000_000,self.c,self.fc)['remaining_s'],1.)

    def test_future_overhead_stamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'stale or future'):remaining(self.b,999_000_000,self.c,self.fc)

    def test_stale_overhead_stamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'stale or future'):remaining(self.b,1_150_000_000,self.c,self.fc)

    def test_bad_covariance_is_rejected(self):
        self.b['covariance'][0]=-1.
        with self.assertRaisesRegex(ValueError,'covariance'):remaining(self.b,1_020_000_000,self.c,self.fc)

    def test_reverse_transport_is_rejected(self):
        self.b['state'][2]=-.02
        with self.assertRaisesRegex(ValueError,'Reverse'):remaining(self.b,1_020_000_000,self.c,self.fc)

    def test_wrong_coordinate_frame_is_rejected(self):
        self.b['frame_id']='panda_hand'
        with self.assertRaisesRegex(ValueError,'frame'):remaining(self.b,1_020_000_000,self.c,self.fc)

    def test_wrist_binds_at_capture_time(self):
        s=dict(episode_id=7,capture_ns=1_100_000_000,handle_xyz=[.54015,0.,.4229])
        r=bind_capture(s,[self.b],7,3,self.c,self.fc)
        self.assertLess(r['binding_error_m'],1e-9);self.assertEqual(r['track_id'],3)

    def test_binding_does_not_use_a_future_overhead_state(self):
        b=copy.deepcopy(self.b);b['capture_ns']=1_110_000_000
        s=dict(episode_id=7,capture_ns=1_100_000_000,handle_xyz=[.54015,0.,.4229])
        with self.assertRaisesRegex(ValueError,'No earlier'):bind_capture(s,[b],7,3,self.c,self.fc)

    def test_binding_rejects_other_track_and_episode(self):
        s=dict(episode_id=7,capture_ns=1_100_000_000,handle_xyz=[.54015,0.,.4229])
        with self.assertRaisesRegex(ValueError,'No earlier'):bind_capture(s,[self.b],7,4,self.c,self.fc)
        s['episode_id']=8
        with self.assertRaisesRegex(ValueError,'another episode'):bind_capture(s,[self.b],7,3,self.c,self.fc)

    def test_binding_rejects_large_spatial_error(self):
        s=dict(episode_id=7,capture_ns=1_100_000_000,handle_xyz=[.55,0.,.4229])
        with self.assertRaisesRegex(ValueError,'does not match'):bind_capture(s,[self.b],7,3,self.c,self.fc)


if __name__=='__main__':unittest.main()
