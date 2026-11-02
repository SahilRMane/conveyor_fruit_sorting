"""Near-goal transition to wrist placement; no ROS node or simulator truth input."""
from pathlib import Path
from unittest.mock import Mock
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/sorting_control/scripts'),str(ROOT/'src/sorting_perception/scripts')]
from wrist_support import coarse_lower_transition
from camera_sort import SortController

class LowerTransitionTests(unittest.TestCase):
    def gate(self,position=(.45,.42,.646),goal=(.45,.42,.532),elapsed=.03,duration=8.,payload=.71025,loaded=5.10146,tolerance=.0015):
        return coarse_lower_transition(position,goal,elapsed,duration,payload,loaded,tolerance)
    def test_reported_airborne_unload_does_not_end_coarse_motion(self):
        d=self.gate();self.assertTrue(d['low_wrist_load']);self.assertFalse(d['ready'])
        self.assertGreater(d['distance_to_goal_m'],.11)
    def test_zero_or_negative_load_far_from_goal_is_not_floor_support(self):
        for load in (0.,-3.):self.assertFalse(self.gate(payload=load)['ready'])
    def test_time_completion_alone_does_not_end_coarse_motion(self):
        self.assertFalse(self.gate(elapsed=9.,payload=5.1)['ready'])
    def test_near_goal_unload_can_end_coarse_motion_early(self):
        d=self.gate(position=(.45,.42,.533),elapsed=7.9)
        self.assertTrue(d['ready']);self.assertEqual(d['reason'],'near_goal_low_wrist_load')
    def test_near_goal_full_load_waits_for_trajectory_completion(self):
        args=dict(position=(.45,.42,.5322),payload=5.1)
        self.assertFalse(self.gate(elapsed=7.9,**args)['ready'])
        d=self.gate(elapsed=8.1,**args);self.assertTrue(d['ready']);self.assertEqual(d['reason'],'coarse_goal_reached')
    def test_lateral_error_prevents_switch_even_at_correct_height(self):
        self.assertFalse(self.gate(position=(.46,.42,.532),elapsed=9.)['ready'])
    def test_invalid_input_cannot_trigger_transition(self):
        for args in (dict(payload=float('nan')),dict(loaded=0.),dict(tolerance=.009),dict(duration=0.),dict(position=(1.,2.))):
            with self.assertRaises(RuntimeError):self.gate(**args)
    def test_live_method_keeps_coarse_reference_after_early_force_dip(self):
        c=Mock();c.s=dict(release_motion_lease_s=60.,lower_duration_s=8.,placement_position_tolerance_m=.0015)
        c.now_ns.return_value=1_000_000_000;c.hand_pose.return_value=(np.array([.45,.42,.646]),[0.,0.,0.,1.])
        c.wrist.loaded=5.10146;c.wrist.payload.return_value=(1_030_000_000,.71025);c.result={}
        goal=[.45,.42,.532]
        def until(predicate,timeout):
            c.now_ns.return_value=1_030_000_000
            self.assertFalse(predicate());self.assertFalse(c.result['coarse_lower_transition']['near_goal'])
            self.assertIsNotNone(c.command_function)
            c.now_ns.return_value=9_100_000_000;c.hand_pose.return_value=(np.array(goal),[0.,0.,0.,1.])
            c.wrist.payload.return_value=(9_100_000_000,5.10146)
            self.assertTrue(predicate())
        c.until.side_effect=until
        SortController.approach_bin_with_wrist(c,goal,[0.,0.,0.,1.])
        c.lower_to_wrist_support.assert_called_once()
        self.assertEqual(c.result['coarse_lower_transition']['reason'],'coarse_goal_reached')
    def test_native_actor_uses_same_transition_rule(self):
        source=(ROOT/'src/sorting_perception/scripts/native_sort.py').read_text()
        self.assertIn('coarse_lower_transition(plant.feedback()[1:4],goal,elapsed',source)
        self.assertNotIn('contact=lambda:sign*',source)

if __name__=='__main__':unittest.main()
