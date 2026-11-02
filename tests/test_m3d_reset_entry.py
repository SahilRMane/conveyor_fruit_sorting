"""Replay reset-entry callback ordering and retain physical checks; no ROS node."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import copy,json,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/sorting_control/scripts'),str(ROOT/'src/sorting_perception/scripts')]
from std_msgs.msg import String
from m3b_evaluator import Evaluator
from m3d_evaluator import SortEvaluator
from replay_m3d_reset import observer,state,reset_entry

class ResetEntryTests(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((ROOT/'tests/fixtures/m3d_reset_entry.json').read_text())
    def evaluator(self,phase='verify_placement'):
        e=observer();e.grasp=copy.deepcopy(self.data['grasp']);e.decision=SimpleNamespace(proposed_bin='bin_a')
        state(e,phase);e.contact=copy.deepcopy(self.data['placed']);return e
    def test_released_tray_reaches_reset_guard_with_checks_still_active(self):
        e=self.evaluator();result=reset_entry(e,self.data['placed'])
        self.assertEqual(result['reached'],'reset_guard_request');self.assertTrue(result['guarded'])
        self.assertEqual(result['phase'],'verify_placement');self.assertEqual(e.grip_trace,[])
    def test_old_early_clear_reproduces_exact_false_slip_failure(self):
        def old_reset(e):
            e.last_placement_phase=None;e.placement_started=False;e.release_seen=False
            return Evaluator.reset_episode(e)
        with patch.object(SortEvaluator,'reset_episode',old_reset):
            with self.assertRaisesRegex(RuntimeError,'Physical tray slip in hand frame exceeds 3 mm'):
                reset_entry(self.evaluator(),self.data['placed'])
    def test_forbidden_contact_is_not_hidden_while_reset_authorization_is_pending(self):
        value=copy.deepcopy(self.data['placed']);value['forbidden_contacts']=1
        with self.assertRaisesRegex(RuntimeError,'Forbidden placement contact'):reset_entry(self.evaluator(),value)
    def test_bin_overload_is_not_hidden_while_reset_authorization_is_pending(self):
        value=copy.deepcopy(self.data['placed']);value['bin_floor_force_N'][0]=10.3
        with self.assertRaisesRegex(RuntimeError,'twice payload weight'):reset_entry(self.evaluator(),value)
    def test_wrong_bin_contact_is_not_hidden_before_reset(self):
        value=copy.deepcopy(self.data['placed']);value['bin_floor_force_N'][1]=.2
        with self.assertRaisesRegex(RuntimeError,'unselected bin'):reset_entry(self.evaluator(),value)
    def test_actual_held_slip_still_fails_if_reset_is_requested_during_lowering(self):
        value=copy.deepcopy(self.data['lower']);value['hand_xyz'][2]+=.004
        with self.assertRaisesRegex(RuntimeError,'Physical tray slip in hand frame exceeds 3 mm'):
            reset_entry(self.evaluator('lower'),value)
    def test_failed_reset_guard_request_preserves_placement_phase(self):
        e=self.evaluator();e.reset_guard=Mock();e.owner=dict(owner='HOLD',busy=False)
        def wait(predicate,timeout=30.):state(e,'idle');e.check(self.data['placed']);self.assertTrue(predicate())
        e.until=wait;e.call=Mock(side_effect=TimeoutError('reset guard deadline'))
        with self.assertRaisesRegex(TimeoutError,'reset guard deadline'):e.reset_episode()
        self.assertEqual(e.last_placement_phase,'verify_placement');self.assertTrue(e.guarded)
    def test_real_reset_sequence_handles_callbacks_before_and_after_world_reset(self):
        e=self.evaluator();e.owner=dict(owner='HOLD',busy=False,ready=True)
        e.reset_guard=Mock();e.q={};e.v={};e.truth={};e.pending_errors={};e.filter_errors=[]
        e.scene=dict(valid=True,episode_id=1,estimated_attachment=False)
        e.current=lambda:SimpleNamespace(episode_id=1)
        authorization=[];world_reset=[False]
        def spin(node,timeout_sec):
            state(e,'idle')
            value=self.data['home'] if world_reset[0] else self.data['placed']
            e.receive_contact(String(data=json.dumps(value)))
        def call(client,request):
            self.assertIs(client,e.reset_guard);authorization.append(request.data)
            if request.data:
                self.assertTrue(e.guarded);self.assertEqual(e.phase,'verify_placement')
            return SimpleNamespace(success=True,message='test response')
        def reset_home():
            e.prepare_for_reset();self.assertTrue(e.reset_in_progress);self.assertFalse(e.guarded)
            world_reset[0]=True;spin(None,0.)
            self.assertEqual(e.phase,'reset');self.assertIsNone(e.grasp)
            # The real reset manager re-enables contact checks before returning.
            e.guarded=True;e.check(self.data['home'])
            return dict(status='PASS',reset_seq=1)
        e.call=call;e.reset_manager=SimpleNamespace(reset_home=reset_home)
        with patch('test_arm_ownership.rclpy.spin_once',side_effect=spin):result=e.reset_episode()
        self.assertEqual(authorization,[True,False]);self.assertEqual(result['status'],'PASS')
        self.assertFalse(e.reset_in_progress);self.assertTrue(e.guarded)
        self.assertIsNone(e.last_placement_phase);state(e,'idle');e.check(self.data['home'])
        e.prepare_trial();self.assertIsNone(e.grasp);self.assertEqual(e.maximum_slip,0.)
    def test_completed_base_reset_clears_phase_before_next_episode(self):
        e=self.evaluator();e.scene_clients=[]
        with patch.object(Evaluator,'reset_episode',return_value=dict(status='PASS')):result=e.reset_episode()
        self.assertEqual(result['status'],'PASS');self.assertIsNone(e.last_placement_phase)
        self.assertFalse(e.placement_started);self.assertFalse(e.release_seen)
        state(e,'idle');self.assertEqual(e.phase,'idle')

if __name__=='__main__':unittest.main()
