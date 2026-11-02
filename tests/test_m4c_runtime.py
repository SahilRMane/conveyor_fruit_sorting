"""Exercise live wrapper guards without constructing a ROS node."""
import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as S
from unittest.mock import patch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4c_runtime import MovingInspectionExecutor
from m4c_core import DeadlinePass
from m4a_runtime import InspectionExecutor
from test_m4b import ClassInspectionExecutor
from sorting_interfaces.msg import TargetBelief
from m4a_runtime import stamp
from collections import deque


class M4CRuntimeTests(unittest.TestCase):
    def node(self):
        n=object.__new__(MovingInspectionExecutor);n.tracking_active=True;n.operation=None;n.policy_started=None
        n.mc=yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4c.yaml').read_text())
        n.costs=dict(initial_assess=11.,inspect_left=14.,return_assess=15.)
        n.events=[];n.now=lambda:2_000_000_000;n.target=lambda:dict(remaining_s=60.)
        return n

    def wait(self,n,**kwargs):
        with patch.object(InspectionExecutor,'until',lambda self,predicate,**kw:predicate()):
            return n.until(lambda:True,**kwargs)

    def test_fresh_target_allows_wait(self):self.assertTrue(self.wait(self.node()))

    def test_source_failure_stops_active_wait(self):
        n=self.node();n.target=lambda:(_ for _ in ()).throw(ValueError('Overhead feedback source stopped'))
        with self.assertRaisesRegex(ValueError,'source stopped'):self.wait(n)

    def test_cleanup_bypasses_failed_target(self):
        n=self.node();n.target=lambda:(_ for _ in ()).throw(ValueError('source stopped'))
        self.assertTrue(self.wait(n,check=False))

    def test_dynamic_deadline_declines_target(self):
        n=self.node();n.operation=dict(begin_ns=1_000_000_000,budget_s=10.,tail_s=15.)
        n.target=lambda:dict(remaining_s=20.)
        with self.assertRaisesRegex(DeadlinePass,'deadline_changed'):self.wait(n)

    def test_cost_overrun_is_a_failure_not_successful_pass(self):
        n=self.node();n.operation=dict(begin_ns=1_000_000_000,budget_s=.5,tail_s=0.)
        with self.assertRaisesRegex(ValueError,'movement budget') as result:self.wait(n)
        self.assertNotIsInstance(result.exception,DeadlinePass)

    def test_operation_checks_inspection_return_and_pick_reserve(self):
        n=self.node();n.target=lambda:dict(remaining_s=20.)
        with self.assertRaisesRegex(DeadlinePass,'insufficient_slack'):n.begin_operation('inspect_left',capture=True,returning=True)

    def test_arm_lease_does_not_extend_past_exit_margin(self):
        n=self.node();n.target=lambda:dict(remaining_s=20.)
        with patch.object(ClassInspectionExecutor,'acquire',return_value='bounded') as acquire:
            self.assertEqual(n.acquire('SERVO',35.,1),'bounded')
        acquire.assert_called_once_with('SERVO',20.-n.mc['margin_s'],1)

    def test_cleanup_disables_tracking_before_hold(self):
        n=self.node();n.operation={'active':True}
        with patch.object(ClassInspectionExecutor,'stop') as stop:n.stop()
        self.assertFalse(n.tracking_active);self.assertIsNone(n.operation);stop.assert_called_once()

    def test_no_truth_or_pickup_interfaces_in_executor(self):
        source=(ROOT/'src/sorting_supervisor/scripts/m4c_runtime.py').read_text()
        self.assertNotIn('/sorting/ground_truth',source)
        self.assertNotIn('/sorting/contact_state',source)
        self.assertNotIn('StartCameraPickup',source);self.assertNotIn('GripperCommand',source)
        tree=ast.parse(source)
        topics=[n.args[1].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and n.func.attr=='create_subscription' and isinstance(n.args[1],ast.Constant)]
        self.assertEqual(topics,['/sorting/m4c/targets'])

    def receiver(self):
        n=self.node();n.episode=7;n.bound_track=3;n.overhead=None;n.overhead_wall=0.
        n.history=deque(maxlen=20);n.failure=None;n.overhead_frames=0
        b=TargetBelief();b.episode_id=7;b.track_id=3;b.header.stamp=stamp(1_900_000_000)
        b.header.frame_id='world'
        b.view_id='overhead';b.camera_only=True;b.valid=True;b.state=[.54,0.,.0015,0.]
        b.covariance=[1e-7 if i%5==0 else 0. for i in range(16)]
        return n,b

    def test_duplicate_belief_does_not_refresh_source_watchdog(self):
        n,b=self.receiver()
        with patch('m4c_runtime.time.monotonic',return_value=100.):n.receive_targets(S(episode_id=7,targets=[b]))
        with patch('m4c_runtime.time.monotonic',return_value=100.4):n.receive_targets(S(episode_id=7,targets=[b]))
        self.assertEqual(n.overhead_wall,100.);self.assertEqual(len(n.history),1)

    def test_track_reassignment_latches_failure(self):
        n,b=self.receiver();b.track_id=4;n.receive_targets(S(episode_id=7,targets=[b]))
        self.assertIn('identity changed',n.failure)

    def test_old_episode_cannot_replace_current_belief(self):
        n,b=self.receiver();b.episode_id=6;n.receive_targets(S(episode_id=6,targets=[b]))
        self.assertIsNone(n.overhead)


if __name__=='__main__':unittest.main()
