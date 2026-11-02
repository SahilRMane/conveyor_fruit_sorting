"""Exercise supervisor time limits without creating a ROS node."""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from test_m4b import ClassInspectionExecutor
from m4a_runtime import InspectionExecutor


class BudgetTests(unittest.TestCase):
    def node(self):
        node=object.__new__(ClassInspectionExecutor)
        node.policy_started=(1_000_000_000,0.)
        node.policy_config=dict(maximum_policy_sim_s=60.,maximum_policy_wall_s=90.)
        node.now=lambda:2_000_000_000
        return node

    def invoke(self,node,**kwargs):
        with patch.object(InspectionExecutor,'until',lambda self,predicate,**kw:predicate()):
            return node.until(lambda:True,**kwargs)

    def test_normal_budget(self):
        with patch('test_m4b.time.monotonic',return_value=1.):self.assertTrue(self.invoke(self.node()))

    def test_stopped_clock_wall_timeout(self):
        node=self.node();node.now=lambda:1_000_000_000
        with patch('test_m4b.time.monotonic',return_value=91.):
            with self.assertRaisesRegex(ValueError,'wall budget'):self.invoke(node)

    def test_simulation_timeout(self):
        node=self.node();node.now=lambda:62_000_000_000
        with patch('test_m4b.time.monotonic',return_value=1.):
            with self.assertRaisesRegex(ValueError,'simulation budget'):self.invoke(node)

    def test_cleanup_can_confirm_hold_after_timeout(self):
        node=self.node()
        with patch('test_m4b.time.monotonic',return_value=100.):self.assertTrue(self.invoke(node,check=False))

    def test_no_new_truth_or_grasp_interfaces(self):
        source=(ROOT/'src/sorting_supervisor/scripts/test_m4b.py').read_text()
        tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='create_subscription' for n in ast.walk(cls)))
        self.assertNotIn('StartCameraPickup',source)
        self.assertNotIn('GripperCommand',source)


if __name__=='__main__':unittest.main()
