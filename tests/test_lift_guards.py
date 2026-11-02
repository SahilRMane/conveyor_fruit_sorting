"""Test object-height evidence and failure handling without starting ROS."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lift_test", ROOT/"src/sorting_simulation/scripts/test_static_lift.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LiftGuards(unittest.TestCase):
    def test_hand_motion_without_object_lift_is_rejected(self):
        plan = json.loads((ROOT/"results/M1C3-plan.json").read_text())
        test = module.LiftTest.__new__(module.LiftTest)
        test.plan, test.reference, test.grasp = plan, plan["initial_state"], plan["grasp_state"]
        test.stage, test.guarded = "hold", True
        test.maximum_slip, test.maximum_placement_force = 0, 0
        state = copy.deepcopy(plan["held_state"])
        state["tray_xyz"] = test.reference["tray_xyz"][:]
        with self.assertRaisesRegex(RuntimeError, "slip|remain lifted"):
            test.check(state)

    def test_failure_pauses_and_cancels_without_opening(self):
        events = []
        class Client:
            def service_is_ready(self):
                return True
            def call_async(self, request):
                events.append("pause")
                self.request = request
                return SimpleNamespace(success=request.paused, message="paused")
        class Node:
            def get_service_names_and_types(self):
                return [("/simulation/set_pause", ["mujoco_ros2_control_msgs/srv/SetPause"])]
            def create_client(self, service_type, name):
                return Client()
        class Goal:
            def cancel_goal_async(self):
                events.append("cancel")
                return SimpleNamespace(goals_canceling=[1])
        test = module.LiftTest.__new__(module.LiftTest)
        test.node, test.active_goal, test.guarded = Node(), Goal(), True
        test.until = lambda predicate, timeout: self.assertTrue(predicate())
        test.wait_future = lambda future, timeout: future
        test.grip = lambda *args, **kwargs: self.fail("Failure handler must not open the gripper")
        result = test.stop_after_failure()
        self.assertTrue(result["paused"])
        self.assertTrue(result["cancel_requested"])
        self.assertFalse(result["automatic_open_command"])
        self.assertEqual(events, ["pause", "cancel"])


if __name__ == "__main__":
    unittest.main()
