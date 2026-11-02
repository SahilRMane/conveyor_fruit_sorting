"""Check reset evidence and grading without starting ROS."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src/sorting_simulation/scripts"))
from grasp_results import assess
from static_reset import ResetManager, validate_reset_record


class ResetEvidence(unittest.TestCase):
    def setUp(self):
        report = json.loads((ROOT/"results/M1C4-native-reset.json").read_text())
        self.record = report["last_reset_record"]
        self.expected = {name:self.record[name][:] for name in ("qpos", "qvel", "ctrl")}

    def test_actual_native_reset_record_passes(self):
        errors = validate_reset_record(self.record, self.expected)
        self.assertEqual(errors, {"qpos":0, "qvel":0, "ctrl":0})

    def test_old_motion_and_commands_are_rejected(self):
        for name in ("qpos", "qvel", "ctrl"):
            bad = copy.deepcopy(self.record)
            bad[name][0] += 0.01
            with self.assertRaisesRegex(RuntimeError, "restore"):
                validate_reset_record(bad, self.expected)

    def test_applied_force_is_rejected(self):
        bad = copy.deepcopy(self.record)
        bad["xfrc_applied_max_abs"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "clear"):
            validate_reset_record(bad, self.expected)

    def test_paused_switch_uses_steps_until_it_completes(self):
        events = []
        future = SimpleNamespace(complete=False)
        future.done = lambda: future.complete
        future.result = lambda: SimpleNamespace(ok=True, message="switched")
        class Client:
            def service_is_ready(self):
                return True
            def call_async(self, request):
                events.append(("switch", request.deactivate_controllers, request.strictness))
                return future
        manager = ResetManager.__new__(ResetManager)
        manager.switch_client = Client()
        manager.test = SimpleNamespace(until=lambda predicate: self.assertTrue(predicate()))
        def advance(steps):
            events.append(("steps", steps))
            future.complete = True
        manager.advance = advance
        manager.switch_while_paused([], ["panda_arm_controller", "panda_hand_controller"])
        self.assertEqual(events[0][0], "switch")
        self.assertEqual(events[1], ("steps", 10))


class Acceptance(unittest.TestCase):
    @staticmethod
    def trials(passes, failures=0):
        return [{"status":"PASS", "reset_status":"PASS"} for _ in range(passes)] + [
            {"status":"FAIL", "reset_status":"PASS"} for _ in range(failures)]

    def test_eighteen_of_twenty_passes(self):
        self.assertTrue(assess(20, self.trials(18,2))["acceptance_passed"])

    def test_seventeen_of_twenty_fails(self):
        self.assertEqual(assess(20, self.trials(17,3))["status"], "FAIL")

    def test_eighteen_of_eighteen_is_incomplete(self):
        result = assess(20, self.trials(18))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["acceptance_passed"])

    def test_one_cycle_is_only_a_smoke_test(self):
        result = assess(1, self.trials(1))
        self.assertEqual(result["status"], "SMOKE_PASS")
        self.assertFalse(result["acceptance_passed"])

    def test_failed_reset_aborts_and_preserves_missing_trials(self):
        trials = self.trials(5) + [{"status":"NOT_RUN", "reset_status":"FAIL"}]
        result = assess(20, trials, infrastructure_error="reset failed")
        self.assertEqual(result["status"], "ABORTED")
        self.assertEqual(result["not_completed_cycles"], 15)
        self.assertFalse(result["reset_verified"])

    def test_interruption_cannot_pass_acceptance(self):
        result = assess(20, self.trials(20), interrupted=True)
        self.assertEqual(result["status"], "ABORTED")
        self.assertFalse(result["acceptance_passed"])

    def test_final_reset_error_invalidates_acceptance(self):
        result = assess(20, self.trials(20), infrastructure_error="Final reset failed")
        self.assertEqual(result["status"], "ABORTED")
        self.assertFalse(result["reset_verified"])
        self.assertFalse(result["acceptance_passed"])


if __name__ == "__main__":
    unittest.main()
