"""Check the learned decision before the executor can command a view."""

import unittest

import run_episode  # Resolve the checked ROS package and pilot paths.
from learned_inspection import assess


class PolicyTest(unittest.TestCase):
    def test_confident_view_requires_pickup_time_only(self):
        costs = {'inspect_left': 30., 'inspect_right': 25.}
        self.assertEqual(assess([.95, .05], .8, 12., 11., costs, None)[0], 'PICK_READY')
        self.assertEqual(assess([.95, .05], .8, 11., 11., costs, None)[0], 'PASS')

    def test_inspection_requires_positive_utility_and_slack(self):
        costs = {'inspect_left': 30., 'inspect_right': 25.}
        probabilities = [.52, .48]
        self.assertEqual(assess(probabilities, .8, 50., 11., costs, [.9, .7])[1], 'inspect_left')
        self.assertEqual(assess(probabilities, .8, 41., 11., costs, [.9, .7])[1], 'inspect_right')
        self.assertEqual(assess(probabilities, .8, 36., 11., costs, [.9, .7]),
                         ('PASS', None, 'insufficient_inspection_slack'))
        self.assertEqual(assess(probabilities, .8, 50., 11., costs, [.1, .1]),
                         ('PASS', None, 'no_useful_inspection_view'))


class BatchAcceptanceTest(unittest.TestCase):
    def test_safe_miss_requires_verified_exit_and_reset(self):
        from run_seeded_m5 import check_episode

        case = dict(index=7, scenario_selection_seed=1, fixture='green_partial', arrival='upstream',
                    scene='scene.xml', scene_sha256='scene', expected_action='PICK_PLACE',
                    expected_path='inspect', expected_view=None)
        frozen = dict(selector_sha256='selector', suite_sha256='suite', installed_manifest={})
        report = dict(checkpoint='M5F-seeded-episode', status='PASS', acceptance_passed=True,
                      selector_sha256='selector', suite_sha256='suite', scene='scene.xml', scene_sha256='scene',
                      installed_manifest={}, case_index=7, scenario_selection_seed=1,
                      fixture='green_partial', arrival='upstream', action='PASS', policy_path='unresolved',
                      final_reset_status='PASS', final_hold_status='PASS', final_owner='HOLD',
                      overlap_violations=0, physical_grasp='NOT_RUN', placement_status='NOT_RUN',
                      pass_physical=dict(status='PASS', exit_verified=True, no_arm_commands_after_decision=True))
        report.update({key: dict(status='PASS') for key in
                       ('runtime_configuration', 'planning_configuration', 'timing', 'image_evidence')})
        self.assertTrue(check_episode(report, case, frozen))
        report['pass_physical']['exit_verified'] = False
        with self.assertRaisesRegex(ValueError, 'supported exit'):
            check_episode(report, case, frozen)
