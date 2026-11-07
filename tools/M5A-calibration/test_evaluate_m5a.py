from types import SimpleNamespace
import unittest

import numpy as np
from evaluate_m5a import decision_counts, fresh_groups


class CalibrationTests(unittest.TestCase):
    def test_groups_exclude_old_appearances_and_each_other(self):
        original = [{'group_id': 0, 'appearance_instance': 84019}]
        model = SimpleNamespace(meta={'dataset_group_ids': [0]})
        config = {'dataset_seed': 84019}
        splits = fresh_groups(original, model, config)
        self.assertEqual(len(splits['calibration']), 32)
        self.assertEqual(len(splits['test']), 32)
        self.assertEqual(len({row['appearance_instance'] for rows in splits.values() for row in rows}), 64)
        original[0]['appearance_instance'] = splits['test'][0]['appearance_instance']
        with self.assertRaisesRegex(ValueError, 'Appearance seeds overlap'):
            fresh_groups(original, model, config)

    def test_covered_false_accept_and_empty_denominator(self):
        probabilities = np.array([[.95, .05], [.55, .45], [.85, .15]])
        counts = decision_counts(probabilities, np.array([0, 0, 1]),
                                 np.array(['clear', 'partial', 'covered']), .8)
        self.assertEqual(counts['all']['correct_accepts'], 1)
        self.assertEqual(counts['covered']['wrong_accepts'], 1)
        self.assertEqual(counts['partial']['abstained'], 1)
        self.assertIsNone(counts['partial']['accepted_error_rate'])
