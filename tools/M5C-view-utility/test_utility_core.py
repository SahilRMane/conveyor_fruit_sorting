import unittest

import numpy as np
from utility_core import choose_view, fit, predict


class UtilityTests(unittest.TestCase):
    def test_deadline_excludes_higher_scoring_view(self):
        costs = {'inspect_left': 28., 'inspect_right': 25.}
        self.assertEqual(choose_view([.95, .80], costs, 40., 11.), 'inspect_left')
        self.assertEqual(choose_view([.95, .80], costs, 39., 11.), 'inspect_right')
        self.assertIsNone(choose_view([.95, .80], costs, 36., 11.))
        self.assertIsNone(choose_view([.1, .2], costs, 60., 11.))

    def test_model_learns_both_directions(self):
        features = np.array([[-1.], [1.]] * 16)
        labels = np.array([[1., 0.], [0., 1.]] * 16)
        scores = predict(fit(features, labels), features[:2])
        np.testing.assert_array_equal(scores.argmax(axis=1), [0, 1])
