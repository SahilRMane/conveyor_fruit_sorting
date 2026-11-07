import unittest

from probe_m5b import costs


class TimingTests(unittest.TestCase):
    def test_costs_include_gaps_and_return(self):
        timeline = [
            ('PLAN', 'assess', 1), ('SETTLED', 'assess', 3), ('CAPTURE_COMPLETE', 'assess', 4),
            ('PLAN', 'inspect_right', 5), ('SETTLED', 'inspect_right', 8),
            ('CAPTURE_COMPLETE', 'inspect_right', 9), ('PLAN', 'assess', 10), ('SETTLED', 'assess', 12),
        ]
        events = [dict(state=state, pose=pose, enter_ns=t * 10**9) for state, pose, t in timeline]
        self.assertEqual(costs(events, 'inspect_right'), dict(
            travel_and_settle_sim_s=3, capture_sim_s=1, return_and_settle_sim_s=2, total_sim_s=7))
        with self.assertRaisesRegex(ValueError, 'Unexpected wrist motion'):
            costs(events[:-1], 'inspect_right')
        events[-1]['enter_ns'] = 0
        with self.assertRaisesRegex(ValueError, 'out of order'):
            costs(events, 'inspect_right')
