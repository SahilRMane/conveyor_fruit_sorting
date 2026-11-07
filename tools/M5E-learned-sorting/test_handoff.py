from types import SimpleNamespace
import unittest

from run_m5e import RouteLease, check_handoff, forecast, moving_config, proposal_from_capture


class HandoffTests(unittest.TestCase):
    def test_both_views_pass_the_installed_handoff_checks(self):
        for view, probabilities, bin_id in [('inspect_left', [.99, .01], 'bin_a'),
                                             ('inspect_right', [.01, .99], 'bin_b')]:
            with self.subTest(view=view):
                mc = moving_config()
                mc['inspection_view'] = view
                mc['classification_x_ranges'][view] = [.52, .57]
                control = SimpleNamespace(mc=mc, episode=0, bound_track=1, model=SimpleNamespace(sha='model'))
                sample = dict(capture_ns=1_000_000_000, class_evidence=dict(probabilities=probabilities))
                decision = proposal_from_capture(sample, view, control, 1_000_000_000, 45., 25.)
                target = dict(episode_id=0, track_id=1, frame_id='world', view_id='overhead',
                              valid=True, camera_only=True, capture_ns=1_000_000_000,
                              state=[.55, 0., .001, 0.], covariance=[0.] * 16)
                wrist = dict(episode_id=0, capture_ns=1_000_000_000, frame_id='world', view_id='wrist',
                             valid=True, position=[.55, 0., .4229])
                owner = dict(owner='HOLD', ready=True, busy=False, stats=dict(overlap_violations=0))
                check = check_handoff(decision, target, wrist, owner, dict(fresh=True, fingers_open=True),
                                      now_ns=1_000_000_000, episode=0, config=mc,
                                      filter_config={'acceleration_sigma_m_s2': .001}, model_sha='model',
                                      forecast=forecast, remaining=lambda *_: {'remaining_s': 45.})
                self.assertEqual(check['status'], 'PASS')
                lease = RouteLease(decision, 'model', 30.)
                self.assertEqual(lease.check(target, 1_000_000_000, mc)['bin_id'], bin_id)
