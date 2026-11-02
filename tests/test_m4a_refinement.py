"""Reproduce the live endpoint failure and exercise bounded, owned correction."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/sorting_supervisor/scripts'))
from m4a_runtime import InspectionExecutor, stamp, ns


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.c = yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4a.yaml').read_text())

    def executor(self):
        node = object.__new__(InspectionExecutor)
        node.c = self.c
        node.failure = None
        node.refinement_active = True
        node.refinement_goal = 'assess'
        node.refinement_epoch, node.refinement_deadline = 4, stamp(20_000_000_000)
        node.owner = dict(owner='SERVO', busy=False, epoch=4)
        node.now = lambda: 10_000_000_000
        node.hand_pose = lambda: (np.array([.54, -.01, .615]), np.array([.7071067811865476, .7071067811865476, 0., 0.]))
        node.get_clock = lambda: S(now=lambda: S(to_msg=lambda: stamp(node.now())))
        self.messages = []
        node.servo_publisher = S(publish=self.messages.append)
        node.events = []
        return node

    def test_twist_uses_current_epoch_deadline_frame_and_bounded_norm(self):
        node = self.executor(); node.send_refinement()
        self.assertEqual(len(self.messages), 1)
        m = self.messages[0]
        self.assertEqual((m.epoch, m.command_type, ns(m.deadline)), (4, 1, 20_000_000_000))
        self.assertEqual(m.twist.header.frame_id, 'world')
        self.assertEqual(ns(m.twist.header.stamp), node.now())
        v = m.twist.twist.linear
        self.assertLessEqual(np.linalg.norm([v.x, v.y, v.z]), .0040000001)
        self.assertGreater(v.x, 0.)

    def test_inactive_correction_sends_nothing(self):
        node = self.executor(); node.refinement_active = False; node.send_refinement()
        self.assertFalse(self.messages)

    def test_faulted_correction_sends_nothing(self):
        node = self.executor(); node.failure = 'feedback stopped'; node.send_refinement()
        self.assertFalse(self.messages)

    def test_changed_owner_stops_publication(self):
        node = self.executor(); node.owner['owner'] = 'HOLD'; node.send_refinement()
        self.assertFalse(self.messages); self.assertIn('ownership', node.failure)

    def test_changed_epoch_stops_publication(self):
        node = self.executor(); node.owner['epoch'] = 5; node.send_refinement()
        self.assertFalse(self.messages); self.assertFalse(node.refinement_active)

    def test_expired_lease_stops_publication(self):
        node = self.executor(); node.refinement_deadline = stamp(node.now()); node.send_refinement()
        self.assertFalse(self.messages); self.assertIn('lease expired', node.failure)

    def test_stale_hand_transform_stops_publication(self):
        node = self.executor()
        node.hand_pose = lambda: (_ for _ in ()).throw(ValueError('Hand transform is stale or future'))
        node.send_refinement()
        self.assertFalse(self.messages); self.assertIn('stale', node.failure)

    def test_live_failure_fits_correction_region_without_relaxing_final_limit(self):
        fixture = json.loads((ROOT/'tests/fixtures/m4a-planned-endpoint.json').read_text())
        self.assertEqual(self.c['maximum_position_error_m'], .006)
        self.assertGreater(fixture['position_error_m'], self.c['maximum_position_error_m'])
        self.assertLess(fixture['position_error_m'], self.c['refinement']['maximum_entry_position_error_m'])
        self.assertLess(fixture['orientation_error_rad'], self.c['refinement']['maximum_entry_orientation_error_rad'])

    def prepare_refine(self, sequence):
        node = self.executor(); node.refinement_active = False
        self.ticks = 0
        node.now = lambda: self.ticks
        self.calls, self.waits = [], []
        node.pose_error = lambda name: (sequence[0][1], sequence[0][2])
        def acquire(owner, **kwargs):
            self.calls.append((owner, kwargs))
            return 4, stamp(12_000_000_000)
        node.acquire = acquire
        node.wait_sim = self.waits.append
        def until(predicate, **kwargs):
            self.assertEqual(kwargs['sim_timeout'], self.c['refinement']['maximum_sim_s'])
            self.assertEqual(kwargs['timeout'], self.c['refinement']['maximum_wall_s'])
            for at, error, angle in sequence:
                self.ticks = round(at*1e9)
                node.pose_error = lambda name, e=error, a=angle: (e, a)
                if predicate():
                    return
            raise TimeoutError('correction timeout')
        node.until = until
        return node

    def test_correction_requires_continuous_settling_then_holds(self):
        node = self.prepare_refine([(0., .006768, .013613), (.1, .001, .005), (.2, .003, .005),
                                    (.3, .001, .005), (.55, .001, .005)])
        node.refine('assess')
        self.assertEqual(self.ticks, 550_000_000)
        self.assertEqual([c[0] for c in self.calls], ['SERVO', 'HOLD'])
        self.assertEqual(self.calls[0][1]['command_type'], 1)
        self.assertFalse(node.refinement_active)
        self.assertEqual(self.waits, [.5])

    def test_timeout_disables_correction(self):
        node = self.prepare_refine([(0., .006768, .013613), (1., .003, .005)])
        with self.assertRaises(TimeoutError):
            node.refine('assess')
        self.assertFalse(node.refinement_active)

    def test_large_entry_error_never_acquires_servo(self):
        node = self.prepare_refine([(0., .016, .01)])
        with self.assertRaisesRegex(ValueError, 'outside the correction region'):
            node.refine('assess')
        self.assertFalse(self.calls)

    def test_divergent_correction_stops(self):
        node = self.prepare_refine([(0., .006768, .013613), (.5, .016, .005)])
        with self.assertRaisesRegex(ValueError, 'left its bounded correction region'):
            node.refine('assess')
        self.assertFalse(node.refinement_active)
