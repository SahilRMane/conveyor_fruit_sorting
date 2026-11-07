"""Exercise live capture pairing without a ROS node, including delayed metadata."""
from pathlib import Path
import sys
from types import SimpleNamespace as S
from unittest.mock import patch
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/sorting_supervisor/scripts'))
from m4a_runtime import InspectionExecutor
from camera_core import require


class CapturePairingTests(unittest.TestCase):
    def setup_executor(self, sequence):
        self.sequence = sequence
        self.now_ns = 10_000_000_000
        node = object.__new__(InspectionExecutor)
        node.c = dict(maximum_capture_age_s=.15, maximum_metadata_wait_wall_s=.5,
                      maximum_pose_wait_wall_s=8., samples_per_pose=1)
        node.episode, node.epoch_barrier = 3, 0
        node.pending, node.info = None, {}
        node.lc, node.cc, node.events = {}, {}, []
        from collections import Counter
        node.rejections = Counter()
        node.now = lambda: self.now_ns
        tf = S(transform=S(translation=S(x=0., y=0., z=1.), rotation=S(x=0., y=0., z=0., w=1.)))
        node.buffer = S(lookup_transform=lambda *args: tf)
        node.model = S(infer=lambda crop: dict(accepted=False, confidence=.5))

        def until(predicate, **kwargs):
            for capture, metadata, age in self.sequence:
                self.now_ns += 50_000_000
                capture = self.now_ns+int(age*1e9) if capture == 'relative' else capture
                stamp = S(sec=capture//1_000_000_000, nanosec=capture%1_000_000_000)
                from builtin_interfaces.msg import Time
                image = S(header=S(stamp=Time(sec=stamp.sec, nanosec=stamp.nanosec), frame_id='wrist_camera_optical_frame'))
                node.pending = (image, 0.)
                if metadata:
                    node.info[capture] = S()
                if predicate():
                    return
            raise TimeoutError('No admissible capture')
        node.until = until
        return node

    def run_capture(self, sequence):
        node = self.setup_executor(sequence)
        with patch('m4a_runtime.image_rgb', return_value=np.zeros((20, 20, 3), np.uint8)), \
             patch('m4a_runtime.intrinsics', return_value=np.eye(3)), \
             patch('m4a_runtime.localize', return_value=dict(position=[.55, 0., .4229])), \
             patch('m4a_runtime.fruit_crop', return_value=(np.zeros((8, 8, 3), np.uint8), [0, 0, 8, 8])), \
             patch('m4a_runtime.cv2.imwrite', return_value=True):
            return node.capture('assess', Path('/tmp')), node

    def test_matching_capture_admitted(self):
        samples, node = self.run_capture([('relative', True, -.02)])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]['episode_id'], 3)
        work=next(event for event in node.events if event['state']=='CAPTURE_WORK')
        self.assertEqual(work['capture_ns'],samples[0]['capture_ns'])
        self.assertGreaterEqual(work['processing_wall_s'],0.)
        self.assertGreaterEqual(work['evidence_wall_s'],0.)

    def test_pre_motion_capture_never_counted(self):
        samples, node = self.run_capture([(9_999_999_999, True, 0.), ('relative', True, -.02)])
        self.assertGreater(samples[0]['capture_ns'], 10_000_000_000)
        self.assertEqual(node.rejections['previous_state_or_stale_capture'], 1)

    def test_future_capture_never_counted(self):
        samples, node = self.run_capture([('relative', True, .001), ('relative', True, -.02)])
        self.assertEqual(node.rejections['previous_state_or_stale_capture'], 1)

    def test_stale_capture_never_counted(self):
        samples, node = self.run_capture([('relative', True, -.2), ('relative', True, -.02)])
        self.assertEqual(node.rejections['previous_state_or_stale_capture'], 1)

    def test_missing_metadata_never_borrowed_from_other_capture(self):
        samples, node = self.run_capture([('relative', False, -.02), ('relative', True, -.02)])
        self.assertEqual(node.rejections['matching_camera_info_pending'], 1)
        self.assertEqual(len(samples), 1)

    def test_no_metadata_is_bounded_failure(self):
        with self.assertRaises(TimeoutError):
            self.run_capture([('relative', False, -.02)])

    def stop_executor(self, owner):
        node = object.__new__(InspectionExecutor)
        node.active_goal = None
        node.owner = dict(owner=owner, busy=False, epoch=1)
        self.commands = []
        node.belt = S(publish=lambda m: self.commands.append(m.data))
        node.until = lambda predicate, **kwargs: self.assertTrue(predicate())
        node.future = lambda response, **kwargs: response
        return node

    def test_cleanup_keeps_existing_hold(self):
        node = self.stop_executor('HOLD')
        node.stop()
        self.assertEqual(self.commands, [0.0])

    def test_cleanup_retries_a_watchdog_handoff_race(self):
        node = self.stop_executor('PLANNED')
        calls = []
        def request(req):
            calls.append(req.owner)
            if len(calls) == 1:
                return S(success=False, reason='An owner transition is already running')
            node.owner.update(owner='HOLD', epoch=2)
            return S(success=True, epoch=2, reason='held')
        node.owner_client = S(call_async=request)
        node.stop()
        self.assertEqual(calls, ['HOLD', 'HOLD'])
        self.assertEqual(self.commands, [0.0])

    def test_cleanup_retains_fault(self):
        node = self.stop_executor('FAULT')
        with self.assertRaisesRegex(RuntimeError, 'FAULT retained'):
            node.stop()


if __name__ == '__main__':
    unittest.main()
