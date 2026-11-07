"""Image evidence must not encode or write files during active control."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import MethodType
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
import m4a_common
from m4d_runtime import IntegratedInspector
import test_m4a_capture as capture_harness


class CaptureEvidenceTests(unittest.TestCase):
    def inspector(self):
        node=IntegratedInspector.__new__(IntegratedInspector)
        node.deferred_images=[];node.capture_monitoring=True
        node.refinement_active=False;node.active_goal=None
        return node

    def test_active_capture_copies_pixels_without_encoding_or_writing(self):
        node=self.inspector();rgb=np.full((8,8,3),17,np.uint8)
        with patch('m4d_runtime.cv2.imwrite') as write,patch('m4d_runtime.cv2.cvtColor') as convert:
            node.save_capture_image(Path('/tmp/evidence.png'),rgb)
            write.assert_not_called();convert.assert_not_called()
        rgb[:]=0
        self.assertTrue(np.all(node.deferred_images[0][1]==17))

    def test_full_capture_uses_the_deferred_hook(self):
        harness=capture_harness.CapturePairingTests()
        node=harness.setup_executor([('relative',True,-.02)])
        node.deferred_images=[]
        node.save_capture_image=MethodType(IntegratedInspector.save_capture_image,node)
        with patch('m4a_runtime.image_rgb',return_value=np.zeros((20,20,3),np.uint8)), \
             patch('m4a_runtime.intrinsics',return_value=np.eye(3)), \
             patch('m4a_runtime.localize',return_value=dict(position=[.55,0.,.4229])), \
             patch('m4a_runtime.fruit_crop',return_value=(np.zeros((8,8,3),np.uint8),[0,0,8,8])), \
             patch('m4a_runtime.cv2.imwrite') as write:
            samples=node.capture('assess',Path('/tmp'))
            write.assert_not_called()
        self.assertEqual(len(samples),1);self.assertEqual(len(node.deferred_images),1)

    def test_evidence_memory_is_bounded_to_two_views(self):
        node=self.inspector();rgb=np.zeros((8,8,3),np.uint8)
        node.save_capture_image(Path('/tmp/a.png'),rgb);node.save_capture_image(Path('/tmp/b.png'),rgb)
        with self.assertRaisesRegex(ValueError,'exceeded two'):
            node.save_capture_image(Path('/tmp/c.png'),rgb)
        self.assertEqual(len(node.deferred_images),2)

    def test_flush_cannot_run_during_camera_control_or_motion(self):
        node=self.inspector()
        for field,value in (('capture_monitoring',True),('refinement_active',True),('active_goal',object())):
            node.capture_monitoring=False;node.refinement_active=False;node.active_goal=None
            setattr(node,field,value)
            with self.subTest(field=field),patch('m4d_runtime.cv2.imwrite') as write:
                with self.assertRaisesRegex(ValueError,'Stop camera control'):node.flush_capture_images()
                write.assert_not_called()

    def test_flush_preserves_original_pixels_and_reports_write_time(self):
        node=self.inspector();rgb=np.array([[[1,2,3],[4,5,6]]],dtype=np.uint8)
        node.save_capture_image(Path('/tmp/evidence.png'),rgb);node.capture_monitoring=False
        with patch('m4d_runtime.cv2.imwrite',return_value=True) as write:
            result=node.flush_capture_images()
            np.testing.assert_array_equal(write.call_args.args[1],rgb[:,:,::-1])
        self.assertEqual(result['status'],'PASS');self.assertEqual(len(result['images']),1)
        self.assertGreaterEqual(result['images'][0]['write_wall_s'],0.)
        self.assertEqual(node.deferred_images,[])

    def test_failed_write_retains_the_unsaved_image(self):
        node=self.inspector();node.save_capture_image(Path('/tmp/evidence.png'),np.zeros((8,8,3),np.uint8))
        node.capture_monitoring=False
        with patch('m4d_runtime.cv2.imwrite',return_value=False):
            with self.assertRaisesRegex(ValueError,'image write failed'):node.flush_capture_images()
        self.assertEqual(len(node.deferred_images),1)


if __name__=='__main__':unittest.main()
