from pathlib import Path
import copy
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/sorting_perception/scripts"))
from camera_core import CaptureGuard,detect_marker,image_rgb,intrinsics,localize,ray_plane,quaternion_rotation

C=yaml.safe_load((Path(__file__).resolve().parents[1]/"src/sorting_perception/config/m3a.yaml").read_text())
K=np.array([[500.,0,320.],[0,500.,240.],[0,0,1.]])
DOWN=np.diag([1.,-1.,-1.])
def image():
    rgb=np.zeros((480,640,3),np.uint8);rgb[220:260,313:327]=[220,10,220];return rgb

class GeometryTests(unittest.TestCase):
    def test_rgb_marker_centroid_uses_calibrated_pixel_centres(self):
        found=detect_marker(image(),C);self.assertEqual(found["pixel_u"],320.);self.assertEqual(found["pixel_v"],240.)
    def test_nominal_ray_hits_known_handle_reference(self):
        np.testing.assert_allclose(ray_plane([320,240],K,[.65,0,1.35],DOWN,C),[.65,0,C["handle_plane_z_m"]],atol=1e-12)
    def test_camera_motion_requires_transform_at_capture_time(self):
        before=ray_plane([320,240],K,[.60,0,1.35],DOWN,C)
        after=ray_plane([320,240],K,[.65,0,1.35],DOWN,C)
        self.assertAlmostEqual(after[0]-before[0],.05)
    def test_wrong_plane_changes_off_axis_localization(self):
        changed=dict(C,marker_plane_z_m=C["marker_plane_z_m"]+.02)
        before=ray_plane([350,240],K,[.65,0,1.35],DOWN,C)
        after=ray_plane([350,240],K,[.65,0,1.35],DOWN,changed)
        self.assertAlmostEqual(before[0]-after[0],.0012)
    def test_parallel_or_behind_camera_rays_are_rejected(self):
        side=np.array([[0,0,1],[1,0,0],[0,1,0]])
        with self.assertRaisesRegex(ValueError,"ray_near_parallel"):ray_plane([320,240],K,[.65,0,1.35],side,C)
        with self.assertRaisesRegex(ValueError,"plane_outside_camera_depth"):ray_plane([320,240],K,[.65,0,1.35],np.eye(3),C)
    def test_non_rotation_and_nonfinite_pose_are_rejected(self):
        for p,r in (([.65,0,float("nan")],DOWN),([.65,0,1.35],np.zeros((3,3)))):
            with self.assertRaises(ValueError):ray_plane([320,240],K,p,r,C)
    def test_unseen_ambiguous_and_border_markers_are_invalid(self):
        blank=np.zeros((480,640,3),np.uint8)
        with self.assertRaisesRegex(ValueError,"marker_not_found"):detect_marker(blank,C)
        two=image();two[50:90,50:64]=[220,10,220]
        with self.assertRaisesRegex(ValueError,"ambiguous_markers"):detect_marker(two,C)
        border=blank.copy();border[50:90,:14]=[220,10,220]
        with self.assertRaisesRegex(ValueError,"marker_at_image_border"):detect_marker(border,C)
    def test_covariance_is_finite_positive_and_scales_with_pixel_noise(self):
        first=localize(image(),K,[.65,0,1.35],DOWN,C)
        second=localize(image(),K,[.65,0,1.35],DOWN,dict(C,pixel_sigma_px=1.))
        a=np.array(first["planar_covariance"]).reshape(2,2);b=np.array(second["planar_covariance"]).reshape(2,2)
        self.assertGreater(np.linalg.eigvalsh(a).min(),0);np.testing.assert_allclose(b,4*a)
    def test_padded_bgr_image_is_read_without_row_shift(self):
        msg=SimpleNamespace(encoding="bgr8",width=2,height=2,step=8,
                            data=bytes([3,2,1,6,5,4,99,99,9,8,7,12,11,10,99,99]))
        np.testing.assert_array_equal(image_rgb(msg),[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    def test_invalid_encoding_and_buffer_are_rejected(self):
        for msg in (SimpleNamespace(encoding="mono8"),SimpleNamespace(encoding="rgb8",width=2,height=2,step=6,data=bytes(4))):
            with self.assertRaises(ValueError):image_rgb(msg)
    def test_quaternion_sign_does_not_change_optical_rotation(self):
        np.testing.assert_allclose(quaternion_rotation([1,0,0,0]),DOWN)
        np.testing.assert_allclose(quaternion_rotation([-2,0,0,0]),DOWN)
        with self.assertRaises(ValueError):quaternion_rotation([0,0,0,0])
    def test_camera_info_shape_frame_and_distortion_must_match(self):
        info=SimpleNamespace(width=640,height=480,header=SimpleNamespace(frame_id="camera"),distortion_model="plumb_bob",d=[0.]*5,k=K.flatten())
        np.testing.assert_allclose(intrinsics(info,(480,640,3),"camera"),K)
        for field,value in (("width",320),("distortion_model","equidistant"),("d",[.1,0,0,0,0])):
            changed=copy.deepcopy(info);setattr(changed,field,value)
            with self.assertRaises(ValueError):intrinsics(changed,(480,640,3),"camera")

class CaptureTests(unittest.TestCase):
    def test_epoch_is_required_before_using_any_image(self):
        with self.assertRaisesRegex(ValueError,"episode_not_received"):CaptureGuard().check("wrist",10,10,C)
    def test_reset_clears_ordering_but_rejects_queued_old_images(self):
        g=CaptureGuard();g.reset(0,0);g.commit("overhead",1_000_000_000);g.reset(1,2_000_000_000)
        self.assertEqual(g.last_stamp,{})
        with self.assertRaisesRegex(ValueError,"capture_before_reset"):g.check("overhead",1_999_000_000,2_010_000_000,C)
        g.check("overhead",2_005_000_000,2_010_000_000,C)
    def test_duplicate_backward_old_and_future_capture_times_are_rejected(self):
        g=CaptureGuard();g.reset(0,0);g.commit("wrist",1_000_000_000)
        for stamp,now in ((1_000_000_000,1_010_000_000),(999_000_000,1_010_000_000),
                          (1_001_000_000,1_200_000_000),(1_100_000_000,1_001_000_000)):
            with self.assertRaises(ValueError):g.check("wrist",stamp,now,C)
    def test_each_camera_has_independent_capture_order(self):
        g=CaptureGuard();g.reset(0,0);g.commit("wrist",1_000_000_000)
        g.check("overhead",999_000_000,1_010_000_000,C)
    def test_old_epoch_message_cannot_clear_new_episode(self):
        g=CaptureGuard();g.reset(2,2_000_000_000);g.commit("wrist",2_050_000_000)
        self.assertFalse(g.reset(1,1_000_000_000));self.assertEqual(g.episode,2)
        g.commit("wrist",1_000_000_000);self.assertEqual(g.last_stamp["wrist"],2_050_000_000)

if __name__=="__main__":unittest.main()
