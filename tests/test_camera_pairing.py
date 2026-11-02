from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from sorting_interfaces.msg import HandleObservation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/sorting_perception/scripts"))
from m3a_probe import CameraProbe

def observation(seconds=1):
    m=HandleObservation(valid=True,episode_id=0,view_id="overhead")
    m.header.stamp.sec=seconds;return m

class StartupTests(unittest.TestCase):
    def probe(self,phase="stationary",floor=1_000_000_000):
        return SimpleNamespace(phase=phase,phase_begin_ns=floor,pending={},rejections=[],evaluation_enabled=True,epoch=None)
    def test_late_valid_startup_packet_does_not_create_a_fatal_pair_wait(self):
        p=self.probe("startup",0);CameraProbe.receive_observation(p,observation())
        self.assertEqual(p.pending,{})
        self.assertEqual(p.rejections[0]["reason"],"evaluator_startup_or_previous_phase_capture")
        self.assertTrue(p.rejections[0]["localizer_valid"])
    def test_capture_before_phase_barrier_is_retained_as_unscored_evidence(self):
        p=self.probe();CameraProbe.receive_observation(p,observation())
        self.assertEqual(p.pending,{});self.assertEqual(len(p.rejections),1)
    def test_new_capture_still_requires_the_exact_epoch_view_and_stamp(self):
        p=self.probe();m=observation(2)
        with patch("m3a_probe.time.monotonic",return_value=10.):CameraProbe.receive_observation(p,m)
        key=(0,"overhead",2_000_000_000)
        self.assertEqual(p.pending[key],(m,10.,"stationary"))
    def test_localizer_rejections_are_not_hidden_by_startup_handling(self):
        p=self.probe("startup",0);m=observation();m.valid=False;m.reason="capture_age_outside_bound"
        CameraProbe.receive_observation(p,m)
        self.assertEqual(p.rejections[0]["reason"],"capture_age_outside_bound")

class DiagnosticTests(unittest.TestCase):
    def probe(self,missing):
        m=observation(2);key=(0,"overhead",2_000_000_000)
        truth={} if missing=="truth" else {key:SimpleNamespace()}
        info={} if missing=="camera_info" else {("overhead",2_000_000_000):SimpleNamespace()}
        p=SimpleNamespace(pending={key:(m,0.,"stationary")},truth=truth,info=info,
                          pairing_diagnostics=[],reported_pair_waits=set(),evaluation_enabled=True,rejections=[],samples=[])
        p.pairing_detail=lambda *args:CameraProbe.pairing_detail(p,*args)
        return p
    def test_current_phase_missing_truth_remains_a_failure_with_specific_details(self):
        p=self.probe("truth")
        with patch("m3a_probe.time.monotonic",return_value=2.1):
            with self.assertRaisesRegex(RuntimeError,"missing=truth"):CameraProbe.evaluate(p)
        self.assertEqual(p.pairing_diagnostics[-1]["missing"],["truth"])
        self.assertIsNone(p.pairing_diagnostics[-1]["nearest_truth_delta_ns"])
    def test_missing_evaluator_metadata_is_recorded_and_cannot_count_as_success(self):
        p=self.probe("camera_info")
        with patch("m3a_probe.time.monotonic",return_value=2.1):
            CameraProbe.evaluate(p)
        self.assertIsNone(p.pairing_diagnostics[-1]["nearest_camera_info_delta_ns"])
        self.assertEqual(p.pairing_diagnostics[-1]["status"],"SKIPPED_METADATA")
        self.assertEqual(p.pending,{});self.assertEqual(p.samples,[])
        self.assertFalse(p.rejections[-1]["counted"])
    def test_finite_wait_can_recover_but_does_not_invent_a_pair(self):
        p=self.probe("truth")
        with patch("m3a_probe.time.monotonic",return_value=.5):CameraProbe.evaluate(p)
        self.assertEqual(len(p.pending),1);self.assertEqual(len(p.pairing_diagnostics),1)
        with patch("m3a_probe.time.monotonic",return_value=.6):CameraProbe.evaluate(p)
        self.assertEqual(len(p.pairing_diagnostics),1)

if __name__=="__main__":unittest.main()
