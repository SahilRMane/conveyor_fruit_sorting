from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import yaml
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo
from sorting_interfaces.msg import CameraEpoch,CameraTruth,HandleObservation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/sorting_perception/scripts"))
from m3a_probe import CameraProbe

ROOT=Path(__file__).resolve().parents[1]
C=yaml.safe_load((ROOT/"src/sorting_perception/config/m3a.yaml").read_text())

def packet(episode=1,seconds=33,view="overhead",error=0.):
    m=HandleObservation(valid=True,episode_id=episode,view_id=view,camera_frame=view+"_camera_optical_frame",
                        planar_covariance=[1e-6,0.,0.,1e-6])
    m.header.stamp.sec=seconds;m.position.x=.55+error;m.position.z=.4229;return m

def probe():
    p=SimpleNamespace(c=C,phase="moving_belt",phase_begin_ns=20_000_000_000,now_ns=lambda:30_000_000_000,
        evaluation_enabled=True,guarded=True,previous_q=[0.]*7,q={"q":1.},v={"q":0.},pending={},truth={},info={},
        reference={},rejections=[],samples=[],pairing_diagnostics=[],reported_pair_waits=set(),failed_camera_sample=None,
        epoch=CameraEpoch(episode_id=0),camera_status={"episode_id":0},reset_guard=object(),reset_in_progress=False,
        reset_report=None,reset_stage="not_started",belt_speed=0.,contact={"external_force_x_N":0.},
        owner={"owner":"HOLD","ready":True,"busy":False},buffer=Mock())
    p.begin=lambda phase:CameraProbe.begin(p,phase)
    p.prepare_for_reset=lambda:CameraProbe.prepare_for_reset(p)
    p.send_belt=Mock();p.safe_hold=Mock()
    return p

def ready_pair(p,error=0.,view="overhead"):
    p.epoch=CameraEpoch(episode_id=1);p.begin("post_reset")
    m=packet(error=error,view=view);key=(1,view,33_000_000_000)
    t=CameraTruth(episode_id=1,view_id=view);t.handle_position.x=.55;t.handle_position.z=.4229
    t.camera_pose.position.x=.65;t.camera_pose.position.z=1.15;t.camera_pose.orientation.x=1.;t.camera_pose.orientation.w=0.
    info=CameraInfo();info.header.frame_id=m.camera_frame
    transform=TransformStamped();transform.transform.translation.x=.65;transform.transform.translation.z=1.15
    transform.transform.rotation.x=1.;transform.transform.rotation.w=0.
    p.pending[key]=(m,1.9,"post_reset");p.truth[key]=t;p.info[(view,33_000_000_000)]=info
    p.buffer.lookup_transform.return_value=transform
    return key

class ResetFenceTests(unittest.TestCase):
    def test_prepare_reset_disables_scoring_and_flushes_previous_epoch_buffers(self):
        p=probe();p.pending["old"]=object();p.truth["old"]=object();p.info["old"]=object()
        p.prepare_for_reset()
        self.assertEqual(p.phase,"reset");self.assertFalse(p.evaluation_enabled);self.assertFalse(p.guarded)
        self.assertEqual((p.pending,p.truth,p.info,p.q,p.v),({},{},{},{},{}))
    def test_reset_captures_do_not_satisfy_camera_sample_counts(self):
        p=probe();p.prepare_for_reset();CameraProbe.receive_observation(p,packet())
        self.assertEqual(p.pending,{});self.assertEqual(p.rejections[-1]["reason"],"evaluator_reset_or_failure_paused")
        CameraProbe.evaluate(p);p.buffer.lookup_transform.assert_not_called()
    def test_truth_and_info_received_during_reset_are_discarded(self):
        p=probe();p.prepare_for_reset()
        CameraProbe.receive_truth(p,CameraTruth());CameraProbe.receive_info(p,"wrist",CameraInfo())
        self.assertEqual(p.truth,{});self.assertEqual(p.info,{})
    def test_post_reset_requires_current_episode_and_capture_after_settling_barrier(self):
        p=probe();p.epoch=CameraEpoch(episode_id=1);p.begin("post_reset")
        CameraProbe.receive_observation(p,packet(episode=0))
        self.assertEqual(p.rejections[-1]["reason"],"evaluator_episode_mismatch")
        CameraProbe.receive_observation(p,packet(seconds=29))
        self.assertEqual(p.pending,{})
        CameraProbe.receive_observation(p,packet())
        self.assertIn((1,"overhead",33_000_000_000),p.pending)
    def test_physical_reset_and_guard_revalidation_complete_before_scoring_resumes(self):
        p=probe();calls=[]
        def call(client,request):
            self.assertFalse(p.evaluation_enabled);calls.append(request.data)
            p.owner={"owner":"RESETTING" if request.data else "HOLD","ready":not request.data,"busy":request.data}
            return SimpleNamespace(success=True,message="")
        def reset():
            self.assertFalse(p.evaluation_enabled);self.assertTrue(p.reset_in_progress)
            p.prepare_for_reset();p.epoch=CameraEpoch(episode_id=1);p.camera_status={"episode_id":1}
            return {"status":"PASS","reset_seq":1}
        p.call=call;p.until=lambda predicate,*args:self.assertTrue(predicate())
        p.reset_manager=SimpleNamespace(reset_home=reset)
        self.assertEqual(CameraProbe.reset_episode(p)["status"],"PASS")
        self.assertEqual(calls,[True,False]);self.assertTrue(p.evaluation_enabled)
        self.assertEqual(p.phase,"post_reset");self.assertFalse(p.reset_in_progress)
    def test_failed_physical_reset_keeps_guard_closed_and_scoring_off(self):
        p=probe();calls=[]
        def call(client,request):
            calls.append(request.data);p.owner={"owner":"RESETTING","ready":False,"busy":True}
            return SimpleNamespace(success=True,message="")
        p.call=call;p.until=lambda predicate,*args:self.assertTrue(predicate())
        p.reset_manager=SimpleNamespace(reset_home=lambda:{"status":"FAIL","error":"raw home state mismatch"})
        with self.assertRaisesRegex(RuntimeError,"raw home state mismatch"):CameraProbe.reset_episode(p)
        self.assertEqual(calls,[True]);self.assertTrue(p.reset_in_progress);self.assertFalse(p.evaluation_enabled)

class ValidationTests(unittest.TestCase):
    def test_fresh_settled_samples_are_checked_and_accepted(self):
        p=probe();ready_pair(p)
        CameraProbe.evaluate(p);self.assertEqual(len(p.samples),1);self.assertTrue(p.samples[0]["localization_scored"])
    def test_real_post_reset_error_still_fails_at_two_millimetres_for_both_views(self):
        for view in ("overhead","wrist"):
            with self.subTest(view=view):
                p=probe();ready_pair(p,error=.003,view=view)
                with self.assertRaisesRegex(RuntimeError,"Image localization exceeds 2 mm"):CameraProbe.evaluate(p)
                self.assertEqual(p.samples,[]);self.assertEqual(p.failed_camera_sample["phase"],"post_reset")
                self.assertAlmostEqual(p.failed_camera_sample["planar_error_m"],.003)
    def test_recovered_tf_wait_is_marked_resolved(self):
        p=probe();key=ready_pair(p);p.reported_pair_waits.add(key)
        p.pairing_diagnostics=[dict(episode_id=1,view_id="overhead",capture_ns=key[2],phase="post_reset",resolved=False)]
        with patch("m3a_probe.time.monotonic",return_value=2.1):CameraProbe.evaluate(p)
        self.assertTrue(p.pairing_diagnostics[0]["resolved"]);self.assertEqual(p.pairing_diagnostics[0]["status"],"RESOLVED")

class FailureCleanupTests(unittest.TestCase):
    def cleanup_probe(self):
        p=probe();p.pending["bad"]=object()
        def pause(client,request):
            self.assertTrue(request.paused);self.assertFalse(p.evaluation_enabled);self.assertEqual(p.pending,{})
            CameraProbe.evaluate(p)
            return SimpleNamespace(success=True,message="")
        p.reset_manager=SimpleNamespace(client=lambda kind:object(),call=pause)
        return p
    def test_cleanup_disables_callback_validation_before_spinning_for_hold(self):
        p=self.cleanup_probe()
        p.safe_hold=Mock(side_effect=lambda:self.assertFalse(p.evaluation_enabled))
        result=CameraProbe.stop_after_failure(p)
        p.safe_hold.assert_called_once();self.assertEqual(result["hold"],"PASS");self.assertTrue(result["paused"])
    def test_cleanup_does_not_request_hold_while_reset_guard_is_closed(self):
        p=self.cleanup_probe();p.reset_in_progress=True;p.owner["owner"]="RESETTING"
        result=CameraProbe.stop_after_failure(p)
        p.safe_hold.assert_not_called();self.assertEqual(result["hold"],"NOT_REQUESTED");self.assertTrue(result["paused"])

if __name__=="__main__":unittest.main()
