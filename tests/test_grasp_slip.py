from pathlib import Path
import copy
import json
import math
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/sorting_control/scripts"))
from grasp_core import grasp_slip_metrics, hand_relative_position, require_grasp_slip
from m2c_runtime import MovingGrasp

ROOT=Path(__file__).resolve().parents[1]
C=yaml.safe_load((ROOT/"src/sorting_control/config/m2c.yaml").read_text())
FIXTURE=json.loads((ROOT/"tests/fixtures/live-hold-failure.json").read_text())


class SlipFrameTests(unittest.TestCase):
    def test_shared_translation_and_rotation_do_not_count_as_slip(self):
        baseline=dict(hand_xyz=[1.,2.,3.],tray_xyz=[1.,2.,2.78],hand_quat_wxyz=[1.,0.,0.,0.])
        angle=0.02
        state=dict(hand_xyz=[4.,5.,6.],
                   tray_xyz=[4.-0.22*math.sin(angle),5.,6.-0.22*math.cos(angle)],
                   hand_quat_wxyz=[math.cos(angle/2),0.,math.sin(angle/2),0.])
        metrics=grasp_slip_metrics(state,baseline)
        self.assertLess(metrics["relative_slip_m"],1e-12)
        self.assertGreater(metrics["world_offset_change_m"],0.003)
        require_grasp_slip(metrics,C["maximum_slip_m"])

    def test_real_relative_translation_still_fails_at_3mm(self):
        baseline=FIXTURE["grasp_state"]
        for distance,accepted in ((0.0029,True),(0.0031,False)):
            state=copy.deepcopy(baseline);state["tray_xyz"][0]+=distance
            metrics=grasp_slip_metrics(state,baseline)
            self.assertAlmostEqual(metrics["relative_slip_m"],distance,places=12)
            if accepted:require_grasp_slip(metrics,C["maximum_slip_m"])
            else:
                with self.assertRaisesRegex(RuntimeError,"hand frame exceeds 3 mm"):
                    require_grasp_slip(metrics,C["maximum_slip_m"])

    def test_quaternion_sign_and_normalization_do_not_change_the_frame(self):
        baseline=FIXTURE["grasp_state"];state=copy.deepcopy(baseline)
        state["hand_quat_wxyz"]=[-2*x for x in state["hand_quat_wxyz"]]
        np.testing.assert_allclose(hand_relative_position(state),hand_relative_position(baseline),atol=1e-12)

    def test_invalid_pose_cannot_produce_a_valid_slip_measurement(self):
        for field,value in (("hand_quat_wxyz",[0,0,0,0]),("hand_quat_wxyz",[1,0,float("nan"),0]),
                            ("hand_xyz",[0,0,float("inf")]),("tray_xyz",[0,0])):
            state=copy.deepcopy(FIXTURE["grasp_state"]);state[field]=value
            with self.assertRaises(RuntimeError):hand_relative_position(state)

    def test_saved_failure_is_rotation_not_3mm_of_relative_translation(self):
        expected={"held":0.00045012384145337575,"failure":0.000466539848321498,"paused":0.0004419743839036222}
        for label,state in FIXTURE["samples"].items():
            metrics=grasp_slip_metrics(state,FIXTURE["grasp_state"])
            self.assertAlmostEqual(metrics["relative_slip_m"],expected[label],places=12)
            require_grasp_slip(metrics,C["maximum_slip_m"])
        self.assertGreater(grasp_slip_metrics(FIXTURE["samples"]["failure"],FIXTURE["grasp_state"])["world_offset_change_m"],0.003)


class RuntimeReplayTests(unittest.TestCase):
    def probe(self):
        return SimpleNamespace(c=C,phase="final_hold",grasp=FIXTURE["grasp_state"],maximum_slip=0.,
                               maximum_world_offset_change=0.,grip_trace=[])

    def test_real_runtime_checks_accept_all_saved_hold_states(self):
        probe=self.probe()
        for state in FIXTURE["samples"].values():MovingGrasp.check(probe,state)
        self.assertEqual(len(probe.grip_trace),3)
        self.assertLess(probe.maximum_slip,0.0005)
        self.assertGreater(probe.maximum_world_offset_change,0.003)

    def test_runtime_still_rejects_true_slip(self):
        state=copy.deepcopy(FIXTURE["grasp_state"]);state["tray_xyz"][0]+=0.0031
        with self.assertRaisesRegex(RuntimeError,"hand frame exceeds 3 mm"):
            MovingGrasp.check(self.probe(),state)

    def test_runtime_still_rejects_lost_finger_contact(self):
        state=copy.deepcopy(FIXTURE["samples"]["held"]);state["left_force_N"]=0.
        with self.assertRaisesRegex(RuntimeError,"contact was lost"):
            MovingGrasp.check(self.probe(),state)

    def test_runtime_still_rejects_tilt_above_three_degrees(self):
        state=copy.deepcopy(FIXTURE["samples"]["held"])
        state["tray_quat_wxyz"]=[math.cos(0.04),0.,math.sin(0.04),0.]
        with self.assertRaisesRegex(RuntimeError,"tilt exceeds 3 degrees"):
            MovingGrasp.check(self.probe(),state)

    def test_runtime_still_rejects_forbidden_contact(self):
        state=copy.deepcopy(FIXTURE["samples"]["held"]);state["forbidden_contacts"]=1
        with self.assertRaisesRegex(RuntimeError,"Forbidden physical contact"):
            MovingGrasp.check(self.probe(),state)


if __name__=="__main__":unittest.main()
