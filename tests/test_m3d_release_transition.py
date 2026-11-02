"""Release impact, action timing and native between-sample force guards."""
from pathlib import Path
import copy,json,sys,unittest
from unittest.mock import Mock
import numpy as np,yaml
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/sorting_control/scripts'),str(R/'src/sorting_perception/scripts')]
from native_sort import check_floor_force
from native_camera_pickup import NativePlant
from camera_sort import SortController
from action_msgs.msg import GoalStatus
from types import SimpleNamespace as NS

class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.c=yaml.safe_load((R/'src/sorting_control/config/m3d.yaml').read_text())
        self.record=json.loads((R/'tests/fixtures/m3d-release-impact.json').read_text())
    def test_live_peak_is_rejected_at_original_twice_weight_limit(self):
        row=self.record['peak_state'];self.assertAlmostEqual(self.record['floor_force_limit_N'],10.2024)
        self.assertGreater(row['bin_floor_force_N'][0],10.2024)
        with self.assertRaisesRegex(AssertionError,'impact exceeds'):check_floor_force(row,row['bin_floor_force_N'],0)
    def test_impact_between_regular_samples_cannot_be_hidden(self):
        with self.assertRaisesRegex(AssertionError,'impact exceeds'):
            check_floor_force({'bin_floor_force_N':[5.1012,0.]},[10.2618728863,0.],0)
    def test_selected_floor_support_below_limit_is_accepted(self):
        for index in (0,1):
            forces=[0.,0.];forces[index]=5.1012;peaks=forces.copy();peaks[index]=8.
            check_floor_force({'bin_floor_force_N':forces},peaks,index)
    def test_exact_limit_is_rejected(self):
        with self.assertRaises(AssertionError):check_floor_force({'bin_floor_force_N':[10.2024,0.]},[0.,0.],0)
    def test_unselected_floor_peak_is_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Unselected'):
            check_floor_force({'bin_floor_force_N':[5.1012,0.]},[5.2,.1],0)
    def test_invalid_native_force_values_are_rejected(self):
        for peaks in ([float('nan'),0.],[-1.,0.],[1.],[float('inf'),0.]):
            with self.subTest(peaks=peaks):
                with self.assertRaisesRegex(AssertionError,'Invalid native'):
                    check_floor_force({'bin_floor_force_N':[0.,0.]},peaks,0)
    def test_native_peak_read_uses_two_element_output(self):
        plant=NativePlant.__new__(NativePlant)
        def call(name,pointer):
            self.assertEqual(name,'nc_floor_peaks');pointer[0]=8.;pointer[1]=0.
        plant.call=call;np.testing.assert_array_equal(plant.floor_peaks(),[8.,0.])
    def test_cpp_peak_observer_retains_existing_per_step_guards(self):
        text=(R/'src/sorting_simulation/src/native_camera_plant.cpp').read_text()
        self.assertEqual(text.count('sim->guard='),1)
        for fragment in ('state.forbidden==0','finger force bound','measured joint speed bound','state.bin_floor_N[i]','p.floor_peaks={0,0}'):
            self.assertIn(fragment,text)
    def test_native_probe_covers_both_bins_height_bias_and_action_wait(self):
        text=(R/'src/sorting_perception/scripts/floor_release_probe.py').read_text()
        for fragment in ("'bin_a'","'bin_b'",'(-.003,0.,.0022,.003)','(0.,1.05,1.75)','bin_floor_step_peak_N'):
            self.assertIn(fragment,text)
    def test_full_open_follows_measured_support_and_supported_unload(self):
        text=(R/'src/sorting_control/scripts/camera_sort.py').read_text()
        self.assertIn("self.lower_to_wrist_support(orientation)",text)
        self.assertIn("['supported_unload_positions_m']",text)
        self.assertIn("['maximum_open_payload_fraction']",text)
        self.assertNotIn("for position in self.s['release_unload_positions_m']",text)
        self.assertEqual(self.c['supported_release']['supported_unload_positions_m'],[.0045,.0080,.0088,.0094])
        self.assertLessEqual(self.c['supported_release']['support_fraction'],.25)
        self.assertLessEqual(self.c['supported_release']['maximum_open_payload_fraction'],.15)
    def test_release_command_start_is_recorded_even_when_action_wait_fails(self):
        c=SortController.__new__(SortController);c.phase='release';c.input_enabled=True
        c.owner={'owner':'SERVO','busy':False};c.now_ns=lambda:1_000_000_000;c.input_deadline=20_000_000_000
        c.result={};c.hand=Mock();c.hand.send_goal_async.return_value=NS(accepted=True,get_result_async=lambda:None)
        def future(value,*args):
            if args:raise RuntimeError('operation aborted')
            return value
        c.future=future
        with self.assertRaisesRegex(RuntimeError,'operation aborted'):c.release_hand_step(.00492,unload=True)
        self.assertEqual(c.result['release_command_starts'][-1]['position_command_m'],.00492)
        self.assertNotIn('release_commands',c.result)
    def test_runtime_controller_does_not_use_native_force_observer(self):
        text=(R/'src/sorting_control/scripts/camera_sort.py').read_text()
        self.assertNotIn('nc_floor_peaks',text);self.assertNotIn('bin_floor_force_N',text)

if __name__=='__main__':unittest.main()
