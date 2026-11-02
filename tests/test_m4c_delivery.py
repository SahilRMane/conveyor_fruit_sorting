"""Replay the reported 150 ms startup failure through the actual wait loop."""
from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4c_runtime import MovingInspectionExecutor
from m4a_runtime import stamp
from m4c_checks import replay_capture_delivery
from sorting_interfaces.msg import TargetBelief
from ownership_core import NAMES


class ReplayMovingExecutor(MovingInspectionExecutor):
    @property
    def executor(self):return self.replay_executor


class M4CDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.wall=100.;self.sim=7_947_999_999;self.commands=[];self.queue=[]
        self.c=yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4c.yaml').read_text())
        n=object.__new__(ReplayMovingExecutor);self.n=n
        n.mc=n.policy_config=self.c;n.policy_started=None
        n.c=dict(maximum_joint_speed_rad_s=.25)
        n.filter_config=yaml.safe_load((ROOT/'src/sorting_perception/config/m3b_filter.yaml').read_text())
        n.episode=0;n.bound_track=None;n.overhead=None;n.overhead_wall=0.;n.overhead_frames=0
        n.overhead_diagnostic={};n.deadline_diagnostic={};n.history=deque(maxlen=1200)
        n.tracking_active=False;n.operation=None;n.events=[];n.belt_speed=0.
        n.timing=S(status='PASS',sample=lambda *a,**k:None,note=lambda *a,**k:None)
        n.failure=None;n.owner=dict(owner='HOLD',busy=False)
        n.q={name:0. for name in NAMES};n.q.update(panda_finger_joint1=.04,panda_finger_joint2=.04)
        n.v={name:0. for name in NAMES};n.q_received=self.wall;n.q_stamp=self.sim-2_000_000
        n.q_sequence=1;n.q_ignored=0;n.clock_last=0;n.feedback_armed=True;n.feedback_diagnostic={}
        n.now=lambda:self.sim;n.replay_executor=S(spin_once=self.step)
        n.belt=S(publish=lambda msg:self.commands.append((self.sim,msg.data)))
        self.patch=patch('m4c_runtime.time.monotonic',side_effect=lambda:self.wall)
        self.patch.start();self.addCleanup(self.patch.stop)
        # Exact cached age at TRANSPORT_STARTED in the user's failed run: 130 ms.
        self.deliver(7_817_999_999)

    def deliver(self,capture,track=1,valid=True):
        b=TargetBelief();b.header.stamp=stamp(capture);b.header.frame_id='world'
        b.episode_id=0;b.track_id=track;b.view_id='overhead';b.camera_only=True;b.valid=valid
        b.state=[.46,0.,0.,0.];b.covariance=[1e-6 if i%5==0 else 0. for i in range(16)]
        self.n.receive_targets(S(episode_id=0,targets=[b],header=S(stamp=stamp(self.sim))))

    def stream(self,hz=15.,delay=.055,seconds=2.):
        first=7_884_666_666
        self.queue=[(first+round(i/hz*1e9)+round(delay*1e9),first+round(i/hz*1e9))
                    for i in range(round(hz*seconds))]

    def step(self,**kwargs):
        self.wall+=.002;self.sim+=2_000_000
        self.n.q_received=self.wall;self.n.q_stamp=self.sim-2_000_000;self.n.q_sequence+=1
        while self.queue and self.queue[0][0]<=self.sim:
            _,capture=self.queue.pop(0);self.deliver(capture)

    def test_reported_old_cache_cannot_start_transport(self):
        self.stream();self.n.start_transport()
        ready=next(e for e in self.n.events if e['state']=='OVERHEAD_READY')
        self.assertGreaterEqual(ready['captures'],3)
        self.assertLessEqual(ready['feedback']['capture_age_sim_s'],.075)
        self.assertGreater(self.commands[0][0],7_967_999_999)
        self.assertEqual(self.commands[0][1],self.c['belt_speed_m_s'])

    def test_fifteen_hz_delayed_stream_remains_fresh_after_start(self):
        self.stream();self.n.start_transport();self.n.wait_sim(.6)
        self.assertTrue(self.n.tracking_active)
        self.assertLess(self.n.overhead_diagnostic['capture_age_sim_s'],.15)

    def test_old_ten_hz_delay_cannot_arm_without_relaxing_limit(self):
        self.stream(hz=10.)
        with self.assertRaisesRegex(TimeoutError,'three fresh'):self.n.start_transport()
        self.assertFalse(self.n.tracking_active);self.assertEqual(self.commands,[])

    def test_one_cached_capture_has_bounded_startup_failure(self):
        with self.assertRaisesRegex(TimeoutError,'three fresh'):self.n.start_transport()
        self.assertEqual(self.commands,[])
        self.assertFalse(self.n.tracking_active)

    def test_stopped_stream_still_fails_when_active(self):
        self.stream();self.n.start_transport();self.queue=[]
        with self.assertRaisesRegex(ValueError,'stale or future'):self.n.wait_sim(.5)

    def test_track_change_is_not_hidden_by_startup_wait(self):
        self.stream();self.n.start_transport();self.deliver(self.sim-10_000_000,track=2)
        with self.assertRaisesRegex(ValueError,'identity changed'):self.n.wait_sim(.1)

    def test_budgeted_clock_first_replay_reproduces_old_rate_failure(self):
        captures=[round(i*.1*1e9) for i in range(30)]
        r=replay_capture_delivery(captures,.15,.06,.015)
        self.assertGreater(r['stale_checks'],0)

    def test_new_rate_passes_delivery_and_clock_callback_budgets(self):
        captures=[round(i/self.c['camera_rate_hz']*1e9) for i in range(45)]
        r=replay_capture_delivery(captures,.15,.06,.015)
        self.assertEqual(r['stale_checks'],0)
        self.assertLess(r['maximum_capture_age_sim_s'],.15)


if __name__=='__main__':unittest.main()
