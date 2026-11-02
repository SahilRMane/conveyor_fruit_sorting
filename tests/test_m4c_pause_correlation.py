"""Distinguish diagnostic deferral and collection pauses without ROS nodes."""
import gc
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4c_timing import CollectionProbe
import test_m4c_callbacks as callbacks


class CollectionProbeTests(unittest.TestCase):
    def setUp(self):
        self.wall=10.;self.cpu=2.;self.p=CollectionProbe(lambda:{'phase':'moving'})
        self.w=patch('m4c_timing.time.monotonic',side_effect=lambda:self.wall);self.c=patch('m4c_timing.time.thread_time',side_effect=lambda:self.cpu)
        self.w.start();self.c.start();self.addCleanup(self.w.stop);self.addCleanup(self.c.stop);self.addCleanup(self.p.close)

    def collection(self,generation=2,wall=.06,cpu=.05):
        self.p.callback('start',{'generation':generation});self.wall+=wall;self.cpu+=cpu
        self.p.callback('stop',{'generation':generation,'collected':42,'uncollectable':0})

    def test_registration_is_idempotent_and_reversible(self):
        before=list(gc.callbacks);self.p.start();self.p.start()
        self.assertEqual(gc.callbacks.count(self.p.callback),1)
        self.p.close();self.assertEqual(gc.callbacks,before)

    def test_probe_does_not_change_collection_settings(self):
        enabled=gc.isenabled();thresholds=gc.get_threshold();self.p.start();self.p.close()
        self.assertEqual(gc.isenabled(),enabled);self.assertEqual(gc.get_threshold(),thresholds)

    def test_collection_records_phase_wall_and_cpu(self):
        self.collection();r=self.p.report()['maximum_collection']
        self.assertAlmostEqual(r['wall_s'],.06);self.assertAlmostEqual(r['cpu_s'],.05)
        self.assertEqual(r['context']['phase'],'moving');self.assertEqual(r['generation'],2)

    def test_overlap_links_collection_to_callback_interval(self):
        self.collection();self.assertEqual(len(self.p.overlaps(9.99,10.08)),1)
        self.assertEqual(self.p.overlaps(10.1,10.2),[])

    def test_recent_is_bounded_and_peak_is_retained(self):
        self.collection(wall=.06)
        for _ in range(20):self.collection(generation=0,wall=.001,cpu=.001)
        self.assertEqual(len(self.p.recent),8);self.assertAlmostEqual(self.p.worst['wall_s'],.06)
        self.assertEqual(len(self.p.overlaps(10.,10.06)),1)


class PauseRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.h=callbacks.M4CCallbackTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.cpu=0.;self.cp=patch('m4c_runtime.time.thread_time',side_effect=lambda:self.cpu)
        self.cp.start();self.addCleanup(self.cp.stop)

    def test_elapsed_pause_is_not_reported_as_cpu_work(self):
        self.h.queue.append(lambda:setattr(self.h.replay,'wall',self.h.replay.wall+.05))
        self.h.executor.spin_once(timeout_sec=0.)
        b=self.h.executor.callback_diagnostic['current_batch']
        self.assertGreater(b['callback_execution_wall_s'],.049);self.assertEqual(b['callback_execution_cpu_s'],0.)
        self.assertEqual(b['gc_overlaps'],[])

    def test_collection_overlap_is_preserved_in_batch(self):
        p=CollectionProbe(lambda:{'phase':'moving'});self.h.executor.collection_probe=p
        def work():
            p.callback('start',{'generation':2});self.h.replay.wall+=.06;self.cpu+=.05
            p.callback('stop',{'generation':2,'collected':10,'uncollectable':0})
        self.h.queue.append(work);self.h.executor.spin_once(timeout_sec=0.)
        b=self.h.executor.callback_diagnostic['current_batch']
        self.assertEqual(b['gc_overlaps'][0]['generation'],2)
        self.assertAlmostEqual(b['callback_execution_cpu_s'],.05)

    def test_cleanup_wait_drains_timing_without_active_target_checks(self):
        n=self.h.n;seen=[];n.tracking_active=False;n.policy_started=None;n.pipeline_status={}
        n.timing=S(status='PASS',sample=lambda *a,**k:seen.append(k),note=lambda *a,**k:None)
        n.target=lambda:(_ for _ in ()).throw(AssertionError('Cleanup must not check active target'))
        self.h.queue.append(lambda:None)
        n.until(lambda:True,check=False)
        self.assertEqual(len(seen),1);self.assertFalse(seen[0]['enforce'])


if __name__=='__main__':unittest.main()
