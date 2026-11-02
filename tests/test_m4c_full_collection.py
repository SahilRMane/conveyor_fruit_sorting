"""Keep young collection active and restore exact thresholds after a test."""
import gc
import weakref
import unittest
from pathlib import Path
import sys
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4c_timing import FullCollectionWindow
from m4c_runtime import MovingInspectionExecutor
from test_m4b import ClassInspectionExecutor


class FullCollectionWindowTests(unittest.TestCase):
    def setUp(self):
        self.saved=gc.get_threshold();self.enabled=gc.isenabled();self.w=FullCollectionWindow()
        self.addCleanup(self.cleanup)
    def cleanup(self):
        self.w.restore('unit_cleanup');gc.set_threshold(*self.saved)
        if self.enabled:gc.enable()
        else:gc.disable()

    def test_only_old_generation_threshold_changes(self):
        self.w.begin();active=gc.get_threshold()
        self.assertEqual(active[:2],self.saved[:2]);self.assertGreaterEqual(active[2],2**31-1)
        self.assertEqual(gc.isenabled(),self.enabled)

    def test_younger_cycles_can_still_be_collected(self):
        class Cycle:pass
        self.w.begin();x=Cycle();x.x=x;reference=weakref.ref(x);del x
        gc.collect(0);self.assertIsNone(reference())

    def test_exact_thresholds_are_restored(self):
        self.w.begin();self.w.restore('confirmed_hold')
        self.assertEqual(gc.get_threshold(),self.saved)
        self.assertEqual(self.w.report()['restore_reason'],'confirmed_hold')

    def test_restore_is_idempotent(self):
        self.w.begin();self.w.restore('confirmed_hold');first=self.w.report()
        self.w.restore('finalization_fallback');self.assertEqual(self.w.report(),first)

    def test_explicit_full_collection_remains_visible(self):
        self.w.begin();gc.collect(2);self.w.restore('confirmed_hold')
        self.assertGreaterEqual(self.w.report()['active_full_collections'],1)

    def test_disabled_collector_remains_disabled(self):
        gc.disable();before=gc.get_threshold();self.w.begin()
        self.assertFalse(self.w.report()['applied']);self.w.restore('confirmed_hold')
        self.assertFalse(gc.isenabled());self.assertEqual(gc.get_threshold(),before)

    def test_zero_threshold_remains_unchanged(self):
        gc.set_threshold(0,7,9);self.w.begin()
        self.assertEqual(gc.get_threshold(),(0,7,9));self.w.restore('confirmed_hold')
        self.assertEqual(gc.get_threshold(),(0,7,9))

    def test_cannot_begin_twice(self):
        self.w.begin()
        with self.assertRaisesRegex(ValueError,'already started'):self.w.begin()

    def test_unused_window_changes_no_settings(self):
        self.w.restore('finalization_fallback')
        self.assertEqual(gc.get_threshold(),self.saved);self.assertFalse(self.w.report()['started'])

    def test_report_labels_memory_as_peak_kib(self):
        self.w.begin();self.w.restore('confirmed_hold');r=self.w.report()
        self.assertGreater(r['peak_rss_before_kib'],0)
        self.assertGreaterEqual(r['peak_rss_after_kib'],r['peak_rss_before_kib'])

    def test_runtime_restores_only_after_successful_hold(self):
        n=object.__new__(MovingInspectionExecutor);n.tracking_active=True;n.operation={}
        n.full_collection_window=self.w;self.w.begin()
        with patch.object(ClassInspectionExecutor,'stop') as stop:n.stop()
        stop.assert_called_once();self.assertEqual(self.w.report()['restore_reason'],'confirmed_hold')

    def test_failed_hold_leaves_restoration_to_finalization(self):
        n=object.__new__(MovingInspectionExecutor);n.tracking_active=True;n.operation={}
        n.full_collection_window=self.w;self.w.begin()
        with patch.object(ClassInspectionExecutor,'stop',side_effect=RuntimeError('hold failed')):
            with self.assertRaises(RuntimeError):n.stop()
        self.assertFalse(self.w.report()['restored']);self.w.restore('finalization_fallback')
        self.assertEqual(gc.get_threshold(),self.saved)


if __name__=='__main__':unittest.main()
