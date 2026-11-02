"""Confidence branches, state/episode binding and grouped dataset isolation."""
from pathlib import Path
import copy
import sys
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
from m4b_core import SingleViewPolicy
from m4b_data import audit, POSES, CONDITIONS


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.c=yaml.safe_load((ROOT/'src/sorting_supervisor/config/m4b.yaml').read_text())
        self.p=SingleViewPolicy(3,1_000_000_000,self.c)

    def observe(self,p=(.5,.5),**kwargs):
        return self.p.observe(**dict(dict(episode=3,pose='assess',capture_ns=1_050_000_000,
                                         now_ns=1_100_000_000,probabilities=p),**kwargs))

    def inspect(self):
        self.assertEqual(self.observe()['action'],'INSPECT')
        self.p.start_inspection(1_200_000_000)

    def test_clear_red(self):
        d=self.observe((.9,.1));self.assertEqual((d['action'],d['class_id'],d['proposed_bin']),('PICK_READY','red_fruit','bin_a'))

    def test_clear_green(self):
        d=self.observe((.1,.9));self.assertEqual((d['action'],d['class_id'],d['proposed_bin']),('PICK_READY','green_fruit','bin_b'))

    def test_inspection_resolves(self):
        self.inspect();d=self.observe((.1,.9),pose='inspect_left',capture_ns=1_300_000_000,now_ns=1_350_000_000)
        self.assertEqual(d['action'],'PICK_READY');self.assertEqual(self.p.inspections,1)

    def test_inspection_remains_uncertain(self):
        self.inspect();d=self.observe(pose='inspect_left',capture_ns=1_300_000_000,now_ns=1_350_000_000)
        self.assertEqual(d['action'],'PASS')

    def test_no_physical_pick_authorization(self):
        self.assertFalse(self.observe((.9,.1))['physical_pickup_permitted'])

    def test_proposal_only(self):
        self.assertTrue(self.observe((.9,.1))['proposal_only'])

    def test_stale_pass(self):
        self.assertEqual(self.observe((.9,.1),now_ns=1_250_000_000)['action'],'PASS')

    def test_future_pass(self):
        self.assertEqual(self.observe((.9,.1),now_ns=1_000_000_000)['action'],'PASS')

    def test_before_reset_pass(self):
        self.assertEqual(self.observe((.9,.1),capture_ns=999_999_999)['action'],'PASS')

    def test_cross_episode_rejected(self):
        with self.assertRaises(ValueError):self.observe((.9,.1),episode=4)

    def test_wrong_view_rejected(self):
        with self.assertRaises(ValueError):self.observe((.9,.1),pose='inspect_left')

    def test_inspection_requires_start(self):
        self.observe()
        with self.assertRaises(ValueError):self.observe((.9,.1),pose='inspect_left')

    def test_previous_view_cannot_authorize_pick(self):
        self.inspect()
        self.assertEqual(self.observe((.9,.1),pose='inspect_left',capture_ns=1_100_000_000,now_ns=1_350_000_000)['action'],'PASS')

    def test_no_second_inspection(self):
        self.inspect()
        with self.assertRaises(ValueError):self.p.start_inspection(1_250_000_000)

    def test_terminal_immutable(self):
        self.observe((.9,.1))
        with self.assertRaises(ValueError):self.observe((.1,.9))

    def test_bad_probability_sum_pass(self):
        self.assertEqual(self.observe((.9,.9))['action'],'PASS')

    def test_nan_pass(self):
        self.assertEqual(self.observe((float('nan'),.1))['action'],'PASS')

    def test_probability_shape_pass(self):
        self.assertEqual(self.observe((1.,))['action'],'PASS')

    def test_confidence_threshold_unchanged(self):
        self.assertEqual(self.c['acceptance_confidence'],.8)


class GroupTests(unittest.TestCase):
    def rows(self):
        return [dict(group_id=g,appearance_instance=700+g,split=s,pose=p,condition=c,label=l)
                for g,s in [(0,'train'),(1,'test')] for p in POSES for c in CONDITIONS for l in (0,1)]

    def test_matched_groups(self):self.assertTrue(audit(self.rows(),dict(train=1,test=1))['groups_disjoint'])
    def test_group_leakage(self):
        rows=self.rows();rows[0]['split']='test'
        with self.assertRaises(ValueError):audit(rows)
    def test_appearance_leakage(self):
        rows=self.rows()
        for r in rows:r['appearance_instance']=1
        with self.assertRaises(ValueError):audit(rows)
    def test_incomplete_pair_rejected(self):
        with self.assertRaises(ValueError):audit(self.rows()[1:])
    def test_duplicated_pair_rejected(self):
        rows=self.rows();rows[0]=copy.deepcopy(rows[1])
        with self.assertRaises(ValueError):audit(rows)
    def test_group_counts_bound(self):
        with self.assertRaises(ValueError):audit(self.rows(),dict(train=2,test=0))


if __name__=='__main__':unittest.main()
