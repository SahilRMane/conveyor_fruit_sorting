"""Exercise the actual class selector under cross-topic callback ordering."""
import copy,itertools,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np,yaml
from rclpy.serialization import serialize_message,deserialize_message
from sorting_interfaces.msg import TargetBelief,TargetBeliefArray,ClassObservation
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/sorting_control/scripts'),str(R/'src/sorting_perception/scripts')]
from camera_sort import SortController
from classifier_state import EvidenceStore
from m2b_common import timestamp

class PairingTests(unittest.TestCase):
    def controller(self):
        c=SortController.__new__(SortController)
        c.s=yaml.safe_load((R/'src/sorting_control/config/m3d.yaml').read_text())
        c.epoch=NS(episode_id=2);c.bound_track=7;c.classes={};c.classified=None
        c.result={};c.phase='pregrasp_refinement';c.model_sha='test-model';c.clock=NS(ns=1_050_000_000)
        c.now_ns=lambda:c.clock.ns;c.timeline=[];c.seen=[]
        def until(predicate,timeout):
            c.requested_timeout=timeout
            for now,kind,message in c.timeline:
                c.clock.ns=now
                if kind=='score':c.class_score(copy.deepcopy(message))
                else:c.classified=copy.deepcopy(message)
                ready=predicate();c.seen.append(copy.deepcopy(c.result.get('last_class_pairing')))
                if ready:return
            raise TimeoutError('Class pairing wait expired')
        c.until=until
        return c
    def evidence(self,c,capture=1_000_000_000,p=None):
        probabilities=[.99,.01] if p is None else p
        b=TargetBelief(episode_id=2,track_id=7,view_id='wrist',valid=True,camera_only=True,
            fixed_handle_z=.4229,state=[.55,0.,0.,0.],covariance=(np.eye(4)*1e-7).reshape(-1).tolist())
        b.header.frame_id='world';b.header.stamp=timestamp(capture)
        array=TargetBeliefArray(episode_id=2,targets=[b])
        score=ClassObservation(episode_id=2,view_id='wrist',valid=True,accepted=max(probabilities)>=.8,
            class_ids=c.s['class_ids'],probabilities=probabilities,model_sha256=c.model_sha)
        score.header.frame_id='wrist_camera_optical_frame';score.header.stamp=timestamp(capture)
        store=EvidenceStore(yaml.safe_load((R/'src/sorting_perception/config/m3c.yaml').read_text()));store.reset(2,0)
        now=capture+50_000_000
        blank=store.enrich(array,now);store.add(score);enriched=store.enrich(array,now)
        wire=lambda value:deserialize_message(serialize_message(value),type(value))
        return dict(blank=wire(blank),score=wire(score),enriched=wire(enriched)),now
    def schedule(self,c,order=('blank','score','enriched'),count=3,p=None):
        for index in range(count):
            messages,now=self.evidence(c,1_000_000_000+index*70_000_000,p)
            for kind in order:c.timeline.append((now,'score' if kind=='score' else 'target',messages[kind]))
    def test_all_valid_cross_topic_orders_wait_and_then_route(self):
        for order in itertools.permutations(('blank','score','enriched')):
            # blank and enriched share a publisher, whose order is preserved.
            if order.index('blank')>order.index('enriched'):continue
            with self.subTest(order=order):
                c=self.controller();self.schedule(c,order);route=c.select_class('red_fruit')
                self.assertEqual(route['bin_id'],'bin_a');self.assertEqual(route['distinct_captures'],3)
                self.assertEqual(c.result['last_class_pairing']['status'],'matched')
    def test_blank_target_with_available_score_is_pending_not_mismatch(self):
        c=self.controller();messages,now=self.evidence(c)
        c.timeline=[(now,'target',messages['blank']),(now,'score',messages['score'])]
        with self.assertRaises(TimeoutError):c.select_class()
        self.assertEqual(c.result['last_class_pairing']['status'],'waiting_for_target_enrichment')
        self.assertEqual(c.result['last_class_pairing']['target_class_ids'],[])
        self.assertEqual(c.result['last_class_pairing']['score_class_ids'],c.s['class_ids'])
        self.assertEqual(c.requested_timeout,3.)
    def test_enriched_target_before_score_waits_for_that_score(self):
        c=self.controller();self.schedule(c,('blank','enriched','score'))
        self.assertEqual(c.select_class()['class_id'],'red_fruit')
        self.assertTrue(any(d and d['status']=='waiting_for_score' for d in c.seen))
    def test_empty_classes_do_not_count_towards_capture_proof(self):
        c=self.controller();self.schedule(c,('blank','score'),count=10)
        with self.assertRaises(TimeoutError):c.select_class()
        self.assertEqual(c.result['last_class_pairing']['status'],'waiting_for_target_enrichment')
    def test_missing_evidence_cannot_authorize_pick_or_pass(self):
        c=self.controller();self.schedule(c,('blank',),count=10)
        with self.assertRaises(TimeoutError):c.select_class()
    def test_duplicate_publications_are_not_distinct_captures(self):
        c=self.controller();messages,now=self.evidence(c)
        c.timeline=[(now,'score',messages['score'])]+[(now,'target',messages['enriched'])]*20
        with self.assertRaises(TimeoutError):c.select_class()
    def test_complete_nonempty_conflicting_probabilities_abort(self):
        c=self.controller();messages,now=self.evidence(c);messages['enriched'].targets[0].class_probabilities=[.01,.99]
        c.timeline=[(now,'score',messages['score']),(now,'target',messages['enriched'])]
        with self.assertRaisesRegex(RuntimeError,'Class/target evidence mismatch'):c.select_class()
        self.assertEqual(c.result['last_class_pairing']['status'],'mismatch')
    def test_partial_or_wrong_length_class_fields_abort(self):
        for ids,p in [([], [.99,.01]),(['red_fruit','green_fruit'],[]),(['red_fruit','green_fruit'],[1.])]:
            with self.subTest(ids=ids,p=p):
                c=self.controller();messages,now=self.evidence(c)
                b=messages['enriched'].targets[0];b.class_ids=ids;b.class_probabilities=p
                c.timeline=[(now,'score',messages['score']),(now,'target',messages['enriched'])]
                with self.assertRaisesRegex(RuntimeError,'Class/target evidence mismatch'):c.select_class()
    def test_class_order_mismatch_aborts(self):
        c=self.controller();messages,now=self.evidence(c);messages['enriched'].targets[0].class_ids=list(reversed(c.s['class_ids']))
        c.timeline=[(now,'score',messages['score']),(now,'target',messages['enriched'])]
        with self.assertRaisesRegex(RuntimeError,'Class/target evidence mismatch'):c.select_class()
    def test_model_change_aborts_even_when_target_class_fields_are_pending(self):
        c=self.controller();messages,now=self.evidence(c);messages['score'].model_sha256='other-model'
        c.timeline=[(now,'score',messages['score']),(now,'target',messages['blank'])]
        with self.assertRaisesRegex(RuntimeError,'Classifier model changed'):c.select_class()
    def test_nearby_capture_score_is_not_substituted(self):
        c=self.controller();messages,now=self.evidence(c);messages['score'].header.stamp=timestamp(1_002_000_000)
        c.timeline=[(now,'score',messages['score']),(now,'target',messages['enriched'])]
        with self.assertRaises(TimeoutError):c.select_class()
        self.assertEqual(c.result['last_class_pairing']['status'],'waiting_for_score')
    def test_old_episode_and_other_track_cannot_route(self):
        for field in ('episode_id','track_id'):
            c=self.controller();self.schedule(c)
            for _,kind,m in c.timeline:
                if kind=='target':setattr(m.targets[0],field,9)
            with self.assertRaises(TimeoutError):c.select_class()
    def test_stale_and_future_pairs_cannot_route(self):
        for age in (150_000_000,-21_000_000):
            c=self.controller();messages,_=self.evidence(c)
            c.timeline=[(1_000_000_000+age,'score',messages['score']),(1_000_000_000+age,'target',messages['enriched'])]
            with self.assertRaises(TimeoutError):c.select_class()
    def test_covered_complete_pairs_still_select_pass(self):
        c=self.controller();self.schedule(c,p=[.52,.48]);route=c.select_class()
        self.assertEqual(route['action'],'PASS');self.assertEqual(route['bin_id'],'')
    def test_required_class_cannot_change_during_pregrasp(self):
        c=self.controller();self.schedule(c,p=[.01,.99])
        with self.assertRaisesRegex(RuntimeError,'Class evidence changed at pregrasp'):c.select_class('red_fruit')
    def test_green_complete_pairs_select_other_bin(self):
        c=self.controller();self.schedule(c,p=[.01,.99]);self.assertEqual(c.select_class('green_fruit')['bin_id'],'bin_b')
    def test_near_equal_but_different_vectors_are_rejected(self):
        c=self.controller();messages,now=self.evidence(c);messages['enriched'].targets[0].class_probabilities=[.99000001,.00999999]
        c.timeline=[(now,'score',messages['score']),(now,'target',messages['enriched'])]
        with self.assertRaisesRegex(RuntimeError,'Class/target evidence mismatch'):c.select_class()

if __name__=='__main__':unittest.main()
