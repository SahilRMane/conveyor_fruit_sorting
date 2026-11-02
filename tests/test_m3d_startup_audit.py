"""Bounded ROS graph discovery; observed isolation failures never wait."""
from pathlib import Path
import copy,sys,unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/sorting_control/scripts'),str(R/'src/sorting_perception/scripts')]
from m3d_evaluator import SortEvaluator

INPUTS={
 'handle_localizer':['/camera/wrist/image_raw','/camera/wrist/camera_info','/sorting/episode'],
 'camera_motion_filter':['/sorting/perception/wrist/handle_observation','/sorting/episode'],
 'camera_grasp_scene':['/sorting/targets','/joint_states','/sorting/episode'],
 'fruit_classifier':['/camera/wrist/image_raw','/sorting/targets','/sorting/episode'],
 'camera_sort_controller':['/sorting/classified_targets','/sorting/perception/class_observation','/sorting/targets','/sorting/arm/owner_state']}
TRUTH=['/sorting/contact_state','/sorting/ground_truth/camera_sample','/sorting/reset_state']

class AuditTests(unittest.TestCase):
    def complete(self):
        return dict(graph=[(n,'/') for n in INPUTS]+[('test_m3d','/')],inputs=copy.deepcopy(INPUTS),
                    observers={t:[NS(node_name='test_m3d',node_namespace='/')] for t in TRUTH})
    def evaluator(self,snapshots):
        e=SortEvaluator.__new__(SortEvaluator);e.node=Mock();e.audit_diagnostic=None;e.index=0;e.visited=0
        e.node.get_node_names_and_namespaces.side_effect=lambda:snapshots[e.index]['graph']
        e.node.get_subscriber_names_and_types_by_node.side_effect=lambda n,space:[(t,['unused']) for t in snapshots[e.index]['inputs'].get(n,[])]
        e.node.get_subscriptions_info_by_topic.side_effect=lambda t:snapshots[e.index]['observers'].get(t,[])
        def until(predicate,timeout):
            e.requested_timeout=timeout
            for index in range(len(snapshots)):
                e.index=index;e.visited+=1
                if predicate():return
            raise TimeoutError('discovery has not completed')
        e.until=until;return e
    def test_complete_graph_passes_without_delay(self):
        e=self.evaluator([self.complete()]);r=e.audit()
        self.assertEqual(r['status'],'PASS');self.assertEqual(e.visited,1)
        self.assertEqual(r['discovery']['missing_nodes'],[])
    def test_partial_names_are_waited_for_before_audit_passes(self):
        first=self.complete();first['graph']=[v for v in first['graph'] if v[0]!='camera_sort_controller']
        e=self.evaluator([first,self.complete()]);self.assertEqual(e.audit()['status'],'PASS')
        self.assertEqual(e.visited,2);self.assertEqual(e.requested_timeout,10.)
    def test_subscription_discovery_can_lag_node_discovery(self):
        first=self.complete();first['inputs']['fruit_classifier']=[]
        e=self.evaluator([first,self.complete()]);self.assertEqual(e.audit()['status'],'PASS');self.assertEqual(e.visited,2)
    def test_self_observer_discovery_is_required(self):
        first=self.complete();first['observers'][TRUTH[0]]=[]
        e=self.evaluator([first,self.complete()]);self.assertEqual(e.audit()['status'],'PASS');self.assertEqual(e.visited,2)
    def test_missing_node_timeout_identifies_that_node(self):
        first=self.complete();first['graph']=[v for v in first['graph'] if v[0]!='fruit_classifier']
        e=self.evaluator([first])
        with self.assertRaisesRegex(TimeoutError,'discovery timed out'):e.audit()
        self.assertEqual(e.audit_diagnostic['missing_nodes'],['fruit_classifier'])
    def test_missing_input_timeout_records_exact_topic(self):
        first=self.complete();first['inputs']['camera_sort_controller'].remove('/sorting/classified_targets')
        e=self.evaluator([first])
        with self.assertRaises(TimeoutError):e.audit()
        self.assertEqual(e.audit_diagnostic['missing_policy_subscriptions'],{'camera_sort_controller':['/sorting/classified_targets']})
    def test_oracle_is_rejected_immediately_even_with_incomplete_graph(self):
        first=self.complete();first['graph']=[('oracle_target_source','/')]
        e=self.evaluator([first,self.complete()])
        with self.assertRaisesRegex(RuntimeError,'Oracle node present'):e.audit()
        self.assertEqual(e.visited,1);self.assertEqual(e.audit_diagnostic['oracle_nodes'],['oracle_target_source'])
    def test_policy_truth_or_debug_input_is_rejected_before_it_can_disappear(self):
        for topic in ('/sorting/ground_truth/camera_sample','/sorting/debug/oracle_target','/sorting/contact_state','/sorting/reset_state'):
            first=self.complete();first['inputs']['fruit_classifier'].append(topic)
            e=self.evaluator([first,self.complete()])
            with self.assertRaisesRegex(RuntimeError,'Truth input'):e.audit()
            self.assertEqual(e.visited,1)
    def test_unauthorized_truth_observer_is_rejected_immediately(self):
        first=self.complete();first['observers'][TRUTH[0]].append(NS(node_name='extra_observer',node_namespace='/'))
        e=self.evaluator([first,self.complete()])
        with self.assertRaisesRegex(RuntimeError,'subscriber outside'):e.audit()
        self.assertEqual(e.visited,1)
    def test_same_evaluator_name_in_other_namespace_is_not_authorized(self):
        first=self.complete();first['observers'][TRUTH[0]]=[NS(node_name='test_m3d',node_namespace='/other')]
        e=self.evaluator([first])
        with self.assertRaises(RuntimeError):e.audit()
    def test_required_node_in_wrong_namespace_does_not_pass(self):
        first=self.complete();first['graph']=[(name,'/other' if name=='fruit_classifier' else space) for name,space in first['graph']]
        e=self.evaluator([first])
        with self.assertRaises(TimeoutError):e.audit()
        self.assertEqual(e.audit_diagnostic['missing_nodes'],['fruit_classifier'])
    def test_unrelated_nodes_and_permitted_extra_inputs_do_not_block_audit(self):
        first=self.complete();first['graph'].append(('move_group','/m2b_planner'))
        first['inputs']['fruit_classifier'].append('/camera/wrist/camera_info')
        self.assertEqual(self.evaluator([first]).audit()['status'],'PASS')

if __name__=='__main__':unittest.main()
