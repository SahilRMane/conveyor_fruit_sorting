from pathlib import Path
import copy,json,sys,tempfile,unittest
from unittest.mock import Mock,patch,DEFAULT
import numpy as np,yaml
from sorting_interfaces.msg import ClassObservation,TargetBelief,TargetBeliefArray
from rclpy.serialization import serialize_message,deserialize_message
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src/sorting_perception/scripts')]
from classifier_core import softmax,fit_temperature,nll,metrics,fruit_crop,features,Network,Classifier,config_digest,digest
from classifier_state import EvidenceStore
from m3c_data import audit_splits
from validate_m3c_files import assess
class ProbabilityTests(unittest.TestCase):
    def test_extreme_logits_are_finite_and_normalized(self):
        p=softmax(np.array([[1e5,1e5-2.],[-1e5,-1e5]]));self.assertTrue(np.isfinite(p).all());np.testing.assert_allclose(p.sum(axis=1),1.)
    def test_temperature_fit_reduces_overconfident_nll(self):
        logits=np.tile([5.,-5.],(20,1));labels=np.array([0]*12+[1]*8);t=fit_temperature(logits,labels,[.25,20.]);self.assertGreater(t,1.);self.assertLess(nll(logits,labels,t),nll(logits,labels))
    def test_temperature_preserves_class_order(self):
        logits=np.array([[2.,-1.],[-2.,1.]]);np.testing.assert_array_equal(softmax(logits).argmax(axis=1),softmax(logits,3.).argmax(axis=1))
    def test_nonfinite_logits_or_invalid_temperature_rejected(self):
        for logits,t in [(np.array([[np.nan,0.]]),1.),(np.array([[1.,0.]]),0.)]:
            with self.assertRaises(ValueError):softmax(logits,t)
    def test_reliability_bins_count_all_samples(self):
        r=metrics(np.array([[1.,0.],[.5,.5],[.2,.8]]),[0,1,1],10);self.assertEqual(sum(b['count'] for b in r['reliability']),3);self.assertAlmostEqual(r['accuracy'],2/3)
class CropTests(unittest.TestCase):
    def setUp(self):self.c=yaml.safe_load((ROOT/'src/sorting_perception/config/m3c.yaml').read_text())
    def test_crop_uses_known_offset_and_camera_geometry(self):
        c=dict(self.c,fruit_offset_from_handle_m=[0,0,0],fruit_crop_half_extent_m=[.01,.01,.01]);rgb=np.zeros((100,100,3),np.uint8);k=np.array([[100.,0,50],[0,100,50],[0,0,1.]])
        crop,bbox=fruit_crop(rgb,k,[0,0,0],np.eye(3),[0,0,.2],c);self.assertEqual(bbox,[44,44,57,57]);self.assertEqual(features(crop,c).shape,(768,))
    def test_clipped_and_behind_camera_crops_fail(self):
        rgb=np.zeros((100,100,3),np.uint8);k=np.array([[100.,0,50],[0,100,50],[0,0,1.]])
        for p in ([2,0,.2],[0,0,-.2]):
            with self.assertRaises(ValueError):fruit_crop(rgb,k,[0,0,0],np.eye(3),p,self.c)
class SplitTests(unittest.TestCase):
    def fixture(self):
        c=dict(split_group_counts=dict(train=1,validation=1,calibration=1,test=1));rows=[]
        for i,s in enumerate(c['split_group_counts']):
            for kind in (0,1):
                for view in range(3):rows.append(dict(group_id=i,appearance_instance=i+100,split=s,label=kind))
        return c,rows
    def test_groups_are_split_before_frames(self):
        c,r=self.fixture();self.assertTrue(audit_splits(r,c)['groups_disjoint'])
    def test_one_frame_crossing_a_group_split_fails(self):
        c,r=self.fixture();r[0]['split']='calibration'
        with self.assertRaisesRegex(ValueError,'leakage'):audit_splits(r,c)
    def test_shared_appearance_across_splits_fails(self):
        c,r=self.fixture();r[6]['appearance_instance']=100
        with self.assertRaisesRegex(ValueError,'leakage'):audit_splits(r,c)
    def test_incomplete_counterfactual_pair_fails(self):
        c,r=self.fixture();r.pop()
        with self.assertRaisesRegex(ValueError,'incomplete'):audit_splits(r,c)
class StoreTests(unittest.TestCase):
    def setUp(self):self.c=yaml.safe_load((ROOT/'src/sorting_perception/config/m3c.yaml').read_text());self.store=EvidenceStore(self.c);self.store.reset(2,100)
    def score(self):
        m=ClassObservation(episode_id=2,view_id='wrist',valid=True,accepted=False,class_ids=self.c['class_ids'],probabilities=[.52,.48]);m.header.stamp.sec=1;m.header.frame_id='wrist_camera_optical_frame';return m
    def target(self):
        b=TargetBelief(episode_id=2,track_id=8,view_id='wrist',valid=True,camera_only=True,state=[.55,0.,.03,0.],covariance=(np.eye(4)*1e-7).reshape(-1).tolist());b.header.stamp.sec=1;return TargetBeliefArray(targets=[b])
    def test_exact_capture_enrichment_preserves_motion_and_uncertain_scores(self):
        self.store.add(self.score());m=self.target();out=self.store.enrich(m,1_040_000_000)
        self.assertEqual(list(out.targets[0].class_probabilities),[.52,.48]);np.testing.assert_array_equal(out.targets[0].state,m.targets[0].state);np.testing.assert_array_equal(out.targets[0].covariance,m.targets[0].covariance);self.assertEqual(m.targets[0].class_ids,[])
        deserialize_message(serialize_message(out),TargetBeliefArray)
    def test_class_message_preserves_probabilities_calibration_and_crop(self):
        m=self.score();m.confidence=.52;m.entropy=.69;m.temperature=.67;m.inference_wall_s=.0001;m.crop_xyxy=[44,60,120,140];m.model_sha256='abc'
        out=deserialize_message(serialize_message(m),ClassObservation)
        self.assertEqual(list(out.probabilities),[.52,.48]);self.assertAlmostEqual(out.temperature,.67);self.assertEqual(list(out.crop_xyxy),[44,60,120,140])
    def test_nearby_capture_is_not_substituted(self):
        self.store.add(self.score());m=self.target();m.targets[0].header.stamp.nanosec=2_000_000;self.assertFalse(self.store.enrich(m,1_040_000_000).targets[0].class_ids)
    def test_old_score_cannot_be_refreshed_by_publication(self):
        self.store.add(self.score());self.assertFalse(self.store.enrich(self.target(),1_150_000_000).targets[0].class_ids)
    def test_epoch_reset_flushes_old_evidence(self):
        self.store.add(self.score());self.store.reset(3,1_010_000_000);self.assertFalse(self.store.add(self.score()));self.assertFalse(self.store.enrich(self.target(),1_020_000_000).targets[0].class_ids)
    def test_invalid_probability_vector_fails(self):
        m=self.score();m.probabilities=[.8,.8]
        with self.assertRaises(ValueError):self.store.add(m)
class IsolationTests(unittest.TestCase):
    def test_classifier_has_no_truth_inputs_or_motion_outputs(self):
        from fruit_classifier import FruitClassifier
        with patch('fruit_classifier.Node.__init__'),patch('fruit_classifier.Buffer'),patch('fruit_classifier.TransformListener'),patch('fruit_classifier.Classifier'),patch.multiple(FruitClassifier,
            declare_parameter=DEFAULT,get_parameter=DEFAULT,create_subscription=DEFAULT,create_publisher=DEFAULT,create_timer=DEFAULT) as methods:
            node=FruitClassifier();subs=[a.args[1] for a in methods['create_subscription'].call_args_list];pubs=[a.args[1] for a in methods['create_publisher'].call_args_list]
        self.assertTrue('/camera/wrist/image_raw' in subs);self.assertFalse(any('ground_truth' in t or t in ('/sorting/contact_state','/sorting/reset_state') for t in subs));self.assertFalse(any(t.startswith('/sorting/arm/') or t=='/sorting/conveyor_command' for t in pubs))
    def test_m3c_launch_removes_pickup_controller_and_uses_fixture_only_in_simulator(self):
        s=(ROOT/'src/sorting_control/launch/panda_m3c.launch.py').read_text();self.assertNotIn('executable="camera_pickup.py"',s);self.assertIn('executable="fruit_classifier.py"',s)
    def test_all_fixture_visual_geoms_have_zero_mass_and_collision(self):
        import xml.etree.ElementTree as ET
        for p in (ROOT/'src/sorting_description/mujoco/static_grasp').glob('props_camera_m3c_*.xml'):
            for g in ET.parse(p).getroot().findall('.//geom'):
                if g.get('name','').startswith('fruit_'):
                    self.assertEqual(g.get('mass'),'0');self.assertEqual(g.get('contype'),'0');self.assertEqual(g.get('conaffinity'),'0')
class NetworkTests(unittest.TestCase):
    def test_training_selects_validation_and_never_accepts_test_partition(self):
        c=dict(train_seed=73,input_size=1,hidden_units=8,class_ids=['a','b'],epochs=60,batch_size=8,learning_rate=.02,weight_decay=0.)
        x=np.tile(np.array([[-.5,0,0],[.5,0,0]],np.float32),(20,1));y=np.tile([0,1],20);net=Network.initial(c);before=nll(net.logits(x),y);result=net.train(x,y,(x,y),c)
        self.assertLess(nll(net.logits(x),y),before/2);self.assertFalse(result['test_used_for_selection']);self.assertEqual(result['selection_partition'],'validation')
if __name__=='__main__':unittest.main()
