#!/usr/bin/env python3
"""Stationary live RGB classification and exact-capture target enrichment."""
import argparse,json,time,uuid
from datetime import datetime
from pathlib import Path
import numpy as np,yaml
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from sorting_interfaces.msg import ClassObservation,TargetBeliefArray
from ament_index_python.packages import get_package_share_directory
from validate_m3c_files import installed_sources
from classifier_state import ns

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight-dir',type=Path,required=True);p.add_argument('--expected-class',choices=('red_fruit','green_fruit'),required=True);p.add_argument('--expect-abstain',action='store_true');p.add_argument('--samples',type=int,default=30);p.add_argument('--output-dir',type=Path);a=p.parse_args()
    pre=json.loads((a.preflight_dir/'validation.json').read_text())
    if pre.get('status')!='PASS' or pre.get('checkpoint')!='M3C' or pre.get('installed_sources')!=installed_sources():p.error('Run M3C preflight again for these installed files')
    if a.samples<20:p.error('Use at least 20 distinct captures')
    c=yaml.safe_load((Path(get_package_share_directory('sorting_perception'))/'config/m3c.yaml').read_text());folder=(a.output_dir or Path('results')/f'M3C-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}').resolve();folder.mkdir(parents=True,exist_ok=False)
    rclpy.init();node=Node('test_m3c',parameter_overrides=[Parameter('use_sim_time',value=True)]);scores={};targets={};owner={};state={};barrier=None;episode=None
    def score(m):
        nonlocal episode
        if barrier is None or not m.valid or ns(m.header.stamp)<=barrier:return
        if episode is None:episode=m.episode_id
        if m.episode_id!=episode:return
        age=(node.get_clock().now().nanoseconds-ns(m.header.stamp))/1e9
        if not -.02<=age<c['maximum_capture_age_s']:return
        scores[ns(m.header.stamp)]=dict(episode_id=m.episode_id,capture_ns=ns(m.header.stamp),age_sim_s=age,probabilities=list(m.probabilities),class_ids=list(m.class_ids),accepted=m.accepted,
            inference_wall_s=m.inference_wall_s,model_sha256=m.model_sha256,reason=m.reason)
    def target(m):
        for b in m.targets:
            if b.valid and b.class_ids:targets[ns(b.header.stamp)]=dict(episode_id=b.episode_id,track_id=b.track_id,class_ids=list(b.class_ids),probabilities=list(b.class_probabilities))
    def own(m):owner.clear();owner.update(json.loads(m.data))
    def status(m):state.clear();state.update(json.loads(m.data))
    node.create_subscription(ClassObservation,'/sorting/perception/class_observation',score,10);node.create_subscription(TargetBeliefArray,'/sorting/classified_targets',target,10)
    node.create_subscription(String,'/sorting/arm/owner_state',own,1);node.create_subscription(String,'/sorting/perception/classifier_state',status,1)
    report=dict(checkpoint='M3C',status='FAIL',full_M3_gate_passed=False,classifier_ready=False,physical_grasp='NOT_RUN',mode='stationary_rgb_classifier')
    try:
        end=time.monotonic()+45.
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.02)
            if owner.get('owner')=='FAULT':raise RuntimeError('Arm gate fault')
            if barrier is None and owner.get('owner')=='HOLD' and owner.get('ready') and state.get('classifier_ready'):barrier=node.get_clock().now().nanoseconds
            paired=sorted(set(scores)&set(targets))
            if len(paired)>=a.samples:break
        if len(paired)<a.samples:raise TimeoutError('Not enough distinct fresh class/target capture pairs')
        rows=[scores[k] for k in paired[:a.samples]]
        if not all(r['model_sha256']==pre['model_sha256'] for r in rows):raise RuntimeError('Live model differs from preflight')
        if not all(targets[k]['episode_id']==scores[k]['episode_id'] and targets[k]['class_ids']==scores[k]['class_ids'] and np.allclose(targets[k]['probabilities'],scores[k]['probabilities'],atol=1e-12) for k in paired[:a.samples]):raise RuntimeError('Class evidence/target mismatch')
        expected=c['class_ids'].index(a.expected_class);accepted=np.array([r['accepted'] for r in rows]);prediction=np.array([np.argmax(r['probabilities']) for r in rows]);coverage=float(accepted.mean());accuracy=float((prediction==expected).mean())
        if a.expect_abstain:
            if coverage>c['maximum_hidden_coverage']:raise RuntimeError('Covered fixture produces too many confident classes')
        elif accuracy<c['minimum_clear_accuracy'] or coverage<c['minimum_clear_coverage']:raise RuntimeError('Live clear classification or coverage below gate')
        if owner.get('owner')!='HOLD' or owner.get('stats',{}).get('overlap_violations')!=0 or owner.get('stats',{}).get('planned_accepted') or owner.get('stats',{}).get('servo_input_accepted'):raise RuntimeError('Classifier checkpoint must leave the arm in exclusive HOLD')
        # Inspect the real classifier graph: no evaluator/truth input or motion output.
        names=node.get_node_names_and_namespaces();classifier=[n for n,s in names if n=='fruit_classifier']
        if len(classifier)!=1:raise RuntimeError('Expected one classifier node')
        subs=node.get_subscriber_names_and_types_by_node('fruit_classifier','/');pubs=node.get_publisher_names_and_types_by_node('fruit_classifier','/')
        if any('ground_truth' in name or name in ('/sorting/contact_state','/sorting/reset_state') for name,_ in subs):raise RuntimeError('Classifier truth input detected')
        if any(name.startswith('/sorting/arm/') or name=='/sorting/conveyor_command' for name,_ in pubs):raise RuntimeError('Classifier motion output detected')
        report.update(status='PASS',classifier_ready=True,samples=len(rows),coverage=coverage,accuracy=None if a.expect_abstain else accuracy,expected_abstention=a.expect_abstain,
            exact_capture_target_pairs='PASS',evaluator_isolation='PASS',final_owner='HOLD',overlap_violations=0,maximum_age_sim_s=max(r['age_sim_s'] for r in rows),inference_p95_wall_s=float(np.percentile([r['inference_wall_s'] for r in rows],95)))
    except Exception as error:report['error']=str(error)
    finally:
        report['output_directory']=str(folder);(folder/'samples.json').write_text(json.dumps(list(scores.values()),indent=2)+'\n');(folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n');node.destroy_node();rclpy.try_shutdown()
    print(json.dumps(report,indent=2));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
