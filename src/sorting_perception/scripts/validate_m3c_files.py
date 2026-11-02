#!/usr/bin/env python3
"""Held-out classifier, calibration, crop and native runtime validation."""
import argparse,json,time,subprocess,hashlib,os
from pathlib import Path
import numpy as np,yaml
from ament_index_python.packages import get_package_share_directory,get_package_prefix
from classifier_core import Classifier,softmax,metrics,digest,require
from m3c_data import audit_splits

def installed_sources():
    p=Path(get_package_share_directory('sorting_perception'));lib=Path(get_package_prefix('sorting_perception'))/'lib/sorting_perception';d=Path(get_package_share_directory('sorting_description'))/'mujoco/static_grasp'
    files=[p/'config/m3c.yaml',p/'config/cameras_m3c.yaml',p/'config/m3b_localization.yaml']+[lib/n for n in ('classifier_core.py','classifier_state.py','fruit_classifier.py','m3c_data.py','camera_core.py','handle_localizer.py','motion_filter.py','camera_qos.py')]
    files+=[d/'panda_camera_m3c.xml']
    files+=[d/f'scene_camera_m3c_{f}.xml' for f in ('red_clear','green_clear','red_covered','green_covered')]
    files+=[d/f'props_camera_m3c_{f}.xml' for f in ('red_clear','green_clear','red_covered','green_covered')]
    files+=[Path(get_package_share_directory('sorting_control'))/'launch/panda_m3c.launch.py']
    return {str(f.name):digest(f) for f in files}
def assess(c,prob,y,conditions):
    clear=conditions=='wrist_clear';hidden=~clear;accepted=prob.max(axis=1)>=c['acceptance_confidence'];wrong=prob.argmax(axis=1)!=y
    report=dict(all=metrics(prob,y,c['ece_bins']),clear=metrics(prob[clear],y[clear],c['ece_bins']),hidden=metrics(prob[hidden],y[hidden],c['ece_bins']),
        clear_coverage=float(accepted[clear].mean()),hidden_coverage=float(accepted[hidden].mean()),
        accepted_error=float(wrong[accepted].mean()) if accepted.any() else 1.,
        view_confidence_gain=float(prob[clear].max(axis=1).mean()-prob[conditions=='overhead_hidden'].max(axis=1).mean()))
    checks=dict(clear_accuracy=report['clear']['accuracy']>=c['minimum_clear_accuracy'],clear_coverage=report['clear_coverage']>=c['minimum_clear_coverage'],
        accepted_error=report['accepted_error']<=c['maximum_accepted_error'],hidden_abstention=report['hidden_coverage']<=c['maximum_hidden_coverage'],
        calibration_ece=report['all']['ece']<=c['maximum_ece'],view_reveals_evidence=report['view_confidence_gain']>=c['minimum_view_confidence_gain'])
    return report,checks
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset-dir',type=Path,required=True);parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);a=parser.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(checkpoint='M3C',status='FAIL',live_ROS='NOT_RUN',full_M3_gate_passed=False,synthetic_classes=True)
    try:
        share=Path(get_package_share_directory('sorting_perception'));c=yaml.safe_load((share/'config/m3c.yaml').read_text());model=Classifier(a.model_dir,c)
        data=json.loads((a.dataset_dir/'dataset.json').read_text());audit=audit_splits(data['rows'],c)
        require(model.meta['dataset_manifest_sha256']==digest(a.dataset_dir/'dataset.json') and model.meta['features_sha256']==digest(a.dataset_dir/'features.npz'),'dataset_hash_changed')
        with np.load(a.dataset_dir/'features.npz',allow_pickle=False) as f:x=f['x'];y=f['y']
        selected=np.asarray([r['split']=='test' for r in data['rows']]);conditions=np.asarray([r['condition'] for r in data['rows']])[selected]
        logits=model.net.logits(x[selected]);p=softmax(logits,model.temperature);scores,checks=assess(c,p,y[selected],conditions)
        require(all(checks.values()),'Held-out checks failed: '+str([k for k,v in checks.items() if not v]))
        # Default live fixtures are not training frames. Exercise the same RGB crop path.
        import sys,cv2
        sys.path.insert(0,str(Path(get_package_prefix('sorting_control'))/'lib/sorting_control'))
        from native_camera_pickup import NativePlant
        from camera_core import localize
        from classifier_core import fruit_crop
        simulation=Path(get_package_prefix('sorting_simulation'));description=Path(get_package_share_directory('sorting_description'))/'mujoco/static_grasp'
        localization=yaml.safe_load((share/'config'/c['localization_config']).read_text());fixture_results={}
        for fixture in ('red_clear','green_clear','red_covered','green_covered'):
            plant=NativePlant(simulation/'lib/libnative_camera_plant.so',description/f'scene_camera_m3c_{fixture}.xml')
            try:
                plant.call('nc_step',2.,0.);rgb,k,p0,r0,capture=plant.render('wrist');handle=localize(rgb,k,p0,r0,localization)
                from classifier_core import fruit_crop
                crop,bbox=fruit_crop(rgb,k,p0,r0,handle['position'],c);value=model.infer(crop)
                cv2.imwrite(str(a.output_dir/(fixture+'-crop.png')),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
                expected=0 if fixture.startswith('red') else 1
                require((not value['accepted']) if fixture.endswith('covered') else (value['accepted'] and np.argmax(value['probabilities'])==expected),'Native default fixture classification failed: '+fixture)
                fixture_results[fixture]=value
            finally:plant.destroy()
        checks['default_live_fixture_rgb']='PASS'
        # Timing is inference-only; live report also checks capture age and message delivery.
        samples=[]
        for i in range(100):
            started=time.monotonic();softmax(model.net.logits(x[i:i+1]),model.temperature);samples.append(time.monotonic()-started)
        require(np.percentile(samples,95)<c['maximum_inference_wall_s'],'native_inference_budget_failed')
        simulation=Path(get_package_prefix('sorting_simulation'));description=Path(get_package_share_directory('sorting_description'))/'mujoco/static_grasp'
        subprocess.run([str(simulation/'lib/sorting_simulation/verify_camera_render'),str(description/'scene_camera_m3c_red_clear.xml'),str(description/'scene_camera_m3b.xml'),str(share/'config/m3b_localization.yaml'),str(a.output_dir/'render'), '--benchmark'],check=True)
        budget=yaml.safe_load((a.output_dir/'render/render-benchmark.yaml').read_text())['render_benchmark'];wrist=budget['views']['wrist']['p95_render_read_wall_s']
        require(wrist<1./15.,'M3C wrist render exceeds nominal period')
        os.environ.setdefault('MPLCONFIGDIR',str(a.output_dir.resolve()/'.mplconfig'))
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5,4));bins=[r for r in scores['all']['reliability'] if r['count']];ax.plot([0,1],[0,1],'--',color='gray');ax.plot([r['confidence'] for r in bins],[r['accuracy'] for r in bins],'o-');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Observed accuracy',title='M3C held-out reliability');fig.tight_layout();fig.savefig(a.output_dir/'reliability.png',dpi=160);plt.close(fig)
        report.update(status='PASS',classifier_ready=True,checks={k:'PASS' for k in checks},native_live_fixture_rgb=fixture_results,held_out=scores,uncalibrated=metrics(softmax(logits),y[selected],c['ece_bins']),
            maximum_marker_error_m=data['maximum_marker_error_m'],split_audit=audit,paired_view_audit=data['paired_view_audit'],model_sha256=model.sha,temperature=model.temperature,
            inference_p95_wall_s=float(np.percentile(samples,95)),wrist_render_p95_wall_s=wrist,installed_sources=installed_sources(),
            model_directory=str(a.model_dir.resolve()),dataset_directory=str(a.dataset_dir.resolve()),
            scope='Fixed wrist image classifier; offline grouped calibration/test; live and class-aware grasp integration pending')
    except Exception as error:report['error']=str(error)
    (a.output_dir/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
