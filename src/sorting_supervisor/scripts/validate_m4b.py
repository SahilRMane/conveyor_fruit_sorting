#!/usr/bin/env python3
"""Native M4A regression plus frozen-model M4B view checks; no ROS node."""
import argparse
import json
from pathlib import Path
import subprocess
import yaml
import numpy as np
from m4b_common import classifier_config, policy_config, model_directory, scenes, generated_hashes, installed_manifest
from m4a_common import share, executable
from classifier_core import Classifier, require, digest
from m4b_data import prepare, assess, audit, policy_audit


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); out = a.output_dir.expanduser().resolve()
    if out.exists(): p.error('Use a new output directory to preserve previous evidence')
    out.mkdir(parents=True)
    r = dict(checkpoint='M4B', status='FAIL', live_ROS='NOT_RUN', full_M4_gate_passed=False,
             moving_conveyor=False, physical_grasp='NOT_RUN', output_directory=str(out))
    try:
        with (out/'m4a.log').open('w') as log:
            subprocess.run([str(executable('sorting_supervisor', 'validate_m4a.py')), '--output-dir', str(out/'m4a')],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        scene_paths = scenes(out)
        frame_checks = {}
        with (out/'fixture-frames.log').open('w') as log:
            for fixture, scene_path in scene_paths.items():
                target=out/(fixture+'-frames.json')
                subprocess.run([str(executable('sorting_control','verify_camera_model')),str(out/'m4a/panda.urdf'),
                                str(out/'m4a/panda.srdf'),scene_path,str(out/'m4a/frames.yaml'),str(target)],
                               check=True,stdout=log,stderr=subprocess.STDOUT,timeout=20)
                frame_checks[fixture]=json.loads(target.read_text())
        c = classifier_config(); gate=policy_config()
        require(gate['class_ids']==c['class_ids'] and gate['acceptance_confidence']==c['acceptance_confidence'],
                'Policy/classifier class order or confidence threshold differs')
        sorting=yaml.safe_load((share('sorting_control')/'config/m3d.yaml').read_text())
        require(gate['bins']==sorting['class_to_bin'], 'Proposed bins differ from the passed M3D routing rules')
        smoke = dict(c, dataset_groups=8)
        (out/'smoke.yaml').write_text(yaml.safe_dump(smoke))
        with (out/'dataset.log').open('w') as log:
            subprocess.run([str(executable('sorting_supervisor', 'render_m4b_dataset')), scene_paths['red_partial'],
                            str(out/'m4a/native/views.yaml'), str(out/'smoke.yaml'), str(out/'raw'), '8000'],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        loc = yaml.safe_load((share('sorting_perception')/'config/m3b_localization.yaml').read_text())
        data = prepare(out/'raw', out/'smoke', c, loc, smoke=True)
        model = Classifier(model_directory(), c)
        require(digest(model_directory()/'dataset.json') == model.meta['dataset_manifest_sha256'], 'Frozen dataset manifest differs')
        training_data = json.loads((model_directory()/'dataset.json').read_text())
        audit(training_data['rows'],c['split_group_counts'])
        require(digest(model_directory()/'test-features.npz') == model.meta['test_features_sha256'], 'Held-out feature hash differs')
        with np.load(model_directory()/'test-features.npz',allow_pickle=False) as frozen:
            held_out_rows=[training_data['rows'][int(i)] for i in frozen['row_indices']]
            require(all(row['split']=='test' for row in held_out_rows), 'Non-test data in frozen test gate')
            held_out=assess(model,frozen['x'],frozen['y'],held_out_rows,policy_config())
            held_out_policy=policy_audit(model,frozen['x'],held_out_rows,policy_config())
        require(set(model.meta['dataset_group_ids']).isdisjoint(r['group_id'] for r in data['rows']), 'smoke_training_group_overlap')
        with np.load(out/'smoke/features.npz', allow_pickle=False) as f:
            checks = assess(model, f['x'], f['y'], data['rows'], policy_config())
            smoke_policy=policy_audit(model,f['x'],data['rows'],policy_config())
        frozen_test = json.loads((model_directory()/'test-report.json').read_text())
        require(frozen_test['status'] == 'PASS' and frozen_test['model_sha256'] == model.sha, 'Frozen held-out report differs')
        r.update(status='PASS', installed_manifest=installed_manifest(), generated_hashes=generated_hashes(out),
                 scene_paths=scene_paths, classifier_ready=True, classifier_new_views_validated=True,
                 frozen_test=frozen_test, fresh_seed_smoke=checks, split_audit=data['split_audit'],
                 held_out_replay=held_out,
                 fixture_frames=frame_checks,
                 held_out_policy=held_out_policy,fresh_seed_policy=smoke_policy,
                 physical_pickup_permitted=False, scope='Stationary one-extra-view class proposals only; moving-belt integration pending')
    except Exception as error:
        r['error'] = str(error)
    (out/'preflight.json').write_text(json.dumps(r, indent=2)+'\n')
    summary={k:r[k] for k in ('checkpoint','status','live_ROS','full_M4_gate_passed','moving_conveyor','physical_grasp',
                             'classifier_new_views_validated','physical_pickup_permitted','output_directory','error') if k in r}
    if r['status']=='PASS':summary.update(held_out_samples=held_out['samples'],fresh_seed_samples=checks['samples'],checked_fixtures=len(frame_checks))
    print(json.dumps(summary,indent=2))
    raise SystemExit(0 if r['status'] == 'PASS' else 1)


if __name__ == '__main__': main()
