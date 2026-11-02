#!/usr/bin/env python3
"""Reproduce the M4B grouped RGB model. No ROS node or robot command."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import yaml
import numpy as np
from m4b_common import classifier_config, policy_config
from m4a_common import share, executable
from m4b_data import prepare, assess
from classifier_core import Network, Classifier, fit_temperature, config_digest, digest, nll


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m4a-preflight-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    c = classifier_config(); loc = yaml.safe_load((share('sorting_perception')/'config/m3b_localization.yaml').read_text())
    subprocess.run([str(executable('sorting_supervisor', 'render_m4b_dataset')),
                    str(a.m4a_preflight_dir.resolve()/'scene-red.xml'), str(a.m4a_preflight_dir.resolve()/'native/views.yaml'),
                    str(share('sorting_supervisor')/'config/m4b_classifier.yaml'), str(out/'raw'), '0'], check=True, timeout=240)
    d = prepare(out/'raw', out, c, loc)
    with np.load(out/'features.npz', allow_pickle=False) as f:
        x, y = f['x'], f['y']
    splits = np.array([r['split'] for r in d['rows']]); net = Network.initial(c)
    selection = net.train(x[splits == 'train'], y[splits == 'train'], (x[splits == 'validation'], y[splits == 'validation']), c)
    logits = net.logits(x[splits == 'calibration']); temperature = fit_temperature(logits, y[splits == 'calibration'], c['temperature_bounds'])
    model = out/'model'; model.mkdir()
    np.savez_compressed(model/'weights.npz', w1=net.w1, b1=net.b1, w2=net.w2, b2=net.b2)
    meta = dict(class_ids=c['class_ids'], temperature=temperature, config_sha256=config_digest(c), weights_sha256=digest(model/'weights.npz'),
                dataset_manifest_sha256=digest(out/'dataset.json'), features_sha256=digest(out/'features.npz'),
                split_audit=d['split_audit'], selection=selection,
                training_groups=list(range(c['split_group_counts']['train'])), dataset_group_ids=list(range(c['dataset_groups'])),
                test_used_for_training=False, test_used_for_calibration=False,
                calibration_nll_before=nll(logits, y[splits == 'calibration']), calibration_nll_after=nll(logits, y[splits == 'calibration'], temperature),
                scope='M4B synthetic class cues at three stationary wrist poses; same asset, grouped nuisance seeds')
    (model/'model.json').write_text(json.dumps(meta, indent=2)+'\n')
    frozen = Classifier(model, c)
    selected = splits == 'validation'
    validation = assess(frozen, x[selected], y[selected], [r for r in d['rows'] if r['split'] == 'validation'], policy_config())
    (model/'validation-report.json').write_text(json.dumps(validation,indent=2)+'\n')
    # Test is opened only after the development/validation gate passes.
    selected = splits == 'test'
    report = assess(frozen, x[selected], y[selected], [r for r in d['rows'] if r['split'] == 'test'], policy_config())
    report.update(split_audit=d['split_audit'], paired_view_audit=d['paired_view_audit'], model_sha256=frozen.sha,
                  test_used_for_selection=False, scope=meta['scope'])
    (model/'test-report.json').write_text(json.dumps(report, indent=2)+'\n')
    shutil.copy2(out/'dataset.json', model/'dataset.json')
    np.savez_compressed(model/'test-features.npz',x=x[selected],y=y[selected],row_indices=np.flatnonzero(selected))
    meta.update(test_features_sha256=digest(model/'test-features.npz'),generator_binary_sha256=digest(executable('sorting_supervisor','render_m4b_dataset')))
    (model/'model.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',model_directory=str(model),held_out_samples=report['samples'],ece=report['ece']),indent=2))


if __name__ == '__main__':
    main()
