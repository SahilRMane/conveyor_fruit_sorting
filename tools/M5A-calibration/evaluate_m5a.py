#!/usr/bin/env python3
"""Evaluate the frozen M4 classifier and a two-view fusion rule offline."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml
from ament_index_python.packages import get_package_prefix

sys.path.insert(0, str(Path(get_package_prefix('sorting_supervisor')) / 'lib/sorting_supervisor'))
from m4b_common import classifier_config, model_directory, policy_config
from m4a_common import executable, share
from m4d_common import checked_preflight
from m4b_data import CONDITIONS, POSES, audit, prepare
from classifier_core import Classifier, digest, fit_temperature, metrics, require, softmax


GROUPS_PER_SPLIT = 32
BINS = 10


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fresh_groups(original, model, config):
    old_groups = sorted({row['group_id'] for row in original})
    require(old_groups == model.meta['dataset_group_ids'], 'Frozen model group list differs from its dataset')
    first = old_groups[-1] + 1
    groups = {
        'calibration': list(range(first, first + GROUPS_PER_SPLIT)),
        'test': list(range(first + GROUPS_PER_SPLIT, first + 2 * GROUPS_PER_SPLIT)),
    }
    old_appearances = {row['appearance_instance'] for row in original}
    seeds = {group: config['dataset_seed'] + group * 7919 for ids in groups.values() for group in ids}
    require(len(set(seeds.values())) == len(seeds)
            and not old_appearances.intersection(seeds.values()), 'Appearance seeds overlap')
    return {name: [{'group_id': group, 'appearance_instance': seeds[group]} for group in ids]
            for name, ids in groups.items()}


def render_split(folder, name, split, scene, poses, config, localization, model):
    folder.mkdir()
    config_path = folder / 'render-config.yaml'
    config_path.write_text(yaml.safe_dump(dict(config, dataset_groups=len(split))))
    render = executable('sorting_supervisor', 'render_m4b_dataset')
    start = time.monotonic()
    with (folder / 'renderer.log').open('w') as log:
        subprocess.run([str(render), str(scene), str(poses), str(config_path), str(folder / 'raw'),
                        str(split[0]['group_id'])], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=240)
    elapsed = time.monotonic() - start
    renderer = next((line for line in (folder / 'renderer.log').read_text().splitlines()
                     if line.startswith('OpenGL renderer:')), '')
    require('NVIDIA' in renderer, 'Expected NVIDIA EGL renderer. See renderer.log.')
    data = prepare(folder / 'raw', folder, config, localization, smoke=True)
    rows = data['rows']
    expected = {(row['group_id'], row['appearance_instance']) for row in split}
    require({(row['group_id'], row['appearance_instance']) for row in rows} == expected,
            'Rendered groups differ from the declared split')
    for row in rows:
        row['split'] = name
    data['split_audit'] = audit(rows, {name: len(split)})
    save_json(folder / 'dataset.json', data)
    with np.load(folder / 'features.npz', allow_pickle=False) as features:
        x, y = features['x'], features['y']
    require(np.array_equal(y, [r['label'] for r in rows]), 'Feature labels differ from the image manifest')
    logits = model.net.logits(x)
    require(np.isfinite(logits).all(), 'Classifier produced non-finite logits')
    lookup = {(r['group_id'], r['condition'], r['label'], r['pose']): i for i, r in enumerate(rows)}
    pairs = [(row['group_id'], condition, label)
             for row in split for condition in CONDITIONS for label in range(len(config['class_ids']))]
    views = {pose: logits[[lookup[(*key, pose)] for key in pairs]] for pose in POSES}
    provenance = dict(renderer=renderer, render_wall_s=elapsed, images=len(rows),
                      dataset_sha256=digest(folder / 'dataset.json'),
                      features_sha256=digest(folder / 'features.npz'),
                      marker_error_max_m=data['maximum_marker_error_m'],
                      paired_view_audit=data['paired_view_audit'])
    return views, np.array([p[2] for p in pairs]), np.array([p[1] for p in pairs]), provenance


def decision_counts(scores, labels, conditions, threshold):
    accepted = scores.max(axis=1) >= threshold
    correct = scores.argmax(axis=1) == labels
    counts = {}
    for condition in (*CONDITIONS, 'all'):
        selected = conditions == condition if condition != 'all' else np.ones(len(labels), dtype=bool)
        count = int((selected & accepted).sum())
        wrong = int((selected & accepted & ~correct).sum())
        total = int(selected.sum())
        counts[condition] = dict(cases=total, accepted=count, correct_accepts=count - wrong,
                                 wrong_accepts=wrong, abstained=total - count,
                                 coverage=count / total, accepted_error_rate=wrong / count if count else None)
    return counts


def reliability_plot(path, views, inspection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout='constrained')
    panels = [(name, values['raw'], values['m4_calibrated'], 'Raw', 'M4 temperature')
              for name, values in views.items()]
    panels.append(('After inspection', inspection['m4_left'], inspection['fusion'], 'M4 left only', 'Fused'))
    for axis, (title, first, second, first_label, second_label) in zip(axes.flat, panels, strict=True):
        axis.plot([0, 1], [0, 1], color='0.6', linestyle='--')
        for data, label, marker in ((first, first_label, 'o'), (second, second_label, 's')):
            points = [row for row in data['reliability'] if row['count']]
            axis.scatter([p['confidence'] for p in points], [p['accuracy'] for p in points],
                         s=[25 + 120 * p['count'] / data['samples'] for p in points],
                         marker=marker, label=label, alpha=.75)
        axis.set(title=title.replace('_', ' '), xlim=(0, 1.04), ylim=(0, 1.04),
                 xlabel='Mean confidence', ylabel='Observed accuracy')
        axis.legend(loc='upper left', fontsize=8)
    fig.suptitle('M5A reliability on fresh synthetic groups')
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4-batch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    gate = args.m4_batch_dir.resolve()
    gate_summary = json.loads((gate / 'summary.json').read_text())
    gate_manifest = json.loads((gate / 'manifest.json').read_text())
    require(gate_summary['status'] == 'PASS' and gate_summary['full_M4_gate_passed']
            and gate_summary['planned_episodes'] == gate_summary['successful_episodes'] == 10
            and gate_summary['failed_episodes'] == gate_summary['not_run_episodes'] == 0
            and gate_summary['manifest_sha256'] == digest(gate / 'manifest.json'),
            'A complete passing M4 gate is required')
    preflight, frozen = checked_preflight(gate_manifest['preflight_directory'])
    require(digest(preflight / 'preflight.json') == gate_manifest['preflight_sha256']
            and frozen['installed_manifest'] == gate_manifest['installed_manifest'],
            'M4 preflight differs from the gate manifest')
    config = classifier_config()
    threshold = policy_config()['acceptance_confidence']
    model = Classifier(model_directory(), config)
    require(model.sha == frozen['model_sha256'], 'Classifier differs from the M4 gate')
    old_dataset = model_directory() / 'dataset.json'
    require(digest(old_dataset) == model.meta['dataset_manifest_sha256'], 'Frozen dataset hash differs')
    groups = fresh_groups(json.loads(old_dataset.read_text())['rows'], model, config)
    native = preflight / 'inspection/m4b/m4a'
    scene, poses = native / 'scene-red.xml', native / 'native/views.yaml'
    localization = yaml.safe_load((share('sorting_perception') / 'config/m3b_localization.yaml').read_text())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'split-manifest.json', dict(
        groups=groups, frozen_model_sha256=model.sha, frozen_dataset_sha256=digest(old_dataset),
        m4_gate_sha256=digest(gate / 'summary.json'), preflight_sha256=gate_manifest['preflight_sha256'],
        scene_sha256=digest(scene), poses_sha256=digest(poses), evaluator_sha256=digest(__file__),
        renderer_binary_sha256=digest(executable('sorting_supervisor', 'render_m4b_dataset'))))

    print('Render calibration groups', flush=True)
    cal, cal_y, _, cal_evidence = render_split(out / 'calibration', 'calibration', groups['calibration'],
                                               scene, poses, config, localization, model)
    cal_inspect = softmax(cal['assess'], model.temperature).max(axis=1) < threshold
    require(cal_inspect.any(), 'No calibration case requests inspection')
    fused_cal = (cal['assess'][cal_inspect] + cal['inspect_left'][cal_inspect]) / 2
    temperature = fit_temperature(fused_cal, cal_y[cal_inspect], config['temperature_bounds'])
    save_json(out / 'calibration-parameters.json', dict(
        fusion_rule='Mean raw logits from assess and inspect_left, then apply a separate temperature',
        m4_temperature=model.temperature, fusion_temperature=temperature, confidence_threshold=threshold,
        fitted_cases=int(cal_inspect.sum()), calibration_group_ids=[row['group_id'] for row in groups['calibration']],
        split_manifest_sha256=digest(out / 'split-manifest.json')))

    print('Render held-out test groups', flush=True)
    test, test_y, conditions, test_evidence = render_split(out / 'test', 'test', groups['test'],
                                                          scene, poses, config, localization, model)
    probs = {pose: softmax(values, model.temperature) for pose, values in test.items()}
    inspect = probs['assess'].max(axis=1) < threshold
    require(inspect.any(), 'No held-out case requests inspection')
    fused = softmax((test['assess'] + test['inspect_left']) / 2, temperature)
    view_metrics = {pose: dict(raw=metrics(softmax(test[pose]), test_y, BINS),
                               m4_calibrated=metrics(probs[pose], test_y, BINS)) for pose in POSES}
    inspection = dict(m4_left=metrics(probs['inspect_left'][inspect], test_y[inspect], BINS),
                      fusion=metrics(fused[inspect], test_y[inspect], BINS))
    policies = dict(fixed_assess=probs['assess'],
                    m4_one_view=np.where(inspect[:, None], probs['inspect_left'], probs['assess']),
                    candidate_fusion=np.where(inspect[:, None], fused, probs['assess']))
    decisions = {name: decision_counts(values, test_y, conditions, threshold) for name, values in policies.items()}
    reliability_plot(out / 'reliability.svg', view_metrics, inspection)
    report = dict(checkpoint='M5A', status='PASS', scope='Offline calibration study with synthetic scene variants',
                  test_cases=len(test_y), inspection_cases=int(inspect.sum()), confidence_threshold=threshold,
                  calibration_parameters_sha256=digest(out / 'calibration-parameters.json'),
                  ece_definition='Top-label ECE, ten equal-width bins, sample-count weighting',
                  brier_definition='Mean sum of squared class probability errors',
                  views=view_metrics, after_inspection=inspection, decisions=decisions,
                  calibration_render=cal_evidence, test_render=test_evidence,
                  full_M5_gate_passed=False, robot_commanded=False)
    save_json(out / 'report.json', report)
    print(json.dumps(dict(checkpoint='M5A', status='PASS', test_cases=len(test_y),
                          inspection_cases=int(inspect.sum()),
                          correct_accepts={name: value['all']['correct_accepts'] for name, value in decisions.items()},
                          wrong_accepts={name: value['all']['wrong_accepts'] for name, value in decisions.items()},
                          full_M5_gate_passed=False, report=str(out / 'report.json')), indent=2))


if __name__ == '__main__':
    main()
