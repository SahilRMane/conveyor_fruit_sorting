#!/usr/bin/env python3
"""Train and test a deadline-aware wrist view selector offline."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml
from ament_index_python.packages import get_package_prefix

from render_utility import FAMILIES, collect
from utility_core import RIDGE_PENALTY, TIME_COST_PER_S, VIEWS, WRONG_CLASS_COST, choose_view, fit, predict

sys.path.insert(0, str(Path(get_package_prefix('sorting_supervisor')) / 'lib/sorting_supervisor'))
from m4a_common import executable, share
from m4b_common import classifier_config, model_directory
from m4c_common import config as moving_config
from m4c_core import pick_reserve
from m4d_common import checked_preflight
from classifier_core import Classifier, digest, require


SPLITS = {'train': (256, 48), 'test': (304, 24)}
POLICIES = ('assess_only', 'fixed_left', 'learned')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def setup(args):
    gate = args.m4_batch_dir
    summary, manifest = read(gate / 'summary.json'), read(gate / 'manifest.json')
    require(summary['status'] == 'PASS' and summary['full_M4_gate_passed']
            and summary['successful_episodes'] == 10 and summary['failed_episodes'] == summary['not_run_episodes'] == 0
            and summary['manifest_sha256'] == digest(gate / 'manifest.json'), 'Complete M4 gate required')
    folder, frozen = checked_preflight(manifest['preflight_directory'])
    preflight_sha = digest(folder / 'preflight.json')
    require(preflight_sha == manifest['preflight_sha256']
            and frozen['installed_manifest'] == manifest['installed_manifest'], 'M4 preflight differs')

    m5a = args.m5a_dir
    report, split = read(m5a / 'report.json'), read(m5a / 'split-manifest.json')
    calibration = m5a / 'calibration-parameters.json'
    require(report['checkpoint'] == 'M5A' and report['status'] == 'PASS'
            and report['calibration_parameters_sha256'] == digest(calibration)
            and read(calibration)['split_manifest_sha256'] == digest(m5a / 'split-manifest.json')
            and split['m4_gate_sha256'] == digest(gate / 'summary.json')
            and split['preflight_sha256'] == preflight_sha
            and split['frozen_model_sha256'] == frozen['model_sha256'], 'M5A evidence differs from M4')
    native = folder / 'inspection/m4b/m4a'
    scene, poses = native / 'scene-red.xml', native / 'native/views.yaml'
    require(digest(scene) == split['scene_sha256'] and digest(poses) == split['poses_sha256'],
            'Native scene or poses differ from M5A')

    times = {view: [] for view in VIEWS}
    sources = {}
    for color, directory in (('red', args.red_probe_dir), ('green', args.green_probe_dir)):
        path = directory / 'summary.json'
        probe = read(path)
        require(probe['checkpoint'] == 'M5B' and probe['status'] == 'PASS'
                and probe['fixture'] == color + '_partial' and probe['arrival'] == 'upstream'
                and probe['wrist_samples'] == 10 and probe['overlap_violations'] == 0
                and probe['final_hold_status'] == probe['final_belt_stop_status'] == probe['overhead_timing']['status'] == 'PASS'
                and probe['final_owner'] == 'HOLD' and probe['preflight_sha256'] == preflight_sha
                and probe['model_sha256'] == frozen['model_sha256'], 'M5B evidence differs for ' + color)
        reference = probe['left_reference']
        ref_path = Path(reference['report'])
        require(ref_path.is_relative_to(gate) and reference['report_sha256'] == digest(ref_path)
                and reference['events_sha256'] == digest(ref_path.with_name('inspection-events.json')),
                'M4 left-view reference differs for ' + color)
        measured = (reference['costs']['total_sim_s'], probe['right_live']['costs']['total_sim_s'])
        require(all(math.isfinite(t) and 0 < t < 45 for t in measured), 'Invalid measured view duration')
        for view, value in zip(VIEWS, measured, strict=True):
            times[view].append(value)
        sources[color] = dict(report=str(path), sha256=digest(path))
    settings = moving_config()
    costs = {view: max(values) * settings['cost_scale'] + settings['cost_addition_s'] for view, values in times.items()}
    require(costs['inspect_right'] < costs['inspect_left'], 'Expected a separate right-only deadline window')
    reserve = pick_reserve(settings)
    deadlines = dict(roomy=60., right_only=(costs['inspect_left'] + costs['inspect_right']) / 2 + reserve, late=20.)
    require(reserve < deadlines['late'] < min(costs.values()) + reserve
            and max(costs.values()) + reserve < deadlines['roomy'], 'Deadline cases differ from the measured costs')
    return scene, poses, split, settings, times, costs, reserve, deadlines, sources, frozen


def payoff(probabilities, label, threshold):
    if max(probabilities) < threshold:
        return 0.
    return 1. if int(np.argmax(probabilities)) == label else -WRONG_CLASS_COST


def observe(rows, threshold):
    uncertain = [r for r in rows if max(r['views']['assess']) < threshold]
    x = np.array([r['features'] for r in uncertain])
    y = np.array([[payoff(r['views'][view], r['label'], threshold) for view in VIEWS] for r in uncertain])
    require(x.ndim == 2 and np.isfinite(x).all(), 'Invalid assess features')
    for index, family in enumerate(('left_reveal', 'right_reveal')):
        for label in (0, 1):
            require(any(r['family'] == family and r['label'] == label and p[index] == 1 and p[1 - index] == 0
                        for r, p in zip(uncertain, y, strict=True)),
                    'Split needs useful examples for both classes from ' + VIEWS[index])
    return uncertain, x, y


def evaluate(rows, model, threshold, costs, reserve, deadlines, output):
    report = {}
    with output.open('w') as stream:
        for window, remaining in deadlines.items():
            counts = {policy: dict(correct_accepts=0, wrong_accepts=0, abstentions=0, inspections=0,
                                   inspection_time_sim_s=0., chosen_views={view: 0 for view in VIEWS})
                      for policy in POLICIES}
            for row in rows:
                initial = row['views']['assess']
                scores = predict(model, row['features'])
                observed = {view: payoff(row['views'][view], row['label'], threshold) for view in VIEWS}
                decisions = {}
                for policy in POLICIES:
                    view = None
                    final = initial
                    if max(initial) < threshold:
                        if policy == 'fixed_left' and remaining > costs['inspect_left'] + reserve:
                            view = 'inspect_left'
                        elif policy == 'learned':
                            view = choose_view(scores, costs, remaining, reserve)
                        if view:
                            final = row['views'][view]
                            counts[policy]['inspections'] += 1
                            counts[policy]['inspection_time_sim_s'] += costs[view]
                            counts[policy]['chosen_views'][view] += 1
                    value = payoff(final, row['label'], threshold) if remaining > reserve else 0.
                    key = 'correct_accepts' if value > 0 else 'wrong_accepts' if value < 0 else 'abstentions'
                    counts[policy][key] += 1
                    decisions[policy] = dict(view=view, outcome=key)
                record = dict(group_id=row['group_id'], family=row['family'], label=row['label'], window=window,
                              remaining_s=remaining, assess_probabilities=initial, decisions=decisions,
                              candidates={view: dict(
                                  predicted_class_payoff=float(score), observed_class_payoff=observed[view],
                                  predicted_utility=float(score) - TIME_COST_PER_S * costs[view],
                                  observed_utility=observed[view] - TIME_COST_PER_S * costs[view],
                                  feasible=remaining > costs[view] + reserve)
                                  for view, score in zip(VIEWS, scores, strict=True)})
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            report[window] = counts
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4-batch-dir', type=Path, required=True)
    parser.add_argument('--m5a-dir', type=Path, required=True)
    parser.add_argument('--red-probe-dir', type=Path, required=True)
    parser.add_argument('--green-probe-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    for name in vars(args):
        setattr(args, name, getattr(args, name).resolve())
    scene, poses, old_split, settings, times, costs, reserve, deadlines, sources, frozen = setup(args)
    config = classifier_config()
    model_dir = model_directory()
    classifier = Classifier(model_dir, config)
    require(classifier.sha == frozen['model_sha256'], 'Frozen classifier changed')
    dataset = model_dir / 'dataset.json'
    require(digest(dataset) == classifier.meta['dataset_manifest_sha256'], 'Frozen classifier dataset changed')
    old_rows = read(dataset)['rows'] + [row for rows in old_split['groups'].values() for row in rows]
    previous_ids = {r['group_id'] for r in old_rows} | set(range(192, 200))
    previous_seeds = {r['appearance_instance'] for r in old_rows} | {
        config['dataset_seed'] + g * 7919 for g in range(192, 200)}
    partition = {}
    for name, (start, count) in SPLITS.items():
        ids = set(range(start, start + count))
        seeds = {config['dataset_seed'] + g * 7919 for g in ids}
        require(ids.isdisjoint(previous_ids) and seeds.isdisjoint(previous_seeds), 'Group or appearance seed overlap')
        previous_ids.update(ids)
        previous_seeds.update(seeds)
        partition[name] = dict(group_ids=sorted(ids), appearance_seeds=sorted(seeds))
    threshold = settings['acceptance_confidence']
    require(read(args.m5a_dir / 'report.json')['confidence_threshold'] == threshold,
            'M4 and M5A confidence thresholds differ')
    renderer = executable('sorting_supervisor', 'render_m4b_dataset')
    localization = yaml.safe_load((share('sorting_perception') / 'config/m3b_localization.yaml').read_text())
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'manifest.json', dict(splits=partition, families=FAMILIES, model_sha256=classifier.sha,
        source_hashes={name: digest(Path(__file__).with_name(name))
                       for name in ('train_m5c.py', 'render_utility.py', 'utility_core.py')},
        m4_gate_sha256=digest(args.m4_batch_dir / 'summary.json'), m5a_sha256=digest(args.m5a_dir / 'report.json'),
        m5b=sources, scene_sha256=digest(scene), poses_sha256=digest(poses), renderer_sha256=digest(renderer),
        ridge_penalty=RIDGE_PENALTY, wrong_class_cost=WRONG_CLASS_COST, time_cost_per_s=TIME_COST_PER_S,
        confidence_threshold=threshold, measured_times_sim_s=times, view_costs_sim_s=costs,
        pick_reserve_sim_s=reserve, deadline_windows_sim_s=deadlines))

    print('Render training groups and fit the selector.', flush=True)
    train, train_error = collect(out / 'train', *SPLITS['train'], scene, poses, renderer, config, localization, classifier)
    _, x, y = observe(train, threshold)
    learned = fit(x, y)
    model_path = out / 'view-utility.npz'
    np.savez_compressed(model_path, **learned)
    model_sha = digest(model_path)

    print('Render held-out groups and score the frozen selector.', flush=True)
    test, test_error = collect(out / 'test', *SPLITS['test'], scene, poses, renderer, config, localization, classifier)
    _, tx, ty = observe(test, threshold)
    require(digest(model_path) == model_sha, 'Fitted selector changed before test')
    with np.load(model_path, allow_pickle=False) as data:
        learned = {name: data[name] for name in data.files}
    errors = np.abs(predict(learned, tx) - ty)
    outcomes = evaluate(test, learned, threshold, costs, reserve, deadlines, out / 'held-out-decisions.jsonl')
    report = dict(checkpoint='M5C', status='PASS', full_M5_gate_passed=False, robot_commanded=False,
                  scope='Offline paired synthetic wrist views with modeled deadlines',
                  train_groups=SPLITS['train'][1], test_groups=SPLITS['test'][1], test_cases=len(test),
                  test_cases_per_window=len(test), uncertain_test_cases=len(tx), selector_sha256=model_sha,
                  test_used_for_training=False, manifest_sha256=digest(out / 'manifest.json'),
                  maximum_marker_error_m=max(train_error, test_error),
                  predicted_vs_observed_payoff_mae={view: float(errors[:, i].mean()) for i, view in enumerate(VIEWS)},
                  results=outcomes, output_directory=str(out))
    save(out / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('checkpoint', 'status', 'test_cases',
                                            'predicted_vs_observed_payoff_mae', 'full_M5_gate_passed',
                                            'output_directory')}, indent=2))
    for window, counts in outcomes.items():
        print(window, {p: (r['correct_accepts'], r['wrong_accepts'], r['abstentions'])
                       for p, r in counts.items()})


if __name__ == '__main__':
    main()
