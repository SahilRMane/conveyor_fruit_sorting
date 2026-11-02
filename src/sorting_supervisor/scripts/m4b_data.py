"""Grouped image preparation and per-view acceptance; no ROS node."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
import yaml
# Resolve the installed shared perception utilities before importing them.
from m4a_common import share
from classifier_core import features, fruit_crop, require, config_digest, metrics
from camera_core import localize

POSES = ('assess', 'inspect_left', 'inspect_right')
CONDITIONS = ('clear', 'partial', 'covered')


def audit(rows, counts=None):
    groups, appearances = {}, {}
    for row in rows:
        groups.setdefault(row['group_id'], []).append(row)
        appearances.setdefault(row['appearance_instance'], set()).add(row['split'])
    require(bool(groups), 'empty_groups')
    expected = {(p, c, label) for p in POSES for c in CONDITIONS for label in (0, 1)}
    for group in groups.values():
        require(len(group) == 18 and {(r['pose'], r['condition'], r['label']) for r in group} == expected,
                'incomplete_counterfactual_group')
        require(len({r['split'] for r in group}) == 1, 'group_split_leakage')
    require(all(len(s) == 1 for s in appearances.values()), 'appearance_split_leakage')
    observed = {s: sum(g[0]['split'] == s for g in groups.values()) for s in {r['split'] for r in rows}}
    if counts is not None:
        require(observed == counts, 'group_counts_changed')
    return dict(groups=len(groups), group_counts=observed, groups_disjoint=True, procedural_seeds_disjoint=True,
                matched_class_view_groups=True, assets='Same synthetic asset; held-out nuisance seeds')


def prepare(raw, out, config, localization, smoke=False):
    rows = yaml.safe_load((raw/'frames.yaml').read_text())['frames']
    split_by_group = {}
    start = 0
    for split, count in config['split_group_counts'].items():
        split_by_group.update({g: split for g in range(start, start+count)})
        start += count
    xs, ys, pairs, errors = [], [], {}, []
    for row in rows:
        row['split'] = 'smoke' if smoke else split_by_group[row['group_id']]
        rgb = cv2.cvtColor(cv2.imread(str(raw/row['image'])), cv2.COLOR_BGR2RGB)
        f = row['height']/2/np.tan(np.deg2rad(row['fovy_deg'])/2)
        k = np.array([[f, 0, row['width']/2], [0, f, row['height']/2], [0, 0, 1.]])
        rotation = np.asarray(row['camera_optical_rotation']).reshape(3, 3)
        found = localize(rgb, k, row['camera_xyz'], rotation, localization)
        error = float(np.linalg.norm(np.asarray(found['position'][:2])-row['evaluator_handle_xyz'][:2]))
        require(error < localization['maximum_planar_error_m'], 'dataset_localization_exceeds_2mm')
        crop, box = fruit_crop(rgb, k, row['camera_xyz'], rotation, found['position'], config)
        row.update(crop_xyxy=box, position_source='neutral_RGB_marker')
        xs.append(features(crop, config)); ys.append(row['label']); errors.append(error)
        pairs[(row['group_id'], row['pose'], row['condition'], row['label'])] = crop
    split_audit = audit(rows, None if smoke else config['split_group_counts'])
    differences = {}
    for group in sorted({r['group_id'] for r in rows}):
        for pose in POSES:
            for condition in CONDITIONS:
                a, b = [pairs[(group, pose, condition, label)] for label in (0, 1)]
                require(a.shape == b.shape, 'class_changed_crop_geometry')
                delta = float(np.abs(a.astype(float)-b).mean()/255.)
                differences.setdefault(pose+'/'+condition, []).append(delta)
    paired = {key: dict(mean_rgb_difference=float(np.mean(v)), maximum_rgb_difference=max(v))
              for key, v in differences.items()}
    for key, value in paired.items():
        visible = key.endswith('/clear') or key == 'inspect_left/partial'
        require(value['mean_rgb_difference'] > .001 if visible else value['maximum_rgb_difference'] < .001,
                'view_observability_changed: '+key)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'features.npz', x=np.asarray(xs, np.float32), y=np.asarray(ys, np.int64))
    result = dict(rows=rows, split_audit=split_audit, paired_view_audit=paired,
                  config_sha256=config_digest(config), maximum_marker_error_m=max(errors))
    (out/'dataset.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def assess(model, x, y, rows, gate):
    from classifier_core import softmax
    started = time.perf_counter(); probabilities = softmax(model.net.logits(x), model.temperature)
    elapsed = time.perf_counter()-started
    confidence = probabilities.max(axis=1); predicted = probabilities.argmax(axis=1)
    accepted = confidence >= gate['acceptance_confidence']
    checks = {}
    for pose in POSES:
        for condition in CONDITIONS:
            selected = np.array([r['pose'] == pose and r['condition'] == condition for r in rows])
            require(selected.any(), 'missing_view_case')
            accept = selected & accepted; visible = condition == 'clear' or (condition == 'partial' and pose == 'inspect_left')
            coverage = float(accepted[selected].mean())
            accuracy = float((predicted[selected] == y[selected]).mean())
            error = float((predicted[accept] != y[accept]).mean()) if accept.any() else None
            ok = (accuracy >= gate['minimum_visible_accuracy'] and coverage >= gate['minimum_visible_coverage']
                  and (error is None or error <= gate['maximum_accepted_error'])) if visible else coverage <= gate['maximum_hidden_coverage']
            checks[pose+'/'+condition] = dict(status='PASS' if ok else 'FAIL', samples=int(selected.sum()),
                                            coverage=coverage, accuracy=accuracy if visible else None,
                                            accepted_error=error, observable=visible)
    calibration = metrics(probabilities, y, 10)
    require(all(r['status'] == 'PASS' for r in checks.values()), 'per_view_classifier_gate_failed: '+json.dumps(checks))
    require(calibration['ece'] <= gate['maximum_ece'], 'classifier_ECE_exceeded')
    times = []
    # Time the production crop inference, not the batched matrix multiply.
    for row in x[:64]:
        from classifier_core import softmax
        begin = time.perf_counter(); softmax(model.net.logits(row[None, :]), model.temperature)
        times.append(time.perf_counter()-begin)
    p95 = float(np.percentile(times, 95)); require(p95 < gate['maximum_inference_p95_wall_s'], 'classifier_inference_budget')
    return dict(status='PASS', cases=checks, ece=calibration['ece'], batched_wall_s=elapsed,
                inference_p95_wall_s=p95, samples=len(y), confidence_threshold=gate['acceptance_confidence'])


def policy_audit(model, x, rows, gate):
    from classifier_core import softmax
    from m4b_core import SingleViewPolicy
    scores=softmax(model.net.logits(x),model.temperature)
    lookup={(r['group_id'],r['condition'],r['label'],r['pose']):i for i,r in enumerate(rows)}
    counts=dict(PICK_READY=0,PASS=0,INSPECT=0); cases=0
    for group in sorted({r['group_id'] for r in rows}):
        for condition in CONDITIONS:
            for label in (0,1):
                policy=SingleViewPolicy(group,0,gate)
                p=scores[lookup[group,condition,label,'assess']]
                decision=policy.observe(episode=group,pose='assess',capture_ns=50_000_000,now_ns=100_000_000,probabilities=p)
                if decision['action']=='INSPECT':
                    counts['INSPECT']+=1;policy.start_inspection(200_000_000)
                    p=scores[lookup[group,condition,label,gate['inspection_view']]]
                    decision=policy.observe(episode=group,pose=gate['inspection_view'],capture_ns=300_000_000,now_ns=350_000_000,probabilities=p)
                expected='PASS' if condition=='covered' else 'PICK_READY'
                require(decision['action']==expected,'native_policy_terminal_mismatch')
                require(policy.inspections==(0 if condition=='clear' else 1),'native_policy_inspection_count')
                if expected=='PICK_READY':require(decision['class_id']==gate['class_ids'][label],'native_policy_wrong_class')
                counts[decision['action']]+=1;cases+=1
    return dict(status='PASS',cases=cases,decisions=counts,physical_grasps=0,
                scope='Native RGB features to frozen classifier to proposal logic; not a live robot episode')
