#!/usr/bin/env python3
"""Check the seven latest M4C cases for the installed version and preflight."""
import argparse
import hashlib
import json
from pathlib import Path
from ros_paths import prepare, write_json

CASES = {
    ('red_partial', 'upstream'): ('red_fruit', 'inspect'),
    ('green_partial', 'upstream'): ('green_fruit', 'inspect'),
    ('red_clear', 'upstream'): ('red_fruit', 'direct'),
    ('green_clear', 'upstream'): ('green_fruit', 'direct'),
    ('red_covered', 'upstream'): ('red_fruit', 'unresolved'),
    ('green_covered', 'upstream'): ('green_fruit', 'unresolved'),
    ('red_clear', 'late'): ('red_fruit', 'deadline'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_case(summary, timing, expected_class, expected_path, config, model_sha):
    require(summary.get('checkpoint') == 'M4C', 'Report is not an M4C test')
    require(summary.get('status') == 'PASS' and summary.get('acceptance_passed') is True,
            'Functional acceptance failed')
    require(summary.get('final_hold_status') == 'PASS' and summary.get('final_owner') == 'HOLD'
            and summary.get('overlap_violations') == 0, 'Final exclusive HOLD failed')
    require(summary.get('model_sha256') == model_sha, 'Classifier model differs')
    require(summary.get('policy_path') == expected_path, 'Policy path differs')
    require(summary.get('physical_pickup_permitted') is False
            and summary.get('physical_grasp') == 'NOT_RUN', 'M4C scope differs')
    require(timing.get('functional_status') == 'PASS', 'Timing report belongs to a failed run')
    observed = timing['timing_diagnostic']['observed']
    require(observed.get('status') == 'PASS' and not observed.get('violation_counts')
            and observed.get('first_violation') is None, 'Observed timing failed')
    require(0 <= observed['maximum_delivery_age_s'] <= config['delivery_budget_s'],
            'Delivery age exceeds the unchanged budget')
    require(0 <= observed['maximum_capture_gap_s'] <= observed['budget']['maximum_capture_gap_s'],
            'Capture gap exceeds the unchanged budget')
    decision = summary['decision']
    if expected_path in ('direct', 'inspect'):
        require(decision.get('action') == 'PICK_READY'
                and decision.get('class_id') == expected_class
                and decision.get('proposed_bin') == config['bins'][expected_class], 'Pickup proposal differs')
    else:
        require(decision.get('action') == 'PASS', 'Covered or late target did not pass')
    return dict(status='PASS', policy_path=expected_path,
                maximum_delivery_age_s=observed['maximum_delivery_age_s'],
                maximum_capture_gap_s=observed['maximum_capture_gap_s'])


def verify(results_root, preflight_folder):
    prepare()
    from m4c_common import checked_preflight, config
    from m4b_common import model_directory
    folder, preflight = checked_preflight(preflight_folder)
    c = config()
    model_sha = digest(model_directory() / 'weights.npz')
    chosen, candidates = {}, []
    for path in sorted(Path(results_root).expanduser().resolve().glob('M4C-*/summary.json'),
                       key=lambda p: (p.stat().st_mtime_ns, str(p))):
        summary = json.loads(path.read_text())
        key = (summary.get('fixture'), summary.get('arrival'))
        if key not in CASES or summary.get('installed_manifest') != preflight['installed_manifest']:
            continue
        candidates.append(dict(summary=str(path), fixture=key[0], arrival=key[1], status=summary.get('status')))
        # Do not replace a newer failure with an older success.
        chosen[key] = (path, summary)
    rows = []
    for key, (expected_class, expected_path) in CASES.items():
        require(key in chosen, 'Missing current-version M4C case: ' + ':'.join(key))
        path, summary = chosen[key]
        timing_path = path.parent / 'timing-diagnosis.json'
        timing = json.loads(timing_path.read_text())
        grade = check_case(summary, timing, expected_class, expected_path, c, model_sha)
        rows.append(dict(fixture=key[0], arrival=key[1], summary=str(path),
                         summary_sha256=digest(path), timing=str(timing_path),
                         timing_sha256=digest(timing_path), **grade))
    return dict(checkpoint='M4C-aggregate', status='PASS', cases_passed=len(rows),
                full_M4_gate_passed=False, physical_pickup_permitted=False,
                preflight_directory=str(folder), preflight_sha256=digest(folder / 'preflight.json'),
                installed_manifest=preflight['installed_manifest'], model_sha256=model_sha,
                cases=rows, current_version_candidates=candidates)


def checked_gate(path, preflight_folder):
    prepare()
    from m4c_common import checked_preflight
    folder, current = checked_preflight(preflight_folder)
    gate = json.loads(Path(path).read_text())
    require(gate.get('checkpoint') == 'M4C-aggregate' and gate.get('status') == 'PASS'
            and gate.get('cases_passed') == 7, 'A passing seven-case M4C aggregate is required')
    require(gate['preflight_directory'] == str(folder)
            and gate['preflight_sha256'] == digest(folder / 'preflight.json')
            and gate['installed_manifest'] == current['installed_manifest'], 'Aggregate preflight differs')
    require({(v['fixture'], v['arrival']) for v in gate['cases']} == set(CASES), 'Aggregate case set differs')
    for row in gate['cases']:
        require(row['status'] == 'PASS' and digest(row['summary']) == row['summary_sha256']
                and digest(row['timing']) == row['timing_sha256'], 'Saved M4C evidence changed')
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--preflight-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file to preserve the previous evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = verify(args.results_root, args.preflight_dir)
    except Exception as error:
        report = dict(checkpoint='M4C-aggregate', status='FAIL', error=str(error), full_M4_gate_passed=False)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('installed_manifest', 'current_version_candidates')}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
