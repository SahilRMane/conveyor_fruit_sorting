#!/usr/bin/env python3
"""Run the frozen learned selector on the M4 cases and the M5 right-reveal case."""

import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid

from run_episode import PROJECTS, load_case, inspection_targets, model_for, read, require, sha, write_json
from run_m5e import check_proposals
from batch_core import assess_episode, schedule
from run_seeded_m4 import cleanup, wait_empty_graph

HERE = Path(__file__).resolve().parent
VISIBLE = PROJECTS / 'M5D-live-selector/m5d.launch.py'


def check_pilot(path, side, suite_dir, suite):
    report = read(path)
    case = suite['cases'][side]
    physical = report['physical']
    require(report['checkpoint'] == 'M5E' and report['case'] == side
            and report['suite_sha256'] == sha(suite_dir / 'suite.json')
            and report['selector_sha256'] == suite['selector_sha256']
            and report['source_sha256'] == sha(PROJECTS / 'M5E-learned-sorting/run_m5e.py')
            and report['pose_bias_sha256'] == sha(PROJECTS / 'M5E-learned-sorting/pose-bias.json')
            and all(report[k] == 'PASS' for k in ('status', 'physical_grasp', 'placement_status', 'final_reset_status'))
            and report['selection']['view'] == case['expected_view']
            and physical['status'] == 'PASS' and physical['true_class'] == case['expected_class']
            and physical['selected_bin'] == ('bin_a' if side == 'left' else 'bin_b')
            and report['reset']['status'] == 'PASS'
            and report['final_owner'] == 'HOLD' and report['overlap_violations'] == 0,
            'M5E physical evidence differs: ' + side)


def manifest(args):
    suite, right = load_case(args.suite_dir, 'right')
    load_case(args.suite_dir, 'left')
    check_proposals(args.suite_dir, suite)
    model_for(suite)
    preflight = Path(suite['preflight'])
    root = read(preflight / 'preflight.json')
    installed = root['installed_manifest']
    require(installed['m4c/render_backend_request'] == 'nvidia'
            and os.environ.get('SORTING_RENDER_BACKEND') == 'nvidia', 'Use the checked NVIDIA render backend')
    for side, report in (('left', args.left_report), ('right', args.right_report)):
        check_pilot(report, side, args.suite_dir, suite)
    baseline_file = args.m4_batch_dir / 'manifest.json'
    baseline = read(baseline_file)
    summary_file = args.m4_batch_dir / 'summary.json'
    summary = read(summary_file)
    seed = baseline['base_seed']
    original = schedule(seed)
    require(summary['status'] == 'PASS' and summary['full_M4_gate_passed']
            and summary['manifest_sha256'] == sha(baseline_file)
            and baseline['preflight_sha256'] == suite['preflight_sha256']
            and baseline['preflight_directory'] == str(preflight)
            and baseline['installed_manifest'] == installed and baseline['cases'] == original,
            'M4 baseline does not match this preflight and schedule')
    require(len(summary['cases']) == len(original), 'M4 baseline episode count differs')
    references = []
    for case, outcome in zip(original, summary['cases'], strict=True):
        report = Path(outcome['report_path'])
        require(outcome['status'] == 'PASS' and sha(report) == outcome['report_sha256'],
                'M4 episode evidence differs')
        assess_episode(read(report), case, installed)
        references.append(dict(path=str(report), sha256=outcome['report_sha256']))
    scenes = read(preflight / 'inspection/preflight.json')['scene_paths']
    cases = []
    for case, reference in zip(original, references, strict=True):
        fixture, arrival = case['fixture'], case['arrival']
        scene = scenes[fixture + ':' + arrival]
        cases.append(dict(case, scene=scene, scene_sha256=sha(scene), expected_view=None,
                          baseline=reference))
    extra = random.Random(seed)
    index = extra.randrange(len(cases) + 1)
    cases.insert(index, dict(fixture='green_right_reveal', arrival=right['arrival'],
                            expected_action='PICK_PLACE', expected_path='inspect',
                            expected_class=right['expected_class'], expected_view=right['expected_view'],
                            scene=right['scene'], scene_sha256=sha(right['scene']),
                            scenario_selection_seed=extra.getrandbits(32), baseline=None))
    for index, case in enumerate(cases, start=1):
        case['index'] = index
        inspection_targets(suite, Path(case['scene']), case['arrival'])
    evidence = [args.suite_dir / 'suite.json', baseline_file, summary_file, args.left_report, args.right_report]
    sources = [*sorted(HERE.glob('*.py')), VISIBLE,
               PROJECTS / 'M5E-learned-sorting/run_m5e.py', PROJECTS / 'M5E-learned-sorting/pose-bias.json',
               PROJECTS / 'M4D-seeded-batch-v2/batch_core.py', PROJECTS / 'M4D-seeded-batch-v2/run_seeded_m4.py']
    return dict(checkpoint='M5F-seeded-selector', base_seed=seed, cases=cases, headless=False,
                preflight_directory=str(preflight), selector_sha256=suite['selector_sha256'],
                suite_sha256=sha(args.suite_dir / 'suite.json'), installed_manifest=installed,
                inputs={str(p): sha(p) for p in [*evidence, *sources]},
                seed_scope='Pinned scene order and right-reveal insertion')


def check_episode(report, case, frozen):
    require(report['checkpoint'] == 'M5F-seeded-episode'
            and report['selector_sha256'] == frozen['selector_sha256']
            and report['suite_sha256'] == frozen['suite_sha256']
            and report['scene'] == case['scene'] and report['scene_sha256'] == case['scene_sha256'],
            'Learned episode source or scene differs')
    # A skipped sort must still pass every physical exit and reset check.
    missed = case['expected_action'] == 'PICK_PLACE' and report['action'] == 'PASS'
    observed = dict(case, expected_action='PASS', expected_path=report['policy_path']) if missed else case
    assess_episode(dict(report, checkpoint='M4-seeded-episode'), observed, frozen['installed_manifest'])
    if not missed:
        require(case['expected_view'] is None or report['selected_view'] == case['expected_view'],
                'Learned episode did not use the expected view')
    return missed


def run_case(args, case, destination, frozen):
    folder = destination / f"episode-{case['index']:02d}-{case['fixture']}-{case['arrival']}"
    folder.mkdir()
    write_json(folder / 'case.json', case)
    token = uuid.uuid4().hex
    env = dict(os.environ, SORTING_M4_BATCH_TOKEN=token, ROS_LOG_DIR=str(folder / 'ros-logs'))
    launch = episode = None
    row = dict(case, status='FAIL')
    try:
        wait_empty_graph()
        if case['fixture'] == 'green_right_reveal':
            launch_args = [str(VISIBLE), f'suite_dir:={args.suite_dir}', 'case:=right']
        else:
            launch_args = ['sorting_supervisor', 'panda_m4d.launch.py',
                           f"preflight_dir:={frozen['preflight_directory']}", f"fixture:={case['fixture']}",
                           f"arrival:={case['arrival']}", 'headless:=false']
        with (folder / 'launch.log').open('w') as log, (folder / 'episode.log').open('w') as trace:
            launch = subprocess.Popen(['ros2', 'launch', *launch_args], stdout=log, stderr=subprocess.STDOUT,
                                      env=env, start_new_session=True)
            episode = subprocess.Popen([sys.executable, str(HERE / 'run_episode.py'),
                '--suite-dir', str(args.suite_dir), '--case-file', str(folder / 'case.json'),
                '--output-dir', str(folder / 'result')], stdout=trace, stderr=subprocess.STDOUT,
                env=env, start_new_session=True)
            end = time.monotonic() + 480.
            while episode.poll() is None:
                require(launch.poll() is None, 'Simulation launch exited. Read launch.log.')
                require(time.monotonic() < end, 'Episode exceeded its 480-second process bound')
                time.sleep(.2)
        path = folder / 'result/summary.json'
        require(path.is_file(), 'Episode produced no summary. Read episode.log.')
        report = read(path)
        row.update(report_path=str(path), report_sha256=sha(path))
        require(episode.returncode == 0, report.get('error', 'Episode failed'))
        missed = check_episode(report, case, frozen)
        row.update(status='FAIL' if missed else 'PASS', action=report['action'], policy_path=report['policy_path'],
                   selected_view=report['selected_view'], selected_bin=(report.get('physical') or {}).get('selected_bin'),
                   final_reset_status=report['final_reset_status'], final_hold_status=report['final_hold_status'],
                   overlap_violations=report['overlap_violations'])
        if missed:
            row.update(safe_policy_miss=True, error='The selector safely passed a tray scheduled for sorting')
    except (Exception, KeyboardInterrupt) as error:
        row['error'] = str(error) or 'Interrupted'
        if episode is not None and episode.poll() is None:
            try:
                os.killpg(episode.pid, signal.SIGINT)
                episode.wait(timeout=15.)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
    finally:
        try:
            row['teardown'] = cleanup([episode, launch], token)
        except Exception as error:
            row.update(status='FAIL', safe_policy_miss=False, error=str(error))
        write_json(folder / 'outcome.json', row)
    return row


def run(args, frozen, out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'manifest.json', frozen)
    manifest_hash = sha(out / 'manifest.json')
    rows = [dict(case, status='NOT_RUN') for case in frozen['cases']]
    started = time.monotonic()
    error = None
    try:
        for i, case in enumerate(frozen['cases']):
            require(sha(out / 'manifest.json') == manifest_hash
                    and all(sha(path) == digest for path, digest in frozen['inputs'].items()),
                    'Batch code or input evidence changed')
            rows[i]['status'] = 'RUNNING'
            write_json(out / 'progress.json', dict(status='RUNNING', cases=rows))
            print(f"Run {i + 1}/{len(rows)}, case {case['index']}: {case['fixture']} {case['arrival']}", flush=True)
            rows[i] = run_case(args, case, out, frozen)
            print(f"Episode {case['index']}: {rows[i]['status']}", flush=True)
            if rows[i]['status'] != 'PASS' and not rows[i].get('safe_policy_miss'):
                break
        require(sha(out / 'manifest.json') == manifest_hash
                and all(sha(path) == digest for path, digest in frozen['inputs'].items()),
                'Batch code or input evidence changed')
    except (Exception, KeyboardInterrupt) as failure:
        error = str(failure) or 'Interrupted'
        for row in rows:
            if row['status'] == 'RUNNING':
                row.update(status='FAIL', error=error)
    counts = {state: sum(row['status'] == state for row in rows) for state in ('PASS', 'FAIL', 'NOT_RUN')}
    passed = counts['PASS'] == len(rows) and error is None
    result = dict(checkpoint=frozen['checkpoint'], status='PASS' if passed else 'FAIL',
                  learned_seeded_batch_passed=passed and len(rows) == 11, base_seed=frozen['base_seed'],
                  planned_episodes=len(rows), successful_episodes=counts['PASS'], failed_episodes=counts['FAIL'],
                  not_run_episodes=counts['NOT_RUN'], sorted_trays=sum(row.get('action') == 'PICK_PLACE' for row in rows),
                  exited_trays=sum(row.get('action') == 'PASS' for row in rows),
                  elapsed_wall_s=time.monotonic() - started, output_directory=str(out),
                  manifest_sha256=manifest_hash, cases=rows)
    result['safe_policy_misses'] = sum(row.get('safe_policy_miss', False) for row in rows)
    if error is not None:
        result['error'] = error
    write_json(out / 'summary.json', result)
    write_json(out / 'progress.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--suite-dir', '--m4-batch-dir', '--left-report', '--right-report'):
        parser.add_argument(name, type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument('--output-dir', type=lambda path: Path(path).resolve())
    parser.add_argument('--case-index', type=int, help='Run one numbered case as a probe.')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    frozen = manifest(args)
    if args.case_index is not None:
        require(1 <= args.case_index <= len(frozen['cases']), 'Case index is outside the schedule')
        frozen['cases'] = [frozen['cases'][args.case_index - 1]]
    if args.check:
        print(json.dumps(dict(status='PASS', robot_commanded=False, seed=frozen['base_seed'],
                              episodes=[(c['index'], c['fixture'], c['arrival']) for c in frozen['cases']]), indent=2))
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required for execution')
    require(os.environ.get('DISPLAY'), 'Run Terminal 1 in the graphical desktop session')
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    require(1 <= domain <= 232, 'Set an explicit ROS_DOMAIN_ID from 1 to 232')
    frozen['ros_domain_id'] = domain
    return run(args, frozen, args.output_dir)


if __name__ == '__main__':
    raise SystemExit(main())
