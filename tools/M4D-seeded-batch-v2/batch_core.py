"""Seeded scenario selection and strict M4 evidence checks. No robot commands."""
import hashlib
import json
import math
from pathlib import Path
import random


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def schedule(seed, profile='gate'):
    require(type(seed) is int and 0 <= seed < 2**32, 'Seed must be an unsigned 32-bit integer')
    require(profile in ('gate', 'smoke'), 'Unknown batch profile')
    pool = []
    for color in ('red', 'green'):
        for condition, arrival, action, path in (
            ('clear','upstream','PICK_PLACE','direct'),
            ('partial','upstream','PICK_PLACE','inspect'),
            ('covered','upstream','PASS','unresolved'),
            ('clear','late','PASS','deadline'),
            ('partial','late','PASS','deadline')):
            pool.append(dict(fixture=color+'_'+condition, arrival=arrival,
                             expected_action=action, expected_path=path, expected_class=color+'_fruit'))
    if profile == 'smoke':
        pool = [c for c in pool if (c['fixture'],c['arrival']) in
                (('red_clear','late'),('green_clear','upstream'))]
    source = random.Random(seed)
    result = []
    while pool:
        case_seed = source.getrandbits(32)
        index = random.Random(case_seed).randrange(len(pool))
        case = pool.pop(index)
        result.append(dict(case, index=len(result)+1, scenario_selection_seed=case_seed))
    return result


def source_hashes(folder):
    return {name:digest(Path(folder)/name) for name in
            ('batch_core.py','batch_inspection.py','run_m4_episode.py','run_seeded_m4.py','watch_m4.py')}


def check_prerequisite(report, color, installed):
    require(report.get('checkpoint')=='M4D.2' and report.get('status')=='PASS', 'Partial pilot did not pass')
    require(report.get('fixture')==color+'_partial' and report.get('policy_path')=='inspect', 'Wrong prerequisite path')
    require(report.get('acceptance_passed') is True and report.get('overlap_violations')==0, 'Partial pilot acceptance differs')
    require(all(report.get(k)=='PASS' for k in ('physical_grasp','placement_status','final_reset_status','final_hold_status')),
            'Incomplete partial physical prerequisite')
    require((report.get('image_evidence') or {}).get('status')=='PASS', 'Partial pilot image evidence is missing')
    require(report.get('installed_manifest')==installed, 'Partial pilot source differs from current preflight')


def assess_episode(report, case, installed):
    require(report.get('checkpoint')=='M4-seeded-episode', 'Wrong episode runner')
    require(report.get('status')=='PASS' and report.get('acceptance_passed') is True,
            report.get('error','Episode did not pass'))
    require(report.get('installed_manifest')==installed, 'Episode source differs from batch preflight')
    require(report.get('scenario_selection_seed')==case['scenario_selection_seed'] and report.get('case_index')==case['index'], 'Episode seed or index differs')
    require(all((report.get(k) or {}).get('status')=='PASS' for k in ('runtime_configuration','planning_configuration')), 'Runtime or planning configuration failed')
    require(report.get('fixture')==case['fixture'] and report.get('arrival')==case['arrival'], 'Episode scene differs')
    require(report.get('action')==case['expected_action'] and report.get('policy_path')==case['expected_path'], 'Policy path differs')
    require(report.get('final_reset_status')=='PASS' and report.get('final_hold_status')=='PASS'
            and report.get('final_owner')=='HOLD' and report.get('overlap_violations')==0, 'Reset or final ownership failed')
    require((report.get('timing') or {}).get('status')=='PASS', 'Overhead timing failed')
    require((report.get('image_evidence') or {}).get('status')=='PASS', 'Image evidence failed')
    if case['expected_action']=='PICK_PLACE':
        require(report.get('physical_grasp')=='PASS' and report.get('placement_status')=='PASS', 'Physical pickup/placement failed')
        require((report.get('wrist_timing') or {}).get('status')=='PASS', 'Wrist tracking timing failed')
        require((report.get('physical') or {}).get('selected_bin')==('bin_a' if case['expected_class']=='red_fruit' else 'bin_b'),
                'Wrong physical destination')
    else:
        require(report.get('physical_grasp')=='NOT_RUN' and report.get('placement_status')=='NOT_RUN', 'PASS performed a pickup')
        proof = report.get('pass_physical') or {}
        require(proof.get('status')=='PASS' and proof.get('exit_verified') is True and proof.get('no_arm_commands_after_decision') is True,
                'PASS did not verify supported exit and exclusive HOLD')


def pass_sample(value, initial_z):
    numbers = value['tray_xyz']+value['finger_positions_m']+[value['support_force_N'],value['left_force_N'],value['right_force_N']]
    require(all(math.isfinite(float(v)) for v in numbers), 'Non-finite PASS physical sample')
    require(value['forbidden_contacts']==0 and value['left_force_N']+value['right_force_N']<.1,
            'PASS touched the tray')
    require(min(value['finger_positions_m'])>.039 and value['support_force_N']>4., 'PASS lost open fingers or tray support')
    require(abs(value['tray_xyz'][2]-initial_z)<.003 and sum(value['bin_floor_force_N'])<.1, 'PASS lifted or placed the tray')


def pass_runout_s(target, config):
    lower_x=target['predicted_x_m']-config['uncertainty_sigma']*target['position_sigma_m']
    # Add clearance beyond the exit to the conservative camera position.
    duration=max(0.,config['exit_x_m']+.004-lower_x)/config['belt_speed_m_s']
    require(math.isfinite(duration) and duration<=350., 'PASS runout exceeds its simulation time bound')
    return duration


def verify_pass(history, transport, *, start_ns, decision_ns, end_ns, initial_z, exit_x, speed, before_stats, after_stats):
    samples = [v for v in history if start_ns <= round(v['sim_time']*1e9) <= end_ns]
    require(len(samples)>=20, 'Too few PASS physical samples')
    for value in samples:
        pass_sample(value,initial_z)
    final = max(samples,key=lambda v:v['sim_time'])
    require(final['tray_xyz'][0]>=exit_x, 'PASS tray did not cross the interception exit')
    for key in ('planned_accepted','servo_input_accepted','servo_output_forwarded'):
        require(key in before_stats and after_stats.get(key)==before_stats[key], 'Arm commands continued after PASS: '+key)
    rows = [v for v in transport if start_ns+1_000_000_000<=round(v['sim_time']*1e9)<=end_ns]
    require(len(rows)>=10, 'Too few PASS conveyor samples')
    for row in rows:
        require(row['command_fresh'] and abs(row['requested_speed_m_s']-speed)<1e-9, 'PASS stopped the conveyor before exit')
        require(abs(row['force_x_N'])<=8.000001 and abs(row['tray_vx_m_s']-speed)<.0015, 'PASS transport force or speed failed')
    return dict(status='PASS',exit_verified=True,no_arm_commands_after_decision=True,
                samples=len(samples),transport_samples=len(rows),decision_ns=decision_ns,end_ns=end_ns,
                exit_x_m=exit_x,final_tray_x_m=final['tray_xyz'][0])


def aggregate(manifest, rows):
    cases = manifest['cases']
    require(len(rows)==len(cases), 'Batch accounting differs from manifest')
    passed = sum(row['status']=='PASS' for row in rows)
    failed = sum(row['status']=='FAIL' for row in rows)
    not_run = sum(row['status']=='NOT_RUN' for row in rows)
    complete = passed == len(cases) and not failed and not not_run
    if complete and manifest['profile']=='gate':
        require(len(cases)==10 and {(c['fixture'],c['arrival']) for c in cases}=={(c['fixture'],c['arrival']) for c in schedule(0)}, 'Gate scenario coverage differs')
    return dict(checkpoint='M4-seeded-batch',status='PASS' if complete else 'FAIL',
                full_M4_gate_passed=bool(complete and manifest['profile']=='gate' and len(cases)>=10),
                profile=manifest['profile'],planned_episodes=len(cases),successful_episodes=passed,
                failed_episodes=failed,not_run_episodes=not_run,
                seed_scope='Selection and order of pinned preflight scenes; physics and planner are not reseeded',cases=rows)
