#!/usr/bin/env python3
"""Run the pinned M4 scenarios automatically. Stop the batch on the first failure."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from batch_core import (aggregate, assess_episode, check_prerequisite, digest, require,
                        schedule, source_hashes, write_json)

BUNDLE=Path(__file__).resolve().parent


def precheck(preflight_dir, results_root):
    from ament_index_python.packages import get_package_prefix
    sys.path.insert(0,str(Path(get_package_prefix('sorting_supervisor'))/'lib/sorting_supervisor'))
    from m4d_common import checked_preflight
    folder,report=checked_preflight(preflight_dir)
    require(report['installed_manifest'].get('m4c/render_backend_request')=='nvidia','Current NVIDIA preflight is required')
    require(report.get('carried_planning',{}).get('successful_plans')==40,'Carried planning preflight is incomplete')
    proofs=[]
    for color in ('red','green'):
        pointer=Path(results_root)/f'M4D-partial-{color}-current.txt'
        source=Path(pointer.read_text().strip())/'summary.json'
        pilot=json.loads(source.read_text());check_prerequisite(pilot,color,report['installed_manifest'])
        proofs.append(dict(color=color,path=str(source),sha256=digest(source)))
    return folder,report,proofs


def tagged_pids(token):
    marker=('SORTING_M4_BATCH_TOKEN='+token).encode()
    found=[]
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if marker in (entry/'environ').read_bytes().split(b'\0'):
                found.append(int(entry.name))
        except OSError:pass
    return found


def cleanup(processes,token):
    phases=[]
    for sig,seconds in ((signal.SIGINT,12.),(signal.SIGTERM,6.),(signal.SIGKILL,2.)):
        running=[p for p in processes if p is not None and p.poll() is None]
        owned=tagged_pids(token)
        if not running and not owned:break
        phases.append(dict(signal=sig.name,owned_pids=owned))
        for process in running:
            try:os.killpg(process.pid,sig)
            except ProcessLookupError:pass
        for pid in owned:
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if all(p is None or p.poll() is not None for p in processes) and not tagged_pids(token):break
            time.sleep(.1)
    for process in processes:
        if process is not None and process.poll() is not None:process.wait()
    leftovers=tagged_pids(token)
    require(not leftovers,'Owned simulation processes did not stop: '+str(leftovers))
    return dict(status='PASS',signals=phases,forced_termination=any(p['signal']=='SIGKILL' for p in phases))


def wait_empty_graph():
    end=time.monotonic()+20.
    while True:
        command=subprocess.run(['ros2','node','list','--no-daemon','--spin-time','2'],
            text=True,capture_output=True,timeout=8.)
        require(command.returncode==0,'Cannot inspect ROS graph: '+command.stderr[-500:])
        nodes=[line.strip() for line in command.stdout.splitlines() if line.strip().startswith('/')]
        if not nodes:return
        require(time.monotonic()<end,'Stop the previous Terminal 1 launch before this batch: '+', '.join(nodes))
        time.sleep(1.)


def run_case(case, preflight, destination, root_report, hashes):
    require(source_hashes(BUNDLE)==hashes,'Batch code changed during execution')
    folder=destination/f"episode-{case['index']:02d}-{case['fixture']}-{case['arrival']}"
    folder.mkdir();write_json(folder/'case.json',case)
    token=uuid.uuid4().hex
    env=dict(os.environ,SORTING_M4_BATCH_TOKEN=token,ROS_LOG_DIR=str(folder/'ros-logs'))
    launch=None;episode=None;result=dict(case,status='FAIL')
    try:
        wait_empty_graph()
        with (folder/'launch.log').open('w') as launch_log,(folder/'episode.log').open('w') as episode_log:
            launch=subprocess.Popen(['ros2','launch','sorting_supervisor','panda_m4d.launch.py',
                'preflight_dir:='+str(preflight),'fixture:='+case['fixture'],'arrival:='+case['arrival'],'headless:=true'],
                stdout=launch_log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            episode=subprocess.Popen([sys.executable,str(BUNDLE/'run_m4_episode.py'),
                '--preflight-dir',str(preflight),'--case-file',str(folder/'case.json'),'--output-dir',str(folder/'result')],
                stdout=episode_log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            end=time.monotonic()+480.
            while episode.poll() is None:
                require(launch.poll() is None,'Simulation launch exited during the episode')
                require(time.monotonic()<end,'Episode exceeded its 480-second process bound')
                time.sleep(.2)
            result['episode_returncode']=episode.returncode
        report_path=folder/'result/summary.json'
        report=json.loads(report_path.read_text());result.update(report_path=str(report_path),report_sha256=digest(report_path))
        require(episode.returncode==0,report.get('error','Episode process failed'))
        assess_episode(report,case,root_report['installed_manifest'])
        result.update(status='PASS',action=report['action'],policy_path=report['policy_path'],
            final_reset_status=report['final_reset_status'],final_hold_status=report['final_hold_status'],
            overlap_violations=report['overlap_violations'])
    except (Exception,KeyboardInterrupt) as error:
        result['error']=str(error) or 'Interrupted'
        if episode is not None and episode.poll() is None:
            try:os.killpg(episode.pid,signal.SIGINT)
            except ProcessLookupError:pass
            try:episode.wait(timeout=15.)
            except subprocess.TimeoutExpired:pass
    finally:
        try:result['teardown']=cleanup([episode,launch],token)
        except Exception as error:result.update(status='FAIL',teardown=dict(status='FAIL',error=str(error)))
        write_json(folder/'outcome.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-dir',type=Path,required=True)
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261104)
    parser.add_argument('--profile',choices=('gate','smoke'),default='gate')
    parser.add_argument('--case-index',type=int,choices=range(1,11),
                        help='Run one case from the seeded gate schedule as a separate probe.')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check',action='store_true',help='Check the current preflight and prerequisites. Command no robot.')
    args=parser.parse_args()
    if args.case_index is not None and args.profile != 'gate':
        parser.error('--case-index requires the gate schedule')
    preflight,report,proofs=precheck(args.preflight_dir,args.results_root)
    cases=schedule(args.seed,args.profile);hashes=source_hashes(BUNDLE)
    profile=args.profile
    if args.case_index is not None:
        cases=[cases[args.case_index-1]]
        profile='probe'
    manifest=dict(profile=profile,base_seed=args.seed,cases=cases,preflight_directory=str(preflight),
        preflight_sha256=digest(preflight/'preflight.json'),installed_manifest=report['installed_manifest'],
        batch_source_hashes=hashes,prerequisites=proofs,physics_seed_controlled=False,planner_seed_controlled=False)
    if args.check:
        print(json.dumps(dict(status='PASS',profile=profile,scheduled_episodes=len(cases),
            robot_commanded=False,prerequisites=proofs,seed_scope='Selection and ordering of existing preflight scenes'),indent=2));return 0
    if args.output_dir is None:parser.error('--output-dir is required for execution')
    out=args.output_dir.expanduser().resolve()
    if out.exists():parser.error('Use a new batch directory to preserve previous evidence')
    require('ROS_DOMAIN_ID' in os.environ and 1<=int(os.environ['ROS_DOMAIN_ID'])<=232,'Set an explicit ROS_DOMAIN_ID from 1 to 232')
    manifest['ros_domain_id']=int(os.environ['ROS_DOMAIN_ID'])
    out.mkdir(parents=True);write_json(out/'manifest.json',manifest);manifest_sha=digest(out/'manifest.json')
    rows=[dict(case,status='NOT_RUN') for case in cases];started=time.monotonic()
    write_json(out/'progress.json',dict(status='RUNNING',current_episode=None,planned_episodes=len(cases),cases=rows))
    batch_error=None
    try:
        for index,case in enumerate(cases):
            require(digest(out/'manifest.json')==manifest_sha,'Batch manifest changed during execution')
            current,current_report,current_proofs=precheck(preflight,args.results_root)
            require(current_report['installed_manifest']==manifest['installed_manifest'] and current_proofs==proofs,
                    'Installed preflight or prerequisite evidence changed during the batch')
            rows[index]=dict(case,status='RUNNING')
            write_json(out/'progress.json',dict(status='RUNNING',current_episode=index+1,planned_episodes=len(cases),cases=rows))
            print(f"Episode {index+1}/{len(cases)}: {case['fixture']} {case['arrival']} seed={case['scenario_selection_seed']}",flush=True)
            rows[index]=run_case(case,current,out,report,hashes)
            print(f"Episode {index+1}: {rows[index]['status']}",flush=True)
            if rows[index]['status']!='PASS':break
        require(digest(out/'manifest.json')==manifest_sha,'Batch manifest changed during execution')
        require(source_hashes(BUNDLE)==hashes,'Batch code changed during execution')
        require(digest(preflight/'preflight.json')==manifest['preflight_sha256'],'Preflight changed during execution')
    except (Exception,KeyboardInterrupt) as error:
        running=next((i for i,r in enumerate(rows) if r['status']=='RUNNING'),None)
        if running is not None:rows[running].update(status='FAIL',error=str(error) or 'Interrupted')
        batch_error=str(error) or 'Interrupted'
    result=aggregate(manifest,rows)
    if batch_error:result.update(status='FAIL',full_M4_gate_passed=False,infrastructure_error=batch_error)
    result.update(elapsed_wall_s=time.monotonic()-started,manifest_sha256=manifest_sha,
        preflight_directory=str(preflight),batch_source_hashes=hashes,output_directory=str(out))
    write_json(out/'summary.json',result);write_json(out/'progress.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','batch_source_hashes')},indent=2))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
