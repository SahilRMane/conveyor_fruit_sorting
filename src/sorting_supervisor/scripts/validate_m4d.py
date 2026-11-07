#!/usr/bin/env python3
"""Bind the passed handoffs and check the new installed physical integration."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import yaml
from unittest.mock import patch
from m4d_common import prerequisite_reports,config
from m4a_common import executable,share
from m4c_common import manifest
from m4b_common import model_directory
from classifier_core import digest,require


def launch_check(out,report):
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters
    source=share('sorting_supervisor')/'launch/panda_m4d.launch.py'
    spec=importlib.util.spec_from_file_location('m4d_launch_check',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for color in ('red','green'):
        context=LaunchContext()
        context.launch_configurations.update(preflight_dir=str(out),fixture=color+'_partial',arrival='upstream',headless='true')
        with patch('m4d_common.checked_preflight',return_value=(out,report)):
            nodes=module.setup(context)
        names=[n.node_executable for n in nodes]
        require(names.count('camera_sort_scene.py')==1 and 'publish_static_fixtures.py' not in names,
                'Integrated launch needs one camera-derived scene writer')
        require(names.count('ros2_control_node')==1 and 'camera_sort.py' not in names,
                'Integrated launch cannot start a second physical command process')
        require(names.count('m4d_filter.py')==1 and 'motion_filter.py' not in names,
                'Integrated launch must use the checked overhead-to-wrist transition')
        localizer=next(n for n in nodes if n.node_executable=='handle_localizer.py')
        values=evaluate_parameters(context,localizer._Node__parameters)
        require(any(isinstance(v,dict) and v.get('process_period_s')==.002 for v in values),
                'M4D must use the checked 2 ms image-processing period')
        rows.append(dict(fixture=color+'_partial',executables=names))
    return dict(status='PASS',processes_started=0,cases=rows)


def carried_planning_check(out):
    from carried_checks import fixtures
    from probe_m3d_planning import pipeline_parameters
    folder=out/'carried-planning';folder.mkdir()
    fixtures(folder/'fixtures')
    parameters=folder/'pipeline-parameters.yaml'
    parameters.write_text(yaml.safe_dump(pipeline_parameters()))
    env=dict(os.environ,ROS_DOMAIN_ID='68',ROS_LOG_DIR=str(folder/'ros-logs'))
    with (folder/'pipeline.log').open('w') as log:
        process=subprocess.run([str(executable('sorting_control','verify_carried_pipeline')),
            str(parameters),str(folder/'fixtures/cases.yaml'),str(folder)],
            stdout=log,stderr=subprocess.STDOUT,env=env,timeout=180.)
    report=yaml.safe_load((folder/'pipeline-report.yaml').read_text())
    require(process.returncode==0 and report.get('status')=='PASS' and report.get('successful_plans')==40
            and report.get('scene_context')=='private_fixture_scenes' and report.get('robot_commanded') is False,
            'Carried planning preflight failed; read carried-planning/pipeline.log')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.expanduser().resolve()
    if out.exists():parser.error('Use a new output directory to preserve previous evidence')
    out.mkdir(parents=True);os.environ.setdefault('ROS_LOG_DIR',str(out/'ros-logs'))
    report=dict(checkpoint='M4D.2',status='FAIL',live_physical='NOT_RUN',full_M4_gate_passed=False)
    try:
        c=config();require(c['maximum_route_age_sim_s']<=30.,'Class route lifetime exceeds this checkpoint')
        require(0<c['wrench_tf_wait_s']<=.05 and c['wrench_tf_wait_s']<=c['maximum_wrench_tf_wait_total_s']<=.2,
                'Wrist transform waits exceed the checked bounds')
        reports=prerequisite_reports(args.results_root)
        model=digest(model_directory()/'weights.npz')
        require(all(r['model_sha256']==model for r in reports),'Classifier changed since the handoff checks')
        report.update(prerequisites=reports,model_sha256=model)
        with (out/'inspection-preflight.log').open('w') as log:
            subprocess.run([str(executable('sorting_supervisor','validate_m4c.py')),
                '--m4b-results-root',str(args.results_root.resolve()),'--output-dir',str(out/'inspection')],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        report['launch_check']=launch_check(out,report)
        report['carried_planning']=carried_planning_check(out)
        report.update(status='PASS',installed_manifest=manifest(),output_directory=str(out),
            scope='New source, camera physics, classifier domain, collision scenes and historical handoff evidence; live physical pilot required')
    except Exception as error:report['error']=str(error)
    (out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('installed_manifest','prerequisites')},indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
