#!/usr/bin/env python3
"""Check the installed MoveIt pipeline in complete private fixture scenes."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_prefix
from rclpy.serialization import deserialize_message
from moveit_msgs.msg import MotionPlanResponse
from rosidl_runtime_py.convert import message_to_ordereddict
sys.path.insert(0,str(Path(get_package_prefix('sorting_control'))/'lib/sorting_control'))
from carried_checks import fixtures
from m3d_parameters import model_parameters,planning_parameters
from m3d_manifest import installed_manifest

def pipeline_parameters():
    parameters=dict(model_parameters(),**planning_parameters())
    # Component timing uses wall time. No simulated motion or clock input is used.
    parameters['use_sim_time']=False
    return {'/m3d_fixture_planner':{'ros__parameters':parameters}}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    pre=json.loads((args.preflight_dir/'validation.json').read_text());manifest=installed_manifest()
    if pre.get('checkpoint') not in ('M3D','M3D-components') or pre.get('status')!='PASS' or pre.get('installed_manifest')!=manifest or pre.get('native_carried_motion',{}).get('status')!='PASS':parser.error('Run a fresh installed M3D component preflight first')
    folder=args.output_dir.resolve()
    if folder.exists():parser.error('Use a new output directory')
    folder.mkdir(parents=True)
    report=dict(checkpoint='M3D-carried-planning',status='FAIL',robot_commanded=False,scene_context='private_fixture_scenes',installed_manifest=manifest,output_directory=str(folder))
    try:
        fixtures(folder/'fixtures');parameters=folder/'pipeline-parameters.yaml';parameters.write_text(yaml.safe_dump(pipeline_parameters()))
        executable=Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_carried_pipeline'
        env=dict(os.environ,ROS_DOMAIN_ID='67',ROS_LOG_DIR=str(folder/'component-logs'))
        process=subprocess.run([str(executable),str(parameters),str(folder/'fixtures/cases.yaml'),str(folder)],env=env,capture_output=True,text=True,timeout=180.)
        (folder/'pipeline-log.txt').write_text(process.stdout+process.stderr)
        if not (folder/'pipeline-report.yaml').is_file():raise RuntimeError('Pipeline checker did not produce a report; see pipeline-log.txt: '+process.stderr[-1500:])
        details=yaml.safe_load((folder/'pipeline-report.yaml').read_text());report.update(details)
        for record in report.get('cases',[]):print(json.dumps(record),flush=True)
        for path in folder.glob('*.cdr'):
            response=deserialize_message(path.read_bytes(),MotionPlanResponse)
            path.with_suffix('.json').write_text(json.dumps(message_to_ordereddict(response),indent=2)+'\n')
        if process.returncode!=0 or report.get('status')!='PASS' or report.get('successful_plans')!=40 or report.get('scene_context')!='private_fixture_scenes' or report.get('robot_commanded') is not False:raise RuntimeError(report.get('error','Installed fixture pipeline failed; see pipeline-log.txt'))
    except Exception as error:report.update(status='FAIL',error=str(error))
    (folder/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','installed_manifest','scene_checks','pipeline_configuration')},indent=2))
    return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
