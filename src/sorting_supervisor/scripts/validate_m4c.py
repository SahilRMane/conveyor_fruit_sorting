#!/usr/bin/env python3
"""Offline M4C source, scene, RGB, transport, filter and launch preflight. No ROS node."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import yaml
from m4c_common import config,manifest,scenes,generated_hashes,measured_costs
from m4b_common import model_directory,classifier_config
from m4a_common import share,executable,robot_parameters
from m4c_checks import check_native
from m4c_timing import timing_budget
from classifier_core import digest,require


def launch_check(out, report):
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters
    from unittest.mock import patch
    path=share('sorting_supervisor')/'launch/panda_m4c.launch.py'
    spec=importlib.util.spec_from_file_location('m4c_launch_check',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=[]
    for fixture in ('red_clear','green_clear','red_partial','green_partial','red_covered','green_covered'):
        for arrival in ('upstream','late'):
            context=LaunchContext();context.launch_configurations.update(preflight_dir=str(out),fixture=fixture,arrival=arrival,headless='true')
            with patch('m4c_common.checked_preflight',return_value=(out,report)):
                nodes=module.setup(context)
            require(sum(n.node_executable=='ros2_control_node' for n in nodes)==1,'M4C must launch exactly one hardware process')
            require(not any(n.node_executable in ('camera_pickup.py','camera_sort.py') for n in nodes),'M4C must not launch a pickup executor')
            for node in nodes:
                values=evaluate_parameters(context,node._Node__parameters)
                for value in values:
                    if isinstance(value,Path):require(value.is_file(),'M4C launch parameter file is missing')
                records.append(dict(fixture=fixture,arrival=arrival,package=node.node_package,executable=node.node_executable))
    return dict(status='PASS',processes_started=0,actions_checked=len(records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--m4b-results-root',type=Path,required=True)
    a=parser.parse_args();out=a.output_dir.expanduser().resolve()
    if out.exists():parser.error('Use a new output directory to preserve previous evidence')
    out.mkdir(parents=True);os.environ.setdefault('ROS_LOG_DIR',str(out/'launch-logs'))
    r=dict(checkpoint='M4C',status='FAIL',full_M4_gate_passed=False,live_ROS='NOT_RUN',
           physical_pickup_permitted=False,output_directory=str(out))
    try:
        c=config();cc=classifier_config()
        import xml.etree.ElementTree as ET
        require(ET.parse(share('mujoco_ros2_control_plugins')/'package.xml').findtext('version')=='0.1.2',
                'Camera trace requires the validated mujoco_ros2_control_plugins version 0.1.2')
        r['timing_budget']=timing_budget(c)
        require(c['overhead_start_maximum_capture_age_s']+1./c['camera_rate_hz']<c['maximum_capture_age_s'],
                'Overhead startup does not reserve a complete camera period')
        routing=yaml.safe_load((share('sorting_control')/'config/m3d.yaml').read_text())
        require(c['class_ids']==cc['class_ids'] and c['acceptance_confidence']>=cc['acceptance_confidence'],
                'M4C class order or confidence threshold differs from the frozen classifier')
        require(c['bins']==routing['class_to_bin'],'M4C proposed bins differ from the passed routing rules')
        costs=measured_costs(a.m4b_results_root,digest(model_directory()/'weights.npz'))
        (out/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
        with (out/'m4b-preflight.log').open('w') as log:
            subprocess.run([str(executable('sorting_supervisor','validate_m4b.py')),'--output-dir',str(out/'m4b')],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        paths=scenes(out)
        cameras=yaml.safe_load((share('sorting_perception')/'config/cameras_m3d.yaml').read_text())
        cameras['publish_rate_hz']=c['camera_rate_hz'];cameras['cameras']['overhead']['publish_policy']='streaming'
        cameras['cameras']['overhead']['resolution']=c['overhead_resolution']
        (out/'cameras.yaml').write_text(yaml.safe_dump(cameras))
        plugins=yaml.safe_load((share('sorting_simulation')/'config/camera_plugins_m3d.yaml').read_text())
        camera=plugins['/**']['ros__parameters']['mujoco_plugins']['camera_plugin']
        camera['type']='sorting_simulation/TracedCameraPlugin'
        camera['camera_publish_rate']=c['camera_rate_hz'];camera['overhead']['policy']='streaming'
        (out/'camera_plugins.yaml').write_text(yaml.safe_dump(plugins))
        fc=yaml.safe_load((share('sorting_perception')/'config/m3b_filter.yaml').read_text())
        # This separate filter is checked only for the constant-speed M4C scene.
        fc.update(tracking_view='overhead',acceleration_sigma_m_s2=.005,velocity_sigma_floor_m_s=.0005)
        (out/'filter.yaml').write_text(yaml.safe_dump(fc))
        with (out/'native.log').open('w') as log:
            subprocess.run([str(executable('sorting_supervisor','probe_m4c_scene')),paths['red_partial:upstream'],
                            str(out/'m4b/m4a/native/views.yaml'),str(share('sorting_supervisor')/'config/m4c.yaml'),str(out/'native')],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        native=check_native(out/'native',c,fc,costs['upper_sim_s'])
        with (out/'camera-side.log').open('w') as log:
            subprocess.run([str(executable('sorting_simulation','verify_camera_trace')),paths['red_partial:upstream'],
                str(out/'camera-side-check.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        camera_side=json.loads((out/'camera-side-check.json').read_text())
        require(camera_side.get('status')=='PASS' and camera_side.get('parity_frames')==8 and camera_side.get('worker_context') is True,
                'Camera-side RGB/depth parity, worker EGL or snapshot isolation failed')
        if os.environ.get('SORTING_RENDER_BACKEND')=='nvidia':
            require('NVIDIA' in camera_side.get('gl_vendor',''),'Preflight did not use the requested NVIDIA renderer')
        with (out/'collision.log').open('w') as log:
            subprocess.run([str(executable('sorting_supervisor','verify_m4a_scene')),str(out/'m4b/m4a/panda.urdf'),
                str(out/'m4b/m4a/panda.srdf'),str(out/'m4b/m4a/static-scene.cdr'),str(out/'native/probe.yaml'),
                str(out/'collision-check.json'),str(out/'m4b/m4a/parameters.yaml')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
        # Resolve the model for every launch scene, including the late-arrival case.
        for scene in paths.values():require(str(scene) in robot_parameters(Path(scene))['robot_description'],'M4C model path was not installed')
        r.update(scene_paths=paths,launch_check=launch_check(out,dict(scene_paths=paths)),native=native,
                 collision_check=json.loads((out/'collision-check.json').read_text()),camera_side_check=camera_side,measured_costs=costs,
                 camera_rate_hz=c['camera_rate_hz'],belt_speed_m_s=c['belt_speed_m_s'],
                 scope='Upstream moving inspection, overhead continuity, capture/track binding and deadline proposals only',
                 installed_manifest=manifest(),generated_hashes=generated_hashes(out),status='PASS')
    except Exception as error:r['error']=str(error)
    (out/'preflight.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('checkpoint','status','live_ROS','full_M4_gate_passed','physical_pickup_permitted',
        'belt_speed_m_s','camera_rate_hz','output_directory','error') if k in r},indent=2))
    raise SystemExit(0 if r['status']=='PASS' else 1)


if __name__=='__main__':main()
