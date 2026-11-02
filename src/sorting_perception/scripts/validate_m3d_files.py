#!/usr/bin/env python3
"""Fast read-only component checks followed by native RGB/physics checks."""
import argparse,json,sys,subprocess,os,xml.etree.ElementTree as ET
from pathlib import Path
import xacro,yaml,numpy as np
from ament_index_python.packages import get_package_share_directory,get_package_prefix
from m3d_manifest import installed_manifest
from classifier_core import Classifier,require

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--components-only',action='store_true',help='Run the short component gate; this report cannot authorize a physical trial')
    a=parser.parse_args();folder=a.output_dir.resolve();folder.mkdir(parents=True,exist_ok=True)
    report=dict(checkpoint='M3D',status='FAIL',live_ROS='NOT_RUN',full_M3_gate_passed=False,forced_attachment=False)
    try:
        p=Path(get_package_share_directory('sorting_perception'));d=Path(get_package_share_directory('sorting_description'));c=Path(get_package_share_directory('sorting_control'));s=Path(get_package_prefix('sorting_simulation'))
        sys.path.insert(0,str(Path(get_package_prefix('sorting_control'))/'lib/sorting_control'))
        for package,names in {'sorting_control':('camera_sort.py','camera_sort_scene.py','verify_sort_path'),'sorting_perception':('test_m3d.py','validate_m3d_files.py','summarize_m3d.py')}.items():
            for name in names:require(os.access(Path(get_package_prefix(package))/'lib'/package/name,os.X_OK),'Installed executable missing: '+name)
        require(Path(get_package_share_directory('force_torque_sensor_broadcaster')).is_dir(),'Wrist broadcaster package missing')
        scene=d/'mujoco/static_grasp/scene_camera_m3d_red_clear.xml'
        print('Offline wrist driver selection and MJCF binding',flush=True)
        (folder/'panda.urdf').write_text(xacro.process_file(str(d/'urdf/panda_cameras.urdf.xacro'),mappings={'headless':'true','enable_wrist_ft':'true','camera_config':'cameras_m3d.yaml','mujoco_model':str(scene)}).toxml())
        subprocess.run([str(Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_wrist_model'),str(folder/'panda.urdf'),str(scene),str(d/'mujoco/static_grasp/scene_camera_m3c_red_clear.xml')],check=True)
        report['physical_wrist_interface_schema']='PASS'
        print('Native KDL small-step precision regression',flush=True)
        subprocess.run([str(Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_servo_precision'),
            str(folder/'panda.urdf'),'1e-7',str(folder/'servo-precision.json')],check=True)
        report['native_servo_precision']=json.loads((folder/'servo-precision.json').read_text())
        prefix=Path(get_package_prefix('sorting_control'))/'lib/sorting_control'
        subprocess.run([str(prefix/'verify_precision_smoothing'),str(folder/'smoothing-kernel.json')],check=True)
        report['smoothing_kernel']=json.loads((folder/'smoothing-kernel.json').read_text())
        semantic=ET.parse(Path(get_package_share_directory('moveit_resources_panda_moveit_config'))/'config/panda.srdf').getroot()
        for joint in semantic.findall('virtual_joint'):semantic.remove(joint)
        for group in semantic.findall('group'):
            if group.get('name')=='panda_arm':group.find('chain').set('tip_link','panda_hand')
        ET.ElementTree(semantic).write(folder/'servo-check.srdf',encoding='unicode')
        poses=[[0.,0.,0.,-1.57079,0.,1.57079,-.7853],[.405,.183,.384,-1.591,-.070,1.761,.011],[-.406,.181,-.382,-1.604,.069,1.772,-1.581]]
        (folder/'camera-frame-poses.yaml').write_text(yaml.safe_dump(dict(frames=[dict(view_id=view,joint_positions=pose) for pose in poses for view in ('wrist','overhead')])))
        subprocess.run([str(prefix/'verify_camera_model'),str(folder/'panda.urdf'),str(folder/'servo-check.srdf'),str(scene),str(folder/'camera-frame-poses.yaml'),str(folder/'camera-frames.json')],check=True)
        report['camera_frames']=json.loads((folder/'camera-frames.json').read_text())
        from carried_checks import fixtures as carried_fixtures
        carried_fixtures(folder/'carried-motion')
        subprocess.run([str(prefix/'verify_carried_motion'),str(folder/'panda.urdf'),str(folder/'servo-check.srdf'),
            str(Path(get_package_share_directory('moveit_resources_panda_moveit_config'))/'config/joint_limits.yaml'),
            str(folder/'carried-motion/cases.yaml'),str(c/'config/carried_planning.yaml'),str(folder/'carried-motion/validation.json')],check=True)
        report['native_carried_motion']=json.loads((folder/'carried-motion/validation.json').read_text())
        from m3d_components import capture_pairing_check
        report['capture_time_pairing']=capture_pairing_check(yaml.safe_load((c/'config/m3d.yaml').read_text()))
        print('Installed IK + acceleration smoother component check (no actuation)',flush=True)
        component_env=dict(os.environ,ROS_DOMAIN_ID='67',ROS_LOG_DIR=str(folder/'component-logs'))
        component=subprocess.run([str(prefix/'verify_servo_pipeline'),str(folder/'panda.urdf'),str(folder/'servo-check.srdf'),
            str(Path(get_package_share_directory('moveit_resources_panda_moveit_config'))/'config/joint_limits.yaml'),str(folder/'servo-pipeline.json')],
            env=component_env,capture_output=True,text=True,timeout=30.)
        (folder/'servo-pipeline-log.txt').write_text(component.stdout+component.stderr)
        require(component.returncode==0,'Installed IK/smoothing component check failed; see '+str(folder/'servo-pipeline-log.txt')+'\n'+(component.stdout+component.stderr)[-2000:])
        report['installed_servo_pipeline']=json.loads((folder/'servo-pipeline.json').read_text())
        report['live_ROS']='PRIVATE_COMPONENT_NODE_NO_ACTUATION'
        if a.components_only:
            report.update(checkpoint='M3D-components',status='PASS',installed_manifest=installed_manifest(),scope='Read-only components; physical trials require the full M3D preflight',output_directory=str(folder))
            (folder/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2));return 0
        from native_sort import run_fixture
        from sort_fixtures import write_sort_fixtures
        model=Classifier(p/'models/m3c_baseline',yaml.safe_load((p/'config/m3c.yaml').read_text()))
        sort_config=yaml.safe_load((c/'config/m3d.yaml').read_text());require(sort_config['class_ids']==model.c['class_ids'] and sort_config['minimum_confidence']>=model.c['acceptance_confidence'],'Routing cannot bypass classifier abstention')
        report['native_fixtures']={}
        for fixture in ('red_clear','green_clear','red_covered','green_covered'):
            print('Native fixture: '+fixture,flush=True)
            report['native_fixtures'][fixture]=run_fixture(s/'lib/libnative_camera_plant.so',d/f'mujoco/static_grasp/scene_camera_m3d_{fixture}.xml',p,c,folder/fixture,'red_fruit' if fixture.startswith('red') else 'green_fruit',fixture.endswith('covered'))
        scene=d/'mujoco/static_grasp/scene_camera_m3d_red_clear.xml'
        camera=yaml.safe_load((p/'config/cameras_m3d.yaml').read_text())
        plugins=yaml.safe_load((Path(get_package_share_directory('sorting_simulation'))/'config/camera_plugins_m3d.yaml').read_text())['/**']['ros__parameters']['mujoco_plugins']['camera_plugin']
        require(camera['publish_rate_hz']==20. and plugins['camera_publish_rate']==camera['publish_rate_hz'], 'M3D camera rates differ')
        require(plugins['wrist']['policy']=='streaming' and plugins['overhead']['policy']=='disabled', 'M3D requires the single wrist RGB stream')
        print('Native M3D 20 Hz render budget',flush=True)
        render=folder/'render-budget'
        subprocess.run([str(s/'lib/sorting_simulation/verify_camera_render'),str(scene),str(scene),
            str(p/'config/cameras_m3d.yaml'),str(render),'--benchmark'],check=True)
        benchmark=yaml.safe_load((render/'render-benchmark.yaml').read_text())['render_benchmark']
        period=1./camera['publish_rate_hz'];wrist=benchmark['views']['wrist']
        require(wrist['p95_render_read_wall_s']<.7*period,'Wrist render cost exceeds the 20 Hz budget')
        report['camera_render_budget']=dict(status='PASS',requested_rate_hz=camera['publish_rate_hz'],period_s=period,
            maximum_allowed_p95_wall_s=.7*period,wrist=wrist,scope='Native render/read only; live transport, scheduling and viewer load remain unmeasured')
        from hold_handoff_probe import run_probe
        print('Native loaded-hold regression',flush=True)
        report['native_loaded_hold_regression']=run_probe(s/'lib/libnative_camera_plant.so',scene,p,c,folder/'hold-regression')
        from carried_endpoint_probe import run_probe as run_endpoint_probe
        print('Native mixed MoveIt/Servo endpoint regression',flush=True)
        report['native_carried_endpoint_regression']=run_endpoint_probe(s/'lib/libnative_camera_plant.so',scene,p,c,folder/'endpoint-regression')
        from floor_release_probe import run_probe as run_release_probe
        print('Native controlled release and height-bias regression',flush=True)
        report['native_floor_release_regression']=run_release_probe(s/'lib/libnative_camera_plant.so',d,p,c,folder/'release-regression')
        root=ET.parse(Path(get_package_share_directory('moveit_resources_panda_moveit_config'))/'config/panda.srdf').getroot()
        for joint in root.findall('virtual_joint'):root.remove(joint)
        ET.ElementTree(root).write(folder/'panda.srdf',encoding='unicode')
        for fixture in ('red_clear','green_clear'):
            path=folder/fixture;items=write_sort_fixtures(path,c)
            with (path/'moveit-placement.txt').open('w') as log:
                subprocess.run([str(Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_sort_path'),str(folder/'panda.urdf'),str(folder/'panda.srdf'),*items],stdout=log,stderr=subprocess.STDOUT,check=True)
        report.update(status='PASS',installed_manifest=installed_manifest(),model_sha256=model.sha,native_payload_collision_checks='PASS',native_constrained_ompl_segments='PASS',scope='Installed IK/smoothing components, native RGB, physical wrist feedback, free-body physics and MoveIt carried path. Live MoveIt services/actions/Servo/reset tests remain required.',output_directory=str(folder))
    except Exception as error:report['error']=str(error)
    (folder/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
