#!/usr/bin/env python3
"""Native pose, visual observability, capture geometry and render checks. No ROS node."""
import argparse
import json
from pathlib import Path
import subprocess
import os
import importlib.util
import cv2
import numpy as np
import yaml
from m4a_common import config, create_scenes, executable, generated_hashes, manifest, robot_parameters, share
from camera_core import localize
from classifier_core import Classifier, fruit_crop, require


def check_launch(folder):
    # Construct and resolve launch actions without starting their processes.
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters
    path = share('sorting_supervisor')/'launch/panda_m4a.launch.py'
    spec = importlib.util.spec_from_file_location('m4a_launch_check', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    os.environ.setdefault('ROS_LOG_DIR', str(folder/'launch-logs'))
    context = LaunchContext()
    records = []
    for label in ('red', 'green'):
        for headless in ('true', 'false'):
            for node in module.build_nodes(folder, label, headless):
                # Jazzy Node stores the normalized input here. No execute() call.
                values = evaluate_parameters(context, node._Node__parameters)
                for value in values:
                    if isinstance(value, Path):
                        require(value.is_file(), 'Launch parameter file missing')
                records.append(dict(fixture=label, headless=headless, package=node.node_package,
                                    executable=node.node_executable, parameter_sources=len(values)))
    return dict(status='PASS', actions_checked=len(records), processes_started=0, actions=records)


def evaluate_views(folder, cfg):
    native = yaml.safe_load((folder/'native/views.yaml').read_text())
    perception = share('sorting_perception')
    cc = yaml.safe_load((perception/'config/m3c.yaml').read_text())
    lc = yaml.safe_load((perception/'config/m3b_localization.yaml').read_text())
    model = Classifier(perception/'models/m3c_baseline', cc)
    checks = {}
    for name, view in native['views'].items():
        h, w = view['height'], view['width']
        f = h/2/np.tan(np.deg2rad(view['fovy_deg'])/2)
        k = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1.]])
        r = np.asarray(view['camera_optical_rotation']).reshape(3, 3)
        frames = [cv2.cvtColor(cv2.imread(str(folder/f'native/{name}-{label}.ppm')), cv2.COLOR_BGR2RGB)
                  for label in ('red', 'green')]
        changed = np.max(np.abs(frames[0].astype(int)-frames[1].astype(int)), axis=2) > 15
        entry = dict(counterfactual_changed_pixels=int(changed.sum()), samples=[])
        for label, rgb in zip(('red', 'green'), frames):
            loc = localize(rgb, k, view['camera_xyz'], r, lc)
            error = float(np.linalg.norm(np.asarray(loc['position'][:2])-view['evaluator_handle_xyz'][:2]))
            require(error < cfg['maximum_localization_error_m'], name+': localization exceeds 2 mm')
            crop, box = fruit_crop(rgb, k, view['camera_xyz'], r, loc['position'], cc)
            entry['changed_pixels_inside_crop'] = int(changed[box[1]:box[3], box[0]:box[2]].sum())
            entry['samples'].append(dict(label=label, planar_error_m=error, crop_xyxy=box,
                                         classifier_diagnostic_only=model.infer(crop)))
        checks[name] = entry
    require(checks['assess']['counterfactual_changed_pixels'] <= cfg['maximum_hidden_pixels'], 'Initial cue is visible')
    require(max(checks[v]['changed_pixels_inside_crop'] for v in ('inspect_left', 'inspect_right')) >=
            cfg['minimum_revealed_pixels'], 'Neither candidate reveals class evidence')
    wrist = native['benchmark']['views']['wrist']['p95_render_read_wall_s']
    require(wrist < cfg['maximum_render_p95_wall_s'], 'Wrist render cost exceeds the 20 Hz budget')
    return native, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_dir.expanduser().resolve()
    if folder.exists():
        parser.error('Use a new output directory to preserve the previous evidence')
    folder.mkdir(parents=True)
    report = dict(checkpoint='M4A', status='FAIL', full_M4_gate_passed=False, live_ROS='NOT_RUN',
                  moving_conveyor=False, classifier_new_views_validated=False)
    try:
        cfg = config()
        scenes = create_scenes(folder)
        baseline = share('sorting_description')/'mujoco/static_grasp/scene_camera_m3d_red_covered.xml'
        with (folder/'native.log').open('w') as log:
            subprocess.run([str(executable('sorting_supervisor', 'render_m4a_views')), str(scenes['red']),
                            str(baseline), str(share('sorting_supervisor')/'config/m4a.yaml'), str(folder/'native')],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        native, checks = evaluate_views(folder, cfg)
        model = robot_parameters(scenes['red'])
        (folder/'panda.urdf').write_text(model['robot_description'])
        (folder/'panda.srdf').write_text(model['robot_description_semantic'])
        from m3d_parameters import planning_parameters
        from publish_static_fixtures import scene_message
        from rclpy.serialization import serialize_message
        (folder/'static-scene.cdr').write_bytes(serialize_message(scene_message()))
        parameters = dict(model, **planning_parameters())
        (folder/'parameters.yaml').write_text(yaml.safe_dump({'/m4a_parameter_check': {'ros__parameters': parameters}}))
        with (folder/'collision.log').open('w') as log:
            subprocess.run([str(executable('sorting_supervisor', 'verify_m4a_scene')), str(folder/'panda.urdf'),
                            str(folder/'panda.srdf'), str(folder/'static-scene.cdr'), str(folder/'native/views.yaml'),
                            str(folder/'collision-check.json'), str(folder/'parameters.yaml')],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=30)
        poses = dict(frames=[dict(view_id='wrist', joint_positions=v['joints']) for v in native['views'].values()])
        (folder/'frames.yaml').write_text(yaml.safe_dump(poses))
        with (folder/'frames.log').open('w') as log:
            subprocess.run([str(executable('sorting_control', 'verify_camera_model')), str(folder/'panda.urdf'),
                            str(folder/'panda.srdf'), str(scenes['red']), str(folder/'frames.yaml'),
                            str(folder/'frame-check.json')], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=30)
        # Both class scenes must differ only in the declared skin RGB value.
        import xml.etree.ElementTree as ET
        red = ET.parse(folder/'props-red.xml'); green = ET.parse(folder/'props-green.xml')
        for tree in (red, green):
            tree.find(".//geom[@name='fruit_class_skin']").set('rgba', 'CLASS_COUNTERFACTUAL')
        require(ET.tostring(red.getroot()) == ET.tostring(green.getroot()), 'Class leaks into scene structure')
        report.update(status='PASS', installed_manifest=manifest(), generated_hashes=generated_hashes(folder),
                      views=checks, physics_preserved=native['physics_preserved'],
                      camera_frames=json.loads((folder/'frame-check.json').read_text()), benchmark=native['benchmark'],
                      scope='Stationary development fixture; geometry and timing only; no new-view classifier acceptance gate')
        report['launch_check'] = check_launch(folder)
        report['collision_check'] = json.loads((folder/'collision-check.json').read_text())
        report['native_refinement'] = dict(scope='Integrated native DLS feedback and physical actuators; live MoveIt Servo NOT_RUN',
                                           planned_gravity_bias_added=False, moves=native['refinements'])
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = str(error)
    (folder/'preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    summary = {k: report[k] for k in ('checkpoint', 'status', 'full_M4_gate_passed', 'live_ROS',
               'moving_conveyor', 'classifier_new_views_validated', 'error') if k in report}
    summary['output_directory'] = str(folder)
    if report['status'] == 'PASS':
        summary.update(native_path_states=report['collision_check']['native_path_states_checked'],
                       launch_actions_checked=report['launch_check']['actions_checked'],
                       revealed_pixels=report['views']['inspect_left']['changed_pixels_inside_crop'],
                       wrist_render_p95_wall_s=report['benchmark']['views']['wrist']['p95_render_read_wall_s'])
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
