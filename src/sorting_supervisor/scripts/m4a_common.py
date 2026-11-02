"""Shared installed paths, immutable M3D dependency hashes and M4A fixtures."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory


def share(package):
    return Path(get_package_share_directory(package))


def executable(package, name):
    return Path(get_package_prefix(package))/'lib'/package/name


for _package in ('sorting_control', 'sorting_perception'):
    sys.path.insert(0, str(executable(_package, '')))
from classifier_core import digest
from m3d_manifest import installed_manifest


def config():
    return yaml.safe_load((share('sorting_supervisor')/'config/m4a.yaml').read_text())


def manifest():
    result = installed_manifest()
    package = share('sorting_supervisor')
    for path in sorted((package/'config').glob('*')):
        if path.is_file():
            result['m4a/config/'+path.name] = digest(path)
    for path in sorted((package/'launch').glob('*.py')):
        result['m4a/launch/'+path.name] = digest(path)
    for path in sorted(executable('sorting_supervisor', '').glob('*.py')):
        result['m4a/scripts/'+path.name] = digest(path)
    result['m4a/native'] = digest(executable('sorting_supervisor', 'render_m4a_views'))
    result['m4a/moveit_scene_check'] = digest(executable('sorting_supervisor', 'verify_m4a_scene'))
    return result


def create_scenes(out):
    """One static partial occluder. Pose changes never change its visibility flag."""
    source = share('sorting_description')/'mujoco/static_grasp'
    panda = ET.parse(source/'panda_camera_m3d.xml')
    panda.find('compiler').set('meshdir', str((source/'../franka_emika_panda/assets').resolve()))
    panda.write(out/'panda.xml', encoding='utf-8', xml_declaration=True)
    scenes = {}
    for label in ('red', 'green'):
        props = ET.parse(source/f'props_camera_m3c_{label}_covered.xml')
        cover = props.find(".//geom[@name='fruit_front_occluder']")
        cover.set('pos', '-.0357 .0407 .0776')
        cover.set('size', '.021 .018 .002')
        props.write(out/f'props-{label}.xml', encoding='utf-8', xml_declaration=True)
        scene = ET.parse(source/f'scene_camera_m3d_{label}_covered.xml')
        scene.getroot().set('model', 'M4A partial-occlusion development fixture')
        for item in scene.findall('include'):
            name = item.get('file')
            item.set('file', str(out/f'props-{label}.xml') if name.startswith('props') else
                     str(out/'panda.xml') if name.startswith('panda') else str(source/name))
        scene.write(out/f'scene-{label}.xml', encoding='utf-8', xml_declaration=True)
        scenes[label] = out/f'scene-{label}.xml'
    return scenes


def generated_hashes(out):
    return {p.name: digest(p) for p in sorted(out.glob('*.xml'))}


def checked_preflight(folder):
    folder = Path(folder).expanduser().resolve()
    report = json.loads((folder/'preflight.json').read_text())
    if report.get('status') != 'PASS' or report.get('checkpoint') != 'M4A':
        raise ValueError('A passing M4A preflight is required')
    if report['installed_manifest'] != manifest():
        raise ValueError('Installed files changed after preflight; run a new preflight')
    if report['generated_hashes'] != generated_hashes(folder):
        raise ValueError('Generated scene changed after preflight')
    return folder, report


def robot_parameters(scene, headless='true'):
    from m3d_parameters import model_parameters
    model = model_parameters(headless, 'red_covered')
    robot = ET.fromstring(model['robot_description'])
    robot.find(".//ros2_control/hardware/param[@name='mujoco_model']").text = str(scene)
    model['robot_description'] = ET.tostring(robot, encoding='unicode')
    return model
