"""Use the M4 renderer for grouped views and one new offline cover."""

import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml

from utility_core import assess_features


POSES = ('assess', 'inspect_left', 'inspect_right')
FAMILIES = ('clear', 'left_reveal', 'right_reveal', 'covered')


def right_scene(scene, folder):
    """Create a raised cover that exposes the patch from the right."""
    tree = ET.parse(scene)
    include = next(el for el in tree.findall('include') if Path(el.get('file')).name == 'props-red.xml')
    props = ET.parse(include.get('file'))
    tray = props.find(".//body[@name='tray']")
    ET.SubElement(tray, 'geom', name='m5c_right_cover', type='box',
                  pos='-.050 .062 .183', size='.043 .015 .002',
                  rgba='.42 .42 .42 1', mass='0', contype='0', conaffinity='0', group='2')
    props_path = folder / 'props-m5c-right.xml'
    props.write(props_path, encoding='utf-8', xml_declaration=True)
    include.set('file', str(props_path))
    result = folder / 'scene-m5c-right.xml'
    tree.write(result, encoding='utf-8', xml_declaration=True)
    return result


def collect(folder, start, count, scene, poses, renderer, config, localization, model):
    """Return only measured image data, and keep label fields for offline scoring."""
    from camera_core import localize
    from classifier_core import fruit_crop, require

    folder.mkdir()
    config_path = folder / 'render-config.yaml'
    config_path.write_text(yaml.safe_dump(dict(config, dataset_groups=count)))
    cases = {}
    maximum_marker_error = 0.
    with tempfile.TemporaryDirectory(dir=folder) as scratch:
        scratch = Path(scratch)
        for variant, source in (('original', scene), ('right', right_scene(scene, folder))):
            images = scratch / variant
            log = folder / f'{variant}-renderer.log'
            with log.open('w') as stream:
                subprocess.run([str(renderer), str(source), str(poses), str(config_path), str(images), str(start)],
                               stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=240)
            require('OpenGL renderer: NVIDIA' in log.read_text(), 'The renderer must use NVIDIA EGL')
            rows = yaml.safe_load((images / 'frames.yaml').read_text())['frames']
            for row in rows:
                condition = row['condition']
                if variant == 'right' and condition != 'clear':
                    continue
                family = ('right_reveal' if variant == 'right' else
                          'left_reveal' if condition == 'partial' else condition)
                rgb = cv2.cvtColor(cv2.imread(str(images / row['image'])), cv2.COLOR_BGR2RGB)
                f = row['height'] / (2 * np.tan(np.deg2rad(row['fovy_deg']) / 2))
                k = np.array([[f, 0, row['width'] / 2], [0, f, row['height'] / 2], [0, 0, 1.]])
                rotation = np.asarray(row['camera_optical_rotation']).reshape(3, 3)
                position = localize(rgb, k, row['camera_xyz'], rotation, localization)['position']
                error = float(np.linalg.norm(np.asarray(position[:2]) - row['evaluator_handle_xyz'][:2]))
                require(error < localization['maximum_planar_error_m'], 'Marker localization exceeds its bound')
                maximum_marker_error = max(maximum_marker_error, error)
                crop, _ = fruit_crop(rgb, k, row['camera_xyz'], rotation, position, config)
                scores = model.infer(crop)
                key = row['group_id'], family, row['label']
                case = cases.setdefault(key, dict(group_id=row['group_id'], appearance_seed=row['appearance_instance'],
                                                   family=family, label=row['label'], views={}))
                require(row['pose'] not in case['views'], 'Duplicate view in an image group')
                case['views'][row['pose']] = scores['probabilities']
                if row['pose'] == 'assess':
                    case['features'] = assess_features(crop, scores['confidence']).tolist()
    expected = {(group, family, label) for group in range(start, start + count)
                for family in FAMILIES for label in (0, 1)}
    require(set(cases) == expected and all(set(c['views']) == set(POSES) for c in cases.values()),
            'The rendered split does not contain every class and view')
    ordered = [cases[k] for k in sorted(cases)]
    for case in ordered:
        require(case['appearance_seed'] == config['dataset_seed'] + case['group_id'] * 7919,
                'Rendered appearance seed differs from the split')
    (folder / 'observations.json').write_text(json.dumps(ordered, indent=2, allow_nan=False) + '\n')
    return ordered, maximum_marker_error
