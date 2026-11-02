"""M4B paths, source/model binding and stationary visual fixtures."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
from m4a_common import share, executable, create_scenes, manifest as base_manifest
from classifier_core import digest, require


def classifier_config():
    return yaml.safe_load((share('sorting_supervisor')/'config/m4b_classifier.yaml').read_text())


def policy_config():
    return yaml.safe_load((share('sorting_supervisor')/'config/m4b.yaml').read_text())


def model_directory():
    return share('sorting_supervisor')/'models/m4b_baseline'


def installed_manifest():
    result = base_manifest()
    result['m4b/render_dataset'] = digest(executable('sorting_supervisor', 'render_m4b_dataset'))
    for path in sorted(model_directory().glob('*')):
        if path.is_file():
            result['m4b/model/'+path.name] = digest(path)
    return result


def scenes(out):
    result = {}
    for condition in ('clear', 'partial', 'covered'):
        folder = out/'fixtures'/condition; folder.mkdir(parents=True)
        paths = create_scenes(folder)
        for label, path in paths.items():
            props_path = folder/f'props-{label}.xml'
            props = ET.parse(props_path); cover = props.find(".//geom[@name='fruit_front_occluder']")
            if condition == 'clear':
                cover.set('rgba', '.42 .42 .42 0')
            elif condition == 'covered':
                cover.set('pos', '-.04 .042 .055'); cover.set('size', '.040 .040 .002')
            props.write(props_path, encoding='utf-8', xml_declaration=True)
            result[label+'_'+condition] = str(path)
    return result


def generated_hashes(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted((folder/'fixtures').rglob('*.xml'))}


def checked_preflight(folder):
    folder = Path(folder).expanduser().resolve()
    r = json.loads((folder/'preflight.json').read_text())
    require(r.get('checkpoint') == 'M4B' and r.get('status') == 'PASS', 'Passing M4B preflight required')
    require(r['installed_manifest'] == installed_manifest(), 'Installed source or model changed; run a new M4B preflight')
    require(r['generated_hashes'] == generated_hashes(folder), 'Generated M4B scenes changed')
    return folder, r
