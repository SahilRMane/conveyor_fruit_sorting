"""Installed M4C configuration, source binding, scenes and measured M4B costs."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
from m4b_common import installed_manifest as base_manifest, scenes as baseline_scenes
from m4a_common import share, executable
from classifier_core import digest, require


def config():
    return yaml.safe_load((share('sorting_supervisor')/'config/m4c.yaml').read_text())


def manifest():
    result = base_manifest()
    result['m4c/native_probe'] = digest(executable('sorting_supervisor', 'probe_m4c_scene'))
    result['m4c/motion_filter'] = digest(executable('sorting_perception', 'motion_filter.py'))
    result['m4c/localizer'] = digest(executable('sorting_perception', 'handle_localizer.py'))
    result['m4c/filter_core'] = digest(executable('sorting_perception', 'filter_core.py'))
    return result


def scenes(out):
    c = config(); paths = {}
    for arrival, x in [('upstream', c['upstream_x_m']), ('late', c['late_x_m'])]:
        folder = out/'fixtures'/arrival; folder.mkdir(parents=True)
        entries = baseline_scenes(folder)
        for fixture, name in entries.items():
            path = Path(name); panda = path.parent/'panda.xml'
            scene = ET.parse(path)
            scene.find("./worldbody/camera[@name='overhead']").set('resolution', ' '.join(map(str,c['overhead_resolution'])))
            scene.find("./asset/material[@name='static_floor_material']").set('reflectance', '0')
            scene.write(path, encoding='utf-8', xml_declaration=True)
            tree = ET.parse(panda)
            key = tree.find(".//key[@name='home']")
            values = key.get('qpos').split(); values[9] = str(x); key.set('qpos', ' '.join(values))
            tree.write(panda, encoding='utf-8', xml_declaration=True)
            props = path.parent/('props-'+fixture.split('_')[0]+'.xml')
            tree = ET.parse(props); tree.find(".//body[@name='tray']").set('pos', f'{x} 0 .307')
            tree.write(props, encoding='utf-8', xml_declaration=True)
            paths[fixture+':'+arrival] = str(path)
    return paths


def generated_hashes(out):
    files = list((out/'fixtures').rglob('*.xml'))+[out/'cameras.yaml', out/'filter.yaml', out/'camera_plugins.yaml', out/'costs.json']
    return {str(p.relative_to(out)): digest(p) for p in sorted(files)}


def measured_costs(results, model_sha):
    """Read complete live M4B evidence. Do not infer costs from nominal loop rates."""
    results = Path(results).expanduser().resolve(); chosen = {}
    for path in sorted(results.glob('M4B-*/summary.json'), key=lambda p: p.stat().st_mtime):
        s = json.loads(path.read_text()); fixture = s.get('fixture')
        if (s.get('checkpoint') == 'M4B' and s.get('status') == 'PASS' and s.get('acceptance_passed')
                and s.get('final_hold_status') == 'PASS' and s.get('final_owner') == 'HOLD'
                and s.get('overlap_violations') == 0 and s.get('model_sha256') == model_sha):
            chosen[fixture] = (path, s)
    wanted = {label+'_'+cover for label in ('red','green') for cover in ('clear','partial','covered')}
    require(wanted <= chosen.keys(), 'Six passing live M4B fixture reports are required')
    durations = {'initial_assess': [], 'inspect_left': [], 'return_assess': []}; sources = []
    for fixture in sorted(wanted):
        path, summary = chosen[fixture]; events_path = path.parent/'events.json'
        events = json.loads(events_path.read_text()); pending = None; completed = []
        for event in events:
            if event['state'] == 'PLAN':
                require(pending is None, 'M4B motion events overlap'); pending = event
            elif event['state'] == 'SETTLED':
                require(pending is not None and pending['pose'] == event['pose'], 'Incomplete M4B motion evidence')
                elapsed = (event['enter_ns']-pending['enter_ns'])/1e9
                require(0 < elapsed < 45, 'Invalid M4B motion duration')
                completed.append((event['pose'], elapsed)); pending = None
        expected = ['assess'] if fixture.endswith('_clear') else ['assess', 'inspect_left', 'assess']
        require(pending is None and [v[0] for v in completed] == expected, 'M4B motion evidence does not match fixture')
        require(summary.get('checked_samples') == (5 if len(expected)==1 else 10), 'Incomplete M4B captures')
        durations['initial_assess'].append(completed[0][1])
        if len(completed)>1:
            durations['inspect_left'].append(completed[1][1]); durations['return_assess'].append(completed[2][1])
        sources.append(dict(fixture=fixture, summary=str(path), summary_sha256=digest(path),
                            events_sha256=digest(events_path), installed_manifest=summary['installed_manifest']))
    c = config()
    return dict(scope='Six live M4B pilots; conservative allocations, not timing guarantees',
                measured_maximum_sim_s={k:max(v) for k,v in durations.items()},
                upper_sim_s={k:max(v)*c['cost_scale']+c['cost_addition_s'] for k,v in durations.items()},
                model_sha256=model_sha, sources=sources,
                physical_pickup_timing_validated=False, approach_allocation_s=c['approach_allocation_s'])


def checked_preflight(folder):
    folder = Path(folder).expanduser().resolve()
    r = json.loads((folder/'preflight.json').read_text())
    require(r.get('checkpoint') == 'M4C' and r.get('status') == 'PASS', 'A passing M4C preflight is required')
    require(r['installed_manifest'] == manifest(), 'Installed source changed; run a new M4C preflight')
    require(r['generated_hashes'] == generated_hashes(folder), 'M4C scene, camera, filter or cost files changed')
    return folder, r
