#!/usr/bin/env python3
"""Bind the frozen M5C selector and stage the visible M5D scenes."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix

sys.path.insert(0, str(Path(get_package_prefix('sorting_supervisor')) / 'lib/sorting_supervisor'))
from m4c_common import config
from m4d_common import checked_preflight


CASES = {
    'left': ('red_partial', 'upstream', 'inspect_left', 'red_fruit'),
    'right': ('green_clear', 'upstream', 'inspect_right', 'green_fruit'),
    'late': ('red_partial', 'late', None, None),
}
COVER = dict(name='m5c_right_cover', type='box', pos='-.050 .062 .183', size='.043 .015 .002',
             rgba='.42 .42 .42 1', mass='0', contype='0', conaffinity='0', group='2')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build_right(scene, cover, folder):
    folder.mkdir()
    tree = ET.parse(scene)
    include = next(e for e in tree.findall('include') if Path(e.attrib['file']).name == 'props-green.xml')
    props = ET.parse(include.attrib['file'])
    tray = props.find(".//body[@name='tray']")
    tray.append(deepcopy(cover))
    props_path = folder / 'props-green.xml'
    props.write(props_path, encoding='utf-8', xml_declaration=True)
    include.set('file', str(props_path))
    tree.getroot().set('model', 'M5D right-reveal development fixture')
    path = folder / 'scene-green.xml'
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return path, props_path


def prepare(args):
    preflight, frozen = checked_preflight(args.preflight_dir)
    preflight_sha = sha(preflight / 'preflight.json')
    inspection = read(preflight / 'inspection/preflight.json')
    m5c = args.m5c_dir
    report, manifest = read(m5c / 'report.json'), read(m5c / 'manifest.json')
    weights = m5c / 'view-utility.npz'
    require(report['checkpoint'] == 'M5C' and report['status'] == 'PASS' and report['test_cases'] == 192
            and not report['test_used_for_training'] and report['manifest_sha256'] == sha(m5c / 'manifest.json')
            and report['selector_sha256'] == sha(weights) and manifest['model_sha256'] == frozen['model_sha256'],
            'Frozen M5C evidence differs')
    utility = Path(__file__).with_name('utility_core.py')
    require(manifest['source_hashes']['utility_core.py'] == sha(utility), 'M5C utility code differs')
    native = preflight / 'inspection/m4b/m4a/native/views.yaml'
    require(sha(native) == manifest['poses_sha256'], 'M5C wrist pose file differs')
    cover = ET.parse(m5c / 'test/props-m5c-right.xml').find(".//geom[@name='m5c_right_cover']")
    require(cover is not None and cover.attrib == COVER, 'M5C right cover differs')
    helper = args.m5b_code / 'probe_m5b.py'
    measured = {'inspect_left': [], 'inspect_right': []}
    for source in manifest['m5b'].values():
        require(sha(source['report']) == source['sha256'], 'M5B report differs')
        prior = read(source['report'])
        require(prior['status'] == 'PASS' and prior['preflight_sha256'] == preflight_sha
                and prior['probe_sha256'] == sha(helper), 'M5B probe or preflight differs')
        measured['inspect_left'].append(prior['left_reference']['costs'])
        measured['inspect_right'].append(prior['right_live']['costs'])
    settings = config()
    budgets = {}
    for view, samples in measured.items():
        budgets[view] = {key: max(s[field] for s in samples) * settings['cost_scale'] + settings['cost_addition_s']
                         for key, field in (('travel', 'travel_and_settle_sim_s'), ('return', 'return_and_settle_sim_s'))}
        budgets[view]['total'] = max(manifest['view_costs_sim_s'][view],
                                     sum(budgets[view].values()) + settings['capture_budget_s'])
    require(all(math.isfinite(v) and 0 < v < 45 for d in budgets.values() for v in d.values()),
            'Invalid inspection allocation')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    cases = {}
    for name, (fixture, arrival, view, class_id) in CASES.items():
        path = Path(inspection['scene_paths'][fixture + ':' + arrival])
        props = None
        if name == 'right':
            path, props = build_right(path, cover, out / 'right-scene')
        files = {str(p): sha(p) for p in (path, props) if p is not None}
        cases[name] = dict(fixture=fixture, arrival=arrival, scene=str(path), files=files,
                           expected_view=view, expected_class=class_id)
    result = dict(checkpoint='M5D', preflight=str(preflight), preflight_sha256=preflight_sha,
                  selector=str(weights), selector_sha256=sha(weights), utility_sha256=sha(utility),
                  m5b_helper=str(helper), m5b_helper_sha256=sha(helper), model_sha256=frozen['model_sha256'],
                  poses=str(native), poses_sha256=sha(native), view_budgets=budgets, cases=cases)
    (out / 'suite.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return out


def load_case(suite, name):
    data = read(suite / 'suite.json')
    require(data['checkpoint'] == 'M5D', 'M5D suite required')
    folder, frozen = checked_preflight(data['preflight'])
    case = data['cases'][name]
    require(sha(folder / 'preflight.json') == data['preflight_sha256']
            and frozen['model_sha256'] == data['model_sha256']
            and all(sha(path) == digest for path, digest in case['files'].items()), 'M5D scene or preflight differs')
    return data, case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('--preflight-dir', '--m5c-dir', '--m5b-code', '--output-dir'):
        p.add_argument(option, type=lambda path: Path(path).expanduser().resolve(), required=True)
    print(prepare(p.parse_args()))


if __name__ == '__main__':
    main()
