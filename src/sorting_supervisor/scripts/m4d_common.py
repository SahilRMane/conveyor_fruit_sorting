"""M4D preflight, source binding and historical prerequisite reports."""
import json
from pathlib import Path
import yaml
from m4a_common import share
from m4c_common import manifest
from classifier_core import digest, require


def config():
    return yaml.safe_load((share('sorting_supervisor')/'config/m4d.yaml').read_text())


def prerequisite_reports(root):
    rows = []
    for color in ('red','green'):
        candidates = []
        for path in Path(root).expanduser().resolve().glob('M4D1-'+color+'-handoff-*/summary.json'):
            report=json.loads(path.read_text())
            candidates.append((path,report))
        require(bool(candidates),'Missing M4D.1 '+color+' handoff report')
        path, report=max(candidates,key=lambda row:row[0].stat().st_mtime_ns)
        require(report.get('status')=='PASS' and report.get('handoff_status')=='PASS'
                and report.get('timing_status')=='PASS' and report.get('final_hold_status')=='PASS'
                and report.get('overlap_violations')==0 and not report.get('rejected_handoffs'),
                'Latest M4D.1 '+color+' report did not pass')
        checks=report.get('handoff_checks',[])
        require(len(checks)==1 and checks[0]['class_id']==color+'_fruit'
                and checks[0]['proposed_bin']==('bin_a' if color=='red' else 'bin_b'), 'M4D.1 route differs')
        rows.append(dict(color=color,path=str(path),sha256=digest(path),
                         historical_installed_manifest=report['installed_manifest'],
                         model_sha256=checks[0]['model_sha256']))
    require(rows[0]['historical_installed_manifest']==rows[1]['historical_installed_manifest'],
            'M4D.1 reports used different source versions')
    return rows


def checked_preflight(folder):
    from m4c_common import checked_preflight as checked_inspection
    folder=Path(folder).expanduser().resolve()
    report=json.loads((folder/'preflight.json').read_text())
    require(report.get('checkpoint')=='M4D.2' and report.get('status')=='PASS','A passing M4D.2 preflight is required')
    require(report['installed_manifest']==manifest(),'Installed files or rendering selection changed after M4D preflight')
    for row in report['prerequisites']:
        require(digest(row['path'])==row['sha256'],'Historical handoff evidence changed')
    checked_inspection(folder/'inspection')
    return folder,report
