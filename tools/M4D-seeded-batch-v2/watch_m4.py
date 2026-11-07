#!/usr/bin/env python3
"""Read batch progress without joining the ROS graph."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',type=Path,required=True)
    args=parser.parse_args();previous=None;started=time.monotonic()
    print('Waiting for batch progress in '+str(args.batch_dir),flush=True)
    while True:
        try:report=json.loads((args.batch_dir/'progress.json').read_text())
        except FileNotFoundError:
            if time.monotonic()-started>60:raise TimeoutError('No batch progress file appeared within 60 seconds')
            time.sleep(1.);continue
        if time.time()-(args.batch_dir/'progress.json').stat().st_mtime>600 and report.get('status')=='RUNNING':
            raise TimeoutError('Batch progress exceeded the episode time bound')
        rows=report.get('cases',[])
        state=(report.get('status'),tuple((r['index'],r['status']) for r in rows))
        if state!=previous:
            for row in rows:
                old={} if previous is None else dict(previous[1])
                if old.get(row['index'])!=row['status']:
                    print(f"{row['index']:02d} {row['fixture']} {row['arrival']}: {row['status']}"+
                          (' — '+row['error'] if row.get('error') else ''),flush=True)
            previous=state
        if report.get('status')!='RUNNING':
            print(json.dumps({k:v for k,v in report.items() if k not in ('cases','batch_source_hashes')},indent=2))
            return 0 if report.get('status')=='PASS' else 1
        time.sleep(2.)


if __name__=='__main__':raise SystemExit(main())
