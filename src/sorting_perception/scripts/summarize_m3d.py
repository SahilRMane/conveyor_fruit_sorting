#!/usr/bin/env python3
"""Combine red/green ten-trial batches and two covered one-trial reports."""
import argparse,json
from pathlib import Path
from m3d_acceptance import assess

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',nargs=4,type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    values=[json.loads((f/'summary.json' if f.is_dir() else f).read_text()) for f in a.reports];report=assess(values);report['source_reports']=[str(f.resolve()) for f in a.reports]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
