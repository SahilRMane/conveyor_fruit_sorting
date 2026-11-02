#!/usr/bin/env python3
"""Generate grouped native RGB views; this starts no ROS node."""
import argparse,json,subprocess
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory,get_package_prefix
from m3c_data import prepare

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    perception=Path(get_package_share_directory('sorting_perception'));description=Path(get_package_share_directory('sorting_description'))/'mujoco/static_grasp'
    c=yaml.safe_load((perception/'config/m3c.yaml').read_text());localization=yaml.safe_load((perception/'config'/c['localization_config']).read_text())
    exe=Path(get_package_prefix('sorting_simulation'))/'lib/sorting_simulation/render_m3c_dataset'
    subprocess.run([str(exe),str(description/'scene_camera_m3c_red_clear.xml'),str(description/'scene_camera_m3b.xml'),str(perception/'config/m3c.yaml'),str(out/'raw')],check=True)
    result=prepare(out/'raw',out,c,localization);print(json.dumps(dict(status='PASS',frames=len(result['rows']),**result['split_audit'],paired_view_audit=result['paired_view_audit'],output_directory=str(out)),indent=2))
if __name__=='__main__':main()
