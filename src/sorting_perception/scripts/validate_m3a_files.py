#!/usr/bin/env python3
"""Native RGB, optical-frame and file validation. This creates no ROS node."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory,get_package_prefix
from camera_core import localize,quaternion_rotation,require

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,default=Path("results/M3A-preflight"))
    args=parser.parse_args();folder=args.output_dir.resolve();folder.mkdir(parents=True,exist_ok=True)
    report=dict(status="FAIL",live_test="NOT_RUN",camera_driven_grasp=False,oracle_controller=False)
    try:
        p=Path(get_package_share_directory("sorting_perception"));d=Path(get_package_share_directory("sorting_description"));s=Path(get_package_prefix("sorting_simulation"))
        c=yaml.safe_load((p/"config/m3a.yaml").read_text());scene=d/"mujoco/static_grasp/scene_camera.xml"
        subprocess.run([str(s/"lib/sorting_simulation/verify_camera_render"),str(scene),str(d/"mujoco/static_grasp/scene_tracking.xml"),str(p/"config/m3a.yaml"),str(folder)],check=True)
        native=yaml.safe_load((folder/"native-render.yaml").read_text());errors=[];frames=[]
        for f in native["frames"]:
            rgb=cv2.cvtColor(cv2.imread(str(folder/f["image"])),cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(folder/(f["label"]+".png")),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            focal=f["height"]/2/np.tan(np.deg2rad(f["fovy_deg"])/2)
            k=np.array([[focal,0,f["width"]/2],[0,focal,f["height"]/2],[0,0,1.]])
            result=localize(rgb,k,f["camera_position"],quaternion_rotation(f["camera_quat_xyzw"]),c)
            error=float(np.linalg.norm(np.asarray(result["position"])[:2]-np.asarray(f["handle_position"])[:2]));errors.append(error)
            require(error<c["maximum_planar_error_m"],"Native RGB planar localization exceeds 2 mm: "+f["label"])
            frames.append(dict(label=f["label"],view_id=f["view_id"],planar_error_m=error,**result))
        urdf=xacro.process_file(str(d/"urdf/panda_cameras.urdf.xacro"),mappings={"headless":"true","mujoco_model":str(scene)}).toxml()
        (folder/"panda-cameras.urdf").write_text(urdf)
        root=ET.parse(Path(get_package_share_directory("moveit_resources_panda_moveit_config"))/"config/panda.srdf").getroot()
        for joint in root.findall("virtual_joint"):root.remove(joint)
        ET.ElementTree(root).write(folder/"panda-cameras.srdf",encoding="unicode")
        exe=Path(get_package_prefix("sorting_control"))/"lib/sorting_control/verify_camera_model"
        subprocess.run([str(exe),str(folder/"panda-cameras.urdf"),str(folder/"panda-cameras.srdf"),str(scene),str(folder/"native-render.yaml"),str(folder/"camera-frame-check.json")],check=True)
        model=json.loads((folder/"camera-frame-check.json").read_text())
        subprocess.run([str(s/"lib/sorting_simulation/verify_camera_render"),str(scene),str(d/"mujoco/static_grasp/scene_tracking.xml"),str(p/"config/m3a.yaml"),str(folder),"--benchmark"],check=True)
        benchmark=yaml.safe_load((folder/"render-benchmark.yaml").read_text())["render_benchmark"]
        camera_config=yaml.safe_load((p/"config/cameras.yaml").read_text())
        period=1./camera_config["publish_rate_hz"]
        pair_p95=sum(v["p95_render_read_wall_s"] for v in benchmark["views"].values())
        render_budget=dict(benchmark,estimated_pair_p95_wall_s=pair_p95,nominal_period_sim_s=period,
                           conservative_wall_budget_s=period,within_nominal_wall_budget=pair_p95<period)
        write(folder/"render-budget.json",render_budget)
        require(pair_p95<period,"Native RGB render/read pair exceeds the nominal wall budget; see render-budget.json")
        write(folder/"camera-motion-plan.json",dict(inspection_joint_positions=native["inspection_joint_positions"],
              maximum_native_joint_speed_rad_s=native["maximum_calibration_joint_speed_rad_s"],
              purpose="Known camera calibration fixture; not an interception policy",
              config_sha256=hashlib.sha256((p/"config/cameras.yaml").read_bytes()+(p/"config/m3a.yaml").read_bytes()).hexdigest()))
        report.update(status="PASS",rgb_fixtures=len(frames),maximum_planar_error_m=max(errors),frames=frames,
                      camera_frames=model,physical_mass_and_inertia_preserved=native["physical_mass_and_inertia_preserved"],
                      render_budget=render_budget,
                      maximum_native_joint_speed_rad_s=native["maximum_calibration_joint_speed_rad_s"],
                      output_directory=str(folder),motion_plan=str(folder/"camera-motion-plan.json"))
    except Exception as error:report["error"]=str(error)
    write(folder/"validation.json",report);print(json.dumps(report,indent=2));return 0 if report["status"]=="PASS" else 1

if __name__=="__main__":raise SystemExit(main())
