#!/usr/bin/env python3
"""Native RGB, optical-frame and file validation. This creates no ROS node."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory,get_package_prefix
from m3b_manifest import installed_manifest
from camera_core import localize,quaternion_rotation,require

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,default=Path("results/M3B-preflight"))
    args=parser.parse_args();folder=args.output_dir.resolve();folder.mkdir(parents=True,exist_ok=True)
    report=dict(status="FAIL",live_test="NOT_RUN",camera_driven_grasp="native only; live NOT_RUN",oracle_controller=False)
    try:
        p=Path(get_package_share_directory("sorting_perception"));d=Path(get_package_share_directory("sorting_description"));s=Path(get_package_prefix("sorting_simulation"))
        sys.path.insert(0,str(Path(get_package_prefix("sorting_control"))/"lib/sorting_control"))
        from native_camera_pickup import run
        c=yaml.safe_load((p/"config/m3b_localization.yaml").read_text())
        filter_config=yaml.safe_load((p/"config/m3b_filter.yaml").read_text())
        control_config=yaml.safe_load((Path(get_package_share_directory("sorting_control"))/"config/m3b.yaml").read_text())
        require(abs(filter_config["acceleration_sigma_m_s2"]-control_config["acceleration_sigma_m_s2"])<1e-12,"Filter and controller acceleration noise differ")
        require(abs(filter_config["fixed_handle_z_m"]-control_config["fixed_handle_z_m"])<1e-12 and abs(c["handle_plane_z_m"]-filter_config["fixed_handle_z_m"])<1e-12,"Handle height priors differ")
        scene=d/"mujoco/static_grasp/scene_camera_m3b.xml"
        subprocess.run([str(s/"lib/sorting_simulation/verify_camera_render"),str(scene),str(d/"mujoco/static_grasp/scene_tracking.xml"),str(p/"config/m3b_localization.yaml"),str(folder)],check=True)
        native=yaml.safe_load((folder/"native-render.yaml").read_text());errors=[];frames=[]
        for f in native["frames"]:
            rgb=cv2.cvtColor(cv2.imread(str(folder/f["image"])),cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(folder/(f["label"]+".png")),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            focal=f["height"]/2/np.tan(np.deg2rad(f["fovy_deg"])/2)
            k=np.array([[focal,0,f["width"]/2],[0,focal,f["height"]/2],[0,0,1.]])
            try:result=localize(rgb,k,f["camera_position"],quaternion_rotation(f["camera_quat_xyzw"]),c)
            except ValueError as error:
                if f["view_id"]=="wrist" and str(error) in {"marker_not_found","marker_at_image_border"}:
                    frames.append(dict(label=f["label"],view_id=f["view_id"],localization="NOT_VISIBLE",camera_fk_still_checked=True));continue
                raise
            error=float(np.linalg.norm(np.asarray(result["position"])[:2]-np.asarray(f["handle_position"])[:2]));errors.append(error)
            require(error<c["maximum_planar_error_m"],"Native RGB planar localization exceeds 2 mm: "+f["label"])
            frames.append(dict(label=f["label"],view_id=f["view_id"],planar_error_m=error,**result))
        urdf=xacro.process_file(str(d/"urdf/panda_cameras.urdf.xacro"),mappings={"headless":"true","camera_config":"cameras_m3b.yaml","mujoco_model":str(scene)}).toxml()
        (folder/"panda-cameras.urdf").write_text(urdf)
        root=ET.parse(Path(get_package_share_directory("moveit_resources_panda_moveit_config"))/"config/panda.srdf").getroot()
        for joint in root.findall("virtual_joint"):root.remove(joint)
        ET.ElementTree(root).write(folder/"panda-cameras.srdf",encoding="unicode")
        exe=Path(get_package_prefix("sorting_control"))/"lib/sorting_control/verify_camera_model"
        subprocess.run([str(exe),str(folder/"panda-cameras.urdf"),str(folder/"panda-cameras.srdf"),str(scene),str(folder/"native-render.yaml"),str(folder/"camera-frame-check.json")],check=True)
        model=json.loads((folder/"camera-frame-check.json").read_text())
        subprocess.run([str(s/"lib/sorting_simulation/verify_camera_render"),str(scene),str(d/"mujoco/static_grasp/scene_tracking.xml"),str(p/"config/m3b_localization.yaml"),str(folder),"--benchmark"],check=True)
        benchmark=yaml.safe_load((folder/"render-benchmark.yaml").read_text())["render_benchmark"]
        camera_config=yaml.safe_load((p/"config/cameras_m3b.yaml").read_text())
        period=1./camera_config["publish_rate_hz"]
        active_views=[view for view,cam in camera_config["cameras"].items() if cam.get("publish_policy","streaming")=="streaming"]
        require(active_views==[filter_config["tracking_view"]],"Prepositioned M3B must stream only its selected wrist tracking view")
        plugin_config=yaml.safe_load((Path(get_package_share_directory("sorting_simulation"))/"config/camera_plugins_m3b.yaml").read_text())["/**"]["ros__parameters"]["mujoco_plugins"]["camera_plugin"]
        require(all(plugin_config[view]["policy"]==cam.get("publish_policy","streaming") for view,cam in camera_config["cameras"].items()),"Camera config and installed plugin streaming policies differ")
        pair_p95=sum(v["p95_render_read_wall_s"] for v in benchmark["views"].values())
        active_p95=sum(benchmark["views"][view]["p95_render_read_wall_s"] for view in active_views)
        render_budget=dict(benchmark,estimated_pair_p95_wall_s=pair_p95,active_views=active_views,
                           active_render_p95_wall_s=active_p95,nominal_period_sim_s=period,
                           conservative_wall_budget_s=period,within_nominal_wall_budget=active_p95<period,
                           scope="Both calibration views benchmarked; only the streaming wrist view contributes to M3B runtime load")
        write(folder/"render-budget.json",render_budget)
        require(active_p95<period,"Active M3B RGB render/read exceeds the nominal wall budget; see render-budget.json")
        write(folder/"camera-motion-plan.json",dict(inspection_joint_positions=native["inspection_joint_positions"],
              maximum_native_joint_speed_rad_s=native["maximum_calibration_joint_speed_rad_s"],
              purpose="Known camera calibration fixture; not an interception policy",
              config_sha256=hashlib.sha256((p/"config/cameras_m3b.yaml").read_bytes()+(p/"config/m3b_localization.yaml").read_bytes()).hexdigest()))
        # Replay the real serialized Python scene messages through installed MoveIt.
        from rclpy.serialization import serialize_message
        from camera_scene import scene_for_camera,lightweight_camera_update,TRAY_IDS
        from publish_static_fixtures import scene_message
        from moveit_msgs.msg import PlanningScene,AttachedCollisionObject,CollisionObject
        template=scene_message();detach=PlanningScene(is_diff=True);detach.robot_state.is_diff=True
        for name in TRAY_IDS:
            obj=AttachedCollisionObject(link_name="panda_hand");obj.object.id=name;obj.object.operation=CollisionObject.REMOVE
            detach.robot_state.attached_collision_objects.append(obj)
        home=scene_for_camera(template,[.55,0,0,0],control_config,False)
        fixture_messages=[home,lightweight_camera_update(template,[.62,0,0,0],control_config,False),scene_for_camera(template,[.6,0,0,0],control_config,True),
            scene_for_camera(template,[.6,0,0,0],control_config,True,np.eye(4)),detach,home]
        paths=[]
        for i,message in enumerate(fixture_messages):
            path=folder/f"scene-{i}.cdr";path.write_bytes(serialize_message(message));paths.append(str(path))
        checker=Path(get_package_prefix("sorting_control"))/"lib/sorting_control/verify_camera_scene"
        with (folder/"scene-replay.txt").open("w") as log:
            subprocess.run([str(checker),str(folder/"panda-cameras.urdf"),str(folder/"panda-cameras.srdf"),*paths],stdout=log,stderr=subprocess.STDOUT,check=True)
        pickup=run(s/"lib/libnative_camera_plant.so",scene,p,Path(get_package_share_directory("sorting_control")),folder)
        from lift_support_fixtures import write_fixtures
        support_paths=write_fixtures(folder,control_config)
        support_checker=Path(get_package_prefix("sorting_control"))/"lib/sorting_control/verify_lift_support"
        with (folder/"lift-support-collision.txt").open("w") as log:
            subprocess.run([str(support_checker),str(folder/"panda-cameras.urdf"),str(folder/"panda-cameras.srdf"),*support_paths,str(folder/"lift-support-collision.json")],stdout=log,stderr=subprocess.STDOUT,check=True)
        support_report=json.loads((folder/"lift-support-collision.json").read_text())
        report.update(status="PASS",checkpoint="M3B",installed_manifest=installed_manifest(),native_camera_pickup=pickup,native_scene_replay="PASS",native_lift_support_collision=support_report,rgb_fixtures=len(frames),maximum_planar_error_m=max(errors),frames=frames,
                      camera_frames=model,physical_mass_and_inertia_preserved=native["physical_mass_and_inertia_preserved"],
                      render_budget=render_budget,
                      maximum_native_joint_speed_rad_s=native["maximum_calibration_joint_speed_rad_s"],
                      output_directory=str(folder),motion_plan=str(folder/"camera-motion-plan.json"))
    except Exception as error:report["error"]=str(error)
    write(folder/"validation.json",report);print(json.dumps(report,indent=2));return 0 if report["status"]=="PASS" else 1

if __name__=="__main__":raise SystemExit(main())
