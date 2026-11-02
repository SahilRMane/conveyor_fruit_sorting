#!/usr/bin/env python3
"""Create project-owned camera/marker files from the passed M2C scene."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
PERCEPTION=ROOT/"src/sorting_perception"
DESCRIPTION=ROOT/"src/sorting_description"

def numbers(values):return " ".join(f"{v:.16g}" for v in values)

def rotation(rpy):
    r,p,y=rpy;cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])

def save(tree,path):ET.indent(tree);tree.write(path,encoding="utf-8",xml_declaration=True)

def main():
    c=yaml.safe_load((PERCEPTION/"config/cameras.yaml").read_text());base=DESCRIPTION/"mujoco/static_grasp"
    robot=ET.parse(base/"panda_static.xml");props=ET.parse(base/"props.xml");scene=ET.parse(base/"scene_tracking.xml")
    for include in scene.getroot().findall("include"):
        if include.get("file")=="panda_static.xml":include.set("file","panda_camera.xml")
        if include.get("file")=="props.xml":include.set("file","props_camera.xml")
    for view,cam in c["cameras"].items():
        parent=scene.getroot().find("worldbody") if cam["mjcf_parent"]=="world" else robot.getroot().find(f".//body[@name='{cam['mjcf_parent']}']")
        if parent is None:raise ValueError("Missing camera parent")
        gl=rotation(cam["optical_rpy"])@np.diag([1,-1,-1])
        ET.SubElement(parent,"camera",name=view,mode="fixed",pos=numbers(cam["xyz"]),
                      xyaxes=numbers(np.r_[gl[:,0],gl[:,1]]),fovy=str(c["fovy_deg"]),
                      resolution=f"{c['width']} {c['height']}")
    marker=c["marker"];tray=props.getroot().find(".//body[@name='tray']")
    ET.SubElement(tray,"geom",name="neutral_localization_marker",type="box",pos=numbers(marker["local_xyz"]),
                  size=numbers(marker["half_size"]),rgba=numbers(marker["rgba"]),mass="0",contype="0",conaffinity="0",group="2")
    scene.getroot().set("model","Panda M3 camera calibration and image localization")
    quality=c.get("render_quality")
    if quality is not None:
        visual=scene.getroot().find("visual")
        element=visual.find("quality")
        if element is None:element=ET.SubElement(visual,"quality")
        element.set("offsamples",str(quality["offscreen_samples"]))
        element.set("shadowsize",str(quality["shadow_map_size"]))
        global_view=visual.find("global")
        global_view.set("offwidth",str(c["width"]));global_view.set("offheight",str(c["height"]))
        for tree in (scene,robot,props):
            for light in tree.getroot().findall(".//light"):
                light.set("castshadow","true" if quality["cast_shadows"] else "false")
    save(robot,base/"panda_camera.xml");save(props,base/"props_camera.xml");save(scene,base/"scene_camera.xml")
    plugins=yaml.safe_load((ROOT/"src/sorting_simulation/config/conveyor_plugins.yaml").read_text())
    entries=plugins["/**"]["ros__parameters"]["mujoco_plugins"]
    camera={"type":"mujoco_ros2_control_plugins/CameraPlugin","camera_publish_rate":float(c["publish_rate_hz"]),
            "static_overview":{"policy":"disabled"}}
    for view,cam in c["cameras"].items():
        camera[view]={"policy":"streaming","frame_name":cam["frame"],"info_topic":f"/camera/{view}/camera_info",
                      "image_topic":f"/camera/{view}/image_raw","depth_topic":f"/camera/{view}/depth_unused"}
    entries["camera_plugin"]=camera
    entries["camera_evidence"]={"type":"sorting_simulation/CameraEvidence","enable_evaluator":False}
    (ROOT/"src/sorting_simulation/config/camera_plugins.yaml").write_text(yaml.safe_dump(plugins,sort_keys=False))
    print("Prepared scene_camera.xml and camera_plugins.yaml from cameras.yaml")

if __name__=="__main__":main()
