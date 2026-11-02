#!/usr/bin/env python3
"""Add a physical wrist sensor to M3D-owned copies; preserve M3C scenes."""
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
def save(tree,path):
    ET.indent(tree);tree.write(path,encoding='utf-8',xml_declaration=True)
def main():
    base=ROOT/'src/sorting_description/mujoco/static_grasp'
    panda=ET.parse(base/'panda_camera_m3c.xml')
    ET.SubElement(panda.getroot().find(".//body[@name='hand']"),'site',name='panda_hand_wrench_site',pos='0 0 0',size='.001',rgba='0 0 0 0')
    sensors=panda.getroot().find('sensor')
    if sensors is None:sensors=ET.SubElement(panda.getroot(),'sensor')
    ET.SubElement(sensors,'force',name='panda_hand_wrench_force',site='panda_hand_wrench_site')
    ET.SubElement(sensors,'torque',name='panda_hand_wrench_torque',site='panda_hand_wrench_site')
    save(panda,base/'panda_camera_m3d.xml')
    for fixture in ('red_clear','green_clear','red_covered','green_covered'):
        scene=ET.parse(base/f'scene_camera_m3c_{fixture}.xml')
        for inc in scene.getroot().findall('include'):
            if inc.get('file')=='panda_camera_m3c.xml':inc.set('file','panda_camera_m3d.xml')
        scene.getroot().set('model','M3D wrist-sensed placement '+fixture)
        save(scene,base/f'scene_camera_m3d_{fixture}.xml')
if __name__=='__main__':main()
