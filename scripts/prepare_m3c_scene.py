#!/usr/bin/env python3
"""Add class-bearing visual fruit and class-neutral occluders to owned scenes."""
from pathlib import Path
import copy,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
def save(tree,path):ET.indent(tree);tree.write(path,encoding='utf-8',xml_declaration=True)
def main():
    base=ROOT/'src/sorting_description/mujoco/static_grasp'
    robot=ET.parse(base/'panda_camera_m3b.xml');robot.getroot().find(".//camera[@name='wrist']").set('fovy','80.0');save(robot,base/'panda_camera_m3c.xml')
    for label,color in [('red',[.8,.08,.08,1]),('green',[.08,.8,.08,1])]:
        for covered in (False,True):
            key=label+('_covered' if covered else '_clear')
            props=ET.parse(base/'props_camera_m3b.xml');tray=props.getroot().find(".//body[@name='tray']")
            asset=ET.SubElement(props.getroot(),'asset')
            ET.SubElement(asset,'texture',name='fruit_instance_texture',type='2d',builtin='checker',rgb1='.38 .38 .38',rgb2='.5 .5 .5',width='32',height='32')
            ET.SubElement(asset,'material',name='fruit_neutral_material',texture='fruit_instance_texture',texuniform='false',rgba='1 1 1 1')
            common=dict(mass='0',contype='0',conaffinity='0',group='2')
            ET.SubElement(tray,'geom',name='fruit_visual_body',type='ellipsoid',pos='-.04 0 .045',size='.034 .034 .029',material='fruit_neutral_material',**common)
            ET.SubElement(tray,'geom',name='fruit_class_skin',type='box',pos='-.04 .036 .053',size='.021 .015 .001',quat='.9238795325 -.3826834324 0 0',rgba=' '.join(map(str,color)),**common)
            ET.SubElement(tray,'geom',name='fruit_overhead_shroud',type='box',pos='-.036 -.01 .080',size='.044 .040 .003',rgba='.45 .45 .45 1',**common)
            ET.SubElement(tray,'geom',name='fruit_front_occluder',type='box',pos='-.04 .042 .055',size='.040 .040 .002',quat='.9238795325 -.3826834324 0 0',rgba='.42 .42 .42 '+('1' if covered else '0'),**common)
            save(props,base/f'props_camera_m3c_{key}.xml')
            scene=ET.parse(base/'scene_camera_m3b.xml')
            for inc in scene.getroot().findall('include'):
                if inc.get('file')=='panda_camera_m3b.xml':inc.set('file','panda_camera_m3c.xml')
                if inc.get('file')=='props_camera_m3b.xml':inc.set('file',f'props_camera_m3c_{key}.xml')
            
            for light in scene.getroot().findall('.//light'):light.set('diffuse','.65 .65 .65')
            scene.getroot().find(".//camera[@name='overhead']").set('resolution','480 360')
            scene.getroot().find('visual/global').set('offwidth','480');scene.getroot().find('visual/global').set('offheight','360')
            scene.getroot().set('model','M3C synthetic visual fruit '+key);save(scene,base/f'scene_camera_m3c_{key}.xml')
    print('Prepared four M3C visual fixtures; M3A/M3B scenes unchanged')
if __name__=='__main__':main()
