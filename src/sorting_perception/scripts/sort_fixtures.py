"""Serialize production scenes at encoder poses; no tray truth in geometry."""
import json,numpy as np,yaml
from rclpy.serialization import serialize_message
from camera_scene import scene_for_camera
from sort_core import placement_scene
from lift_support_fixtures import hand_matrix
from publish_static_fixtures import scene_message

def write_sort_fixtures(folder,control):
    data=json.loads((folder/'sort-kinematics.json').read_text());settings=yaml.safe_load((control/'config/m3b.yaml').read_text());template=scene_message();relative=np.asarray(data['relative']);paths=[]
    for i,row in enumerate(data['samples']):
        phase=row['phase'];carried=phase in ('transfer_retract','transfer_raise','transfer','lower','release');floor=data['bin_id']+'_floor' if phase in ('lower','release') else ''
        allow=carried or (phase=='retreat' and hand_matrix(row)[2,3]-np.asarray(data['placed_matrix'])[2,3]<.2199+.035)
        msg=placement_scene(template,[0,0,0,0],settings,allow,relative if carried else None,False,floor,np.asarray(data['placed_matrix']) if not carried else None)
        msg.robot_state.joint_state.name=[f'panda_joint{j}' for j in range(1,8)]+['panda_finger_joint1','panda_finger_joint2'];msg.robot_state.joint_state.position=row['joint_positions']
        label='carry' if carried else 'world';path=folder/f'{i:04d}-{phase}-{label}.cdr';path.write_bytes(serialize_message(msg));paths.append(str(path))
    return paths
