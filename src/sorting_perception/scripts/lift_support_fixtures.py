"""Complete rigid tray fixtures and a camera/encoder-derived carried path."""
import copy,json
import numpy as np
from rclpy.serialization import serialize_message
from camera_scene import scene_for_camera,TRAY_IDS,SUPPORT_CONTACT_IDS,support_contact_state
from camera_core import quaternion_rotation
from publish_static_fixtures import scene_message

def hand_matrix(row):
    t=np.eye(4);q=row['hand_quat_wxyz'];t[:3,:3]=quaternion_rotation([q[1],q[2],q[3],q[0]]);t[:3,3]=row['hand_xyz'];return t

def permission(message,a,b,enabled):
    names=message.allowed_collision_matrix.entry_names
    message.allowed_collision_matrix.entry_values[names.index(a)].enabled[names.index(b)]=enabled
    message.allowed_collision_matrix.entry_values[names.index(b)].enabled[names.index(a)]=enabled

def write_fixtures(folder,settings):
    paths=[];template=scene_message()
    def save(label,message):
        path=folder/f'lift-support-{label}.cdr';path.write_bytes(serialize_message(message));paths.append(str(path))
    # Full eight-part payload, isolated from the robot, attached at a fixed root.
    # These kinematic fixtures reproduce old distance-check failures separately.
    for label,z in (('legacy-internal',.50),('legacy-base-only',.3059),('complete-contact',.3059),('complete-clear',.3311)):
        rel=np.eye(4);rel[:3,3]=[.85,0,z]
        message=scene_for_camera(template,[.85,0,0,0],settings,True,rel,label in {'legacy-base-only','complete-contact'})
        message.robot_state.joint_state.name=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
        message.robot_state.joint_state.position=[0.,0.,0.,-1.57079,0.,1.57079,-.7853,.04,.04]
        for attached in message.robot_state.attached_collision_objects:
            attached.link_name='panda_link0';attached.object.header.frame_id='panda_link0'
        if label=='legacy-internal':
            for a in TRAY_IDS:
                for b in TRAY_IDS:
                    if a!=b:permission(message,a,b,False)
        if label=='legacy-base-only':
            for part in SUPPORT_CONTACT_IDS-{'tray_base'}:permission(message,part,'support_surface',False)
        save(label,message)
    # Actual Panda poses along the native physical lift. No tray truth enters.
    data=json.loads((folder/'native-carried-kinematics.json').read_text());ref=data['reference']
    root=np.eye(4);root[:3,3]=ref['camera_root_xyz'];relative=np.linalg.inv(hand_matrix(ref))@root
    first=round(data['samples'][0]['time_sim_s']*1e9);cleared=False
    for i,row in enumerate(data['samples']):
        now=round(row['time_sim_s']*1e9);world=hand_matrix(row)@relative
        gap,cleared,allowed=support_contact_state(world,now,first,cleared,dict(owner='SERVO',busy=False),settings)
        message=scene_for_camera(template,[0,0,0,0],settings,True,relative,allowed)
        message.robot_state.joint_state.name=[f'panda_joint{j}' for j in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
        message.robot_state.joint_state.position=row['joint_positions']
        save(f'path-{i:03d}',message)
    return paths
