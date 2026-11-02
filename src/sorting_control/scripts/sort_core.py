"""Class routing and estimated placement guards; no simulator state inputs."""
import numpy as np
from ownership_core import require
from camera_scene import TRAY_IDS,SUPPORT_CONTACT_IDS,scene_for_camera

def choose_route(class_ids,p,c):
    require(list(class_ids)==c['class_ids'],'Class order mismatch')
    p=np.asarray(p,float);require(p.shape==(len(class_ids),) and np.isfinite(p).all() and np.all(p>=0) and abs(p.sum()-1)<1e-6,'Invalid class probabilities')
    i=int(p.argmax());confidence=float(p[i])
    if confidence<c['minimum_confidence']:return dict(action='PASS',class_id='',bin_id='',confidence=confidence,probabilities=p.tolist(),reason='low_confidence')
    name=class_ids[i];return dict(action='PICK_PLACE',class_id=name,bin_id=c['class_to_bin'][name],confidence=confidence,probabilities=p.tolist(),reason='confident_fixed_view')
def footprint_inside_estimate(world,bin_id,c):
    b=c['bins'][bin_id];half=np.array([.12,.09,.006]);corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half
    points=corners@world[:3,:3].T+world[:3,3]
    return bool(np.all(np.abs(points[:,:2]-np.array(b['center'])[:2])<np.array(b['half_size'])-c['footprint_margin_m']))
def resting_pose_from_release(world,bin_id,c):
    """Known floor-resting prior after opening; no simulator state is read."""
    world=np.asarray(world,dtype=float)
    require(world.shape==(4,4) and np.isfinite(world).all(),'Invalid release pose')
    require(footprint_inside_estimate(world,bin_id,c),'Estimated release footprint is outside the bin')
    gap=world[2,3]-c['placement_root_z_m']
    require(0<gap<=c['maximum_release_gap_m'] and
            abs(world[2,3]-c['release_root_z_m'])<c['placement_position_tolerance_m'],
            'Estimated release height outside bound')
    placed=world.copy();yaw=float(np.arctan2(world[1,0],world[0,0]));co,si=np.cos(yaw),np.sin(yaw)
    placed[:3,:3]=[[co,-si,0],[si,co,0],[0,0,1]];placed[2,3]=c['placement_root_z_m']
    return placed
def placement_scene(template,target,base_settings,allow,relative,source_contact,floor_id='',placed=None):
    if placed is not None:
        message=scene_for_camera(template,[placed[0,3],placed[1,3],0,0],base_settings,allow)
        from scipy.spatial.transform import Rotation
        q=Rotation.from_matrix(placed[:3,:3]).as_quat()
        for obj in message.world.collision_objects:
            if obj.id in TRAY_IDS:
                nominal=np.array([obj.primitive_poses[0].position.x,obj.primitive_poses[0].position.y,obj.primitive_poses[0].position.z])-np.array([placed[0,3],placed[1,3],base_settings['fixed_handle_z_m']-.117])
                p=placed[:3,:3]@nominal+placed[:3,3];pose=obj.primitive_poses[0];pose.position.x,pose.position.y,pose.position.z=map(float,p);pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,q)
        return message
    message=scene_for_camera(template,target,base_settings,allow,relative,source_contact)
    if floor_id:
        require(relative is not None and floor_id in ('bin_a_floor','bin_b_floor'),'Destination allowance requires carried geometry')
        names=message.allowed_collision_matrix.entry_names
        if floor_id not in names:
            from moveit_msgs.msg import AllowedCollisionEntry
            for row in message.allowed_collision_matrix.entry_values:row.enabled.append(False)
            names.append(floor_id);message.allowed_collision_matrix.entry_values.append(AllowedCollisionEntry(enabled=[False]*(len(names)-1)+[True]))
        for part in SUPPORT_CONTACT_IDS:
            i,j=names.index(part),names.index(floor_id);message.allowed_collision_matrix.entry_values[i].enabled[j]=True;message.allowed_collision_matrix.entry_values[j].enabled[i]=True
    return message
class ClassWindow:
    def __init__(self,c):self.c=c;self.samples={};self.label=None;self.first=None
    def observe(self,capture,route):
        label=route['class_id'] if route['action']=='PICK_PLACE' else 'PASS'
        if self.label!=label:self.samples.clear();self.first=None;self.label=label
        if capture not in self.samples:self.samples[capture]=route
        if self.first is None:self.first=capture
        return len(self.samples)>=self.c['minimum_class_captures'] and (capture-self.first)/1e9>=self.c['minimum_class_span_s']

def supported_floor_pose(world,bin_id,c):
    """Known floor plane after physical wrist support, without simulator pose."""
    world=np.asarray(world,float)
    require(world.shape==(4,4) and np.isfinite(world).all(),'Invalid supported pose')
    require(footprint_inside_estimate(world,bin_id,c),'Supported footprint outside destination')
    limit=c['supported_release']['maximum_model_height_correction_m']
    require(0<limit<=c['maximum_release_gap_m'],'Support correction exceeds the existing model bound')
    require(abs(world[2,3]-c['placement_root_z_m'])<limit,'Supported height correction exceeds bound')
    out=world.copy();yaw=float(np.arctan2(world[1,0],world[0,0]));co,si=np.cos(yaw),np.sin(yaw)
    out[:3,:3]=[[co,-si,0],[si,co,0],[0,0,1]];out[2,3]=c['placement_root_z_m']
    return out
