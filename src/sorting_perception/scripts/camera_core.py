"""RGB marker detection and capture-time calibrated ray/plane localization."""
import math
import cv2
import numpy as np

def require(ok,reason):
    if not ok:raise ValueError(reason)

def quaternion_rotation(xyzw):
    q=np.asarray(xyzw,dtype=float);require(q.shape==(4,) and np.all(np.isfinite(q)),"invalid_quaternion")
    length=np.linalg.norm(q);require(length>1e-12,"invalid_quaternion");x,y,z,w=q/length
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def image_rgb(message):
    require(message.encoding in {"rgb8","bgr8"},"unsupported_image_encoding")
    require(message.height>0 and message.width>0 and message.step>=message.width*3,"invalid_image_shape")
    data=np.frombuffer(bytes(message.data),dtype=np.uint8)
    require(data.size==message.height*message.step,"invalid_image_buffer")
    rgb=data.reshape(message.height,message.step)[:,:message.width*3].reshape(message.height,message.width,3).copy()
    return rgb[:,:,::-1].copy() if message.encoding=="bgr8" else rgb

def intrinsics(info,shape,frame):
    require(info.width==shape[1] and info.height==shape[0] and info.header.frame_id==frame,"camera_info_mismatch")
    require(info.distortion_model=="plumb_bob" and all(abs(v)<1e-12 for v in info.d),"unsupported_distortion")
    k=np.asarray(info.k,dtype=float).reshape(3,3)
    require(np.all(np.isfinite(k)) and k[0,0]>0 and k[1,1]>0 and np.allclose(k[2],[0,0,1]) and
            abs(k[0,1])+abs(k[1,0])<1e-12,"invalid_intrinsics")
    return k

def detect_marker(rgb,c):
    require(rgb.ndim==3 and rgb.shape[2]==3 and rgb.dtype==np.uint8,"invalid_rgb")
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.asarray(c["hsv_lower"],np.uint8),np.asarray(c["hsv_upper"],np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=float(cv2.contourArea(contour))
        if area<c["minimum_marker_area_px"]:continue
        candidates.append((contour,area))
    require(len(candidates)==1,"marker_not_found" if not candidates else "ambiguous_markers")
    contour,area=candidates[0];x,y,w,h=cv2.boundingRect(contour);margin=c["border_margin_px"]
    require(x>=margin and y>=margin and x+w<=rgb.shape[1]-margin and y+h<=rgb.shape[0]-margin,"marker_at_image_border")
    require(area<rgb.shape[0]*rgb.shape[1]*c["maximum_marker_area_fraction"],"marker_area_outside_bound")
    rect=cv2.minAreaRect(contour);rectangle_area=rect[1][0]*rect[1][1]
    rectangularity=area/max(rectangle_area,1e-12)
    require(rectangularity>=c["minimum_rectangularity"],"marker_occluded_or_irregular")
    moments=cv2.moments(contour);offset=c["pixel_center_offset"]
    return dict(pixel_u=moments["m10"]/moments["m00"]+offset,
                pixel_v=moments["m01"]/moments["m00"]+offset,area_px=area,quality=min(1.,rectangularity))

def ray_plane(pixel,k,translation,rotation,c):
    p=np.asarray(translation,dtype=float);r=np.asarray(rotation,dtype=float)
    require(p.shape==(3,) and r.shape==(3,3) and np.all(np.isfinite(p)) and np.all(np.isfinite(r)) and
            np.allclose(r.T@r,np.eye(3),atol=1e-6) and abs(np.linalg.det(r)-1)<1e-6,"invalid_camera_transform")
    ray=np.linalg.solve(k,[float(pixel[0]),float(pixel[1]),1.]);world_ray=r@ray
    require(abs(world_ray[2])/np.linalg.norm(world_ray)>c["minimum_ray_plane_cosine"],"ray_near_parallel")
    scale=(c["marker_plane_z_m"]-p[2])/world_ray[2]
    require(c["minimum_camera_depth_m"]<scale<c["maximum_camera_depth_m"],"plane_outside_camera_depth")
    hit=p+scale*world_ray
    require(c["workspace_x_m"][0]<=hit[0]<=c["workspace_x_m"][1] and
            c["workspace_y_m"][0]<=hit[1]<=c["workspace_y_m"][1],"outside_localization_workspace")
    hit[2]=c["handle_plane_z_m"]
    return hit

def localize(rgb,k,translation,rotation,c):
    detection=detect_marker(rgb,c);uv=np.array([detection["pixel_u"],detection["pixel_v"]])
    position=ray_plane(uv,k,translation,rotation,c);j=np.zeros((2,2));eps=0.01
    for axis in range(2):
        step=np.zeros(2);step[axis]=eps
        j[:,axis]=(ray_plane(uv+step,k,translation,rotation,c)[:2]-ray_plane(uv-step,k,translation,rotation,c)[:2])/(2*eps)
    covariance=(c["pixel_sigma_px"]**2)*(j@j.T)
    if "full_marker_extent_x_m" in c:
        # Reject a clipped rectangular fragment. Known marker size is a
        # calibration prior, independent of simulator object position.
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        mask=cv2.inRange(hsv,np.asarray(c["hsv_lower"],np.uint8),np.asarray(c["hsv_upper"],np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea)
        pixels=contour[:,0,:].astype(float)+c["pixel_center_offset"]
        rays=np.linalg.solve(k,np.c_[pixels,np.ones(len(pixels))].T).T@np.asarray(rotation).T
        p=np.asarray(translation);cosine=np.abs(rays[:,2])/np.linalg.norm(rays,axis=1)
        require(np.all(cosine>c["minimum_ray_plane_cosine"]),"ray_near_parallel")
        depth=(c["marker_plane_z_m"]-p[2])/rays[:,2]
        require(np.all((depth>c["minimum_camera_depth_m"])&(depth<c["maximum_camera_depth_m"])),"plane_outside_camera_depth")
        projected=(p+rays*depth[:,None])[:,:2]
        require(np.all((projected[:,0]>=c["workspace_x_m"][0])&(projected[:,0]<=c["workspace_x_m"][1])&
                       (projected[:,1]>=c["workspace_y_m"][0])&(projected[:,1]<=c["workspace_y_m"][1])),"outside_localization_workspace")
        extent=np.ptp(projected,axis=0)
        require(all(c["full_marker_extent_"+axis+"_m"][0]<=extent[i]<=c["full_marker_extent_"+axis+"_m"][1]
                    for i,axis in enumerate(("x","y"))),"marker_extent_incomplete")
        detection["observed_marker_extent_xy_m"]=extent.tolist()
        # A centroid in the image is biased under oblique perspective.
        # Map the contour to the calibrated plane before taking its centroid.
        world_moments=cv2.moments(projected.astype(np.float32).reshape(-1,1,2))
        require(abs(world_moments["m00"])>1e-10,"degenerate_world_marker")
        position[:2]=[world_moments["m10"]/world_moments["m00"],world_moments["m01"]/world_moments["m00"]]
    return dict(detection,position=position.tolist(),planar_covariance=covariance.reshape(-1).tolist())

class CaptureGuard:
    """Reject old captures and cached packets across a reset barrier."""
    def __init__(self):self.episode=None;self.barrier_ns=0;self.last_stamp={}
    def reset(self,episode,stamp_ns):
        if self.episode is not None and episode<=self.episode:return False
        self.episode=int(episode);self.barrier_ns=int(stamp_ns);self.last_stamp.clear();return True
    def check(self,view,stamp_ns,now_ns,c):
        require(self.episode is not None,"episode_not_received")
        require(stamp_ns>self.barrier_ns,"capture_before_reset")
        require(stamp_ns>self.last_stamp.get(view,0),"out_of_order_capture")
        age=(now_ns-stamp_ns)/1e9
        require(-c["maximum_future_stamp_s"]<=age<c["maximum_image_age_s"],"capture_age_outside_bound")
    def commit(self,view,stamp_ns):self.last_stamp[view]=max(stamp_ns,self.last_stamp.get(view,0))
