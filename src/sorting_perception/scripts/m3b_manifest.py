"""Reproducibility manifest. Never reads simulator state."""
import hashlib
from pathlib import Path
from ament_index_python.packages import get_package_share_directory,get_package_prefix

def installed_manifest():
    entries={
        'sorting_perception':['share/config/m3a.yaml','lib/native_camera_pickup.py','share/config/m3b_filter.yaml','share/config/m3b_localization.yaml','share/config/cameras_m3b.yaml','lib/handle_localizer.py','lib/camera_core.py','lib/filter_core.py','lib/motion_filter.py','lib/m3b_evaluator.py','lib/test_m3b.py','lib/m3b_acceptance.py','lib/lift_support_fixtures.py'],
        'sorting_control':['share/config/m2b.yaml','share/config/fixtures.yaml','share/config/servo.yaml','share/config/ownership_controllers.yaml','share/config/m3b.yaml','share/launch/panda_m3b.launch.py','lib/camera_pickup.py','lib/camera_scene.py','lib/camera_grasp_core.py','lib/camera_scene_checks.py','lib/verify_camera_scene','lib/verify_lift_support'],
        'sorting_description':['share/config/panda_controllers.yaml','share/mujoco/static_grasp/panda_static.xml','share/mujoco/static_grasp/props.xml','share/mujoco/static_grasp/bins.xml','share/mujoco/static_grasp/scene_tracking.xml','share/urdf/panda_cameras.urdf.xacro','share/mujoco/static_grasp/scene_camera_m3b.xml','share/mujoco/static_grasp/panda_camera_m3b.xml','share/mujoco/static_grasp/props_camera_m3b.xml'],
        'sorting_simulation':['share/config/camera_plugins_m3b.yaml','binary/libnative_camera_plant.so']}
    result={}
    for package,files in entries.items():
        for entry in files:
            kind,name=entry.split('/',1)
            base=Path(get_package_share_directory(package)) if kind=='share' else Path(get_package_prefix(package))/'lib'
            if kind=='lib':base=base/package
            result[package+'/'+entry]=hashlib.sha256((base/name).read_bytes()).hexdigest()
    return result
