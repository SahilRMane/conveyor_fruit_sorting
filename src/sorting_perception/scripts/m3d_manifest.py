"""Installed source/model hashes for the fixed-view M3D integration."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory,get_package_prefix
from classifier_core import digest

def installed_manifest():
    groups=dict(sorting_control=['config/m3d.yaml','config/carried_planning.yaml','config/m3b.yaml','config/m2b.yaml','config/servo.yaml','config/ownership.yaml','config/fixtures.yaml','config/ownership_controllers.yaml','config/wrist_sensor.yaml','launch/panda_m3d.launch.py'],sorting_perception=['config/m3c.yaml','config/cameras_m3c.yaml','config/m3b_localization.yaml','config/m3b_filter.yaml'],sorting_description=['mujoco/static_grasp/bins.xml','urdf/panda_cameras.urdf.xacro','urdf/panda_mujoco.urdf.xacro'])
    result={}
    groups['sorting_perception'].append('config/cameras_m3d.yaml')
    groups['sorting_simulation']=['config/camera_plugins_m3d.yaml']
    for package,files in groups.items():
        for rel in files:result[package+'/'+rel]=digest(Path(get_package_share_directory(package))/rel)
    for package,names in dict(sorting_control=['arm_command_gate.py','hold_reference.py','camera_pickup.py','camera_sort.py','camera_sort_scene.py','camera_scene.py','sort_core.py','sort_planning.py','m3d_parameters.py','camera_scene_checks.py','wrist_support.py'],sorting_perception=['handle_localizer.py','motion_filter.py','camera_qos.py','fruit_classifier.py','classifier_core.py','classifier_state.py','m3d_evaluator.py','native_sort.py','hold_handoff_probe.py','carried_endpoint_probe.py','floor_release_probe.py','native_camera_pickup.py','sort_fixtures.py','m3d_acceptance.py','m3d_manifest.py','test_m3d.py','validate_m3d_files.py','replay_m3d_reset.py','probe_m3d_planning.py','carried_checks.py']).items():
        for name in names:result[package+'/'+name]=digest(Path(get_package_prefix(package))/'lib'/package/name)
    d=Path(get_package_share_directory('sorting_description'))/'mujoco/static_grasp'
    for fixture in ('red_clear','green_clear','red_covered','green_covered'):
        for kind in ('scene','props'):result[f'{kind}_{fixture}']=digest(d/(f'{kind}_camera_m3d_{fixture}.xml' if kind=='scene' else f'{kind}_camera_m3c_{fixture}.xml'))
    p=Path(get_package_share_directory('sorting_perception'))/'models/m3c_baseline'
    result['weights.npz']=digest(p/'weights.npz');result['model.json']=digest(p/'model.json')
    result['native_camera_plant']=digest(Path(get_package_prefix('sorting_simulation'))/'lib/libnative_camera_plant.so')
    result['panda_camera_m3c']=digest(d/'panda_camera_m3c.xml')
    result['panda_camera_m3d']=digest(d/'panda_camera_m3d.xml')
    result['native_servo_precision']=digest(Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_servo_precision')
    result['precision_smoothing_library']=digest(Path(get_package_prefix('sorting_control'))/'lib/libsorting_precision_smoothing.so')
    result['precision_smoothing_plugin']=digest(Path(get_package_share_directory('sorting_control'))/'precision_smoothing_plugin.xml')
    result['smoothing_kernel_check']=digest(Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_precision_smoothing')
    result['servo_pipeline_check']=digest(Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_servo_pipeline')
    result['component_checks']=digest(Path(get_package_prefix('sorting_perception'))/'lib/sorting_perception/m3d_components.py')
    result['carried_motion_check']=digest(Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_carried_motion')
    result['carried_fixtures']=digest(Path(get_package_share_directory('sorting_perception'))/'config/m3d_carried_fixtures.json')
    result['carried_margin_library']=digest(Path(get_package_prefix('sorting_control'))/'lib/libsorting_carried_margin_planner.so')
    result['carried_margin_plugin']=digest(Path(get_package_share_directory('sorting_control'))/'carried_margin_plugin.xml')
    result['carried_pipeline_check']=digest(Path(get_package_prefix('sorting_control'))/'lib/sorting_control/verify_carried_pipeline')
    return result
