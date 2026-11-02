"""Shared M3D robot model and MoveIt pipeline parameters; no ROS node."""
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml,xacro
from ament_index_python.packages import get_package_share_directory

def model_parameters(headless="true",scene_fixture="red_clear"):
    description=Path(get_package_share_directory("sorting_description"))
    moveit=Path(get_package_share_directory("moveit_resources_panda_moveit_config"))
    robot = xacro.process_file(str(description/"urdf/panda_cameras.urdf.xacro"),
                              mappings={"camera_config": "cameras_m3d.yaml", "enable_wrist_ft": "true", "headless": headless,
                                        "mujoco_model": str(description/("mujoco/static_grasp/scene_camera_m3d_"+scene_fixture+".xml"))}).toxml()
    semantic = ET.parse(moveit/"config/panda.srdf").getroot()
    for joint in semantic.findall("virtual_joint"): semantic.remove(joint)
    # Make the grasp reference the IK tip. The hand is fixed to link8; the
    # group still has exactly the same seven active arm joints.
    for group in semantic.findall("group"):
        if group.get("name") == "panda_arm": group.find("chain").set("tip_link", "panda_hand")
    model = {"robot_description": robot,
             "robot_description_semantic": ET.tostring(semantic, encoding="unicode"),
             "robot_description_kinematics": yaml.safe_load((moveit/"config/kinematics.yaml").read_text()),
             "robot_description_planning": yaml.safe_load((moveit/"config/joint_limits.yaml").read_text()),
             "use_sim_time": True}
    return model

def planning_parameters():
    control=Path(get_package_share_directory("sorting_control"))
    moveit=Path(get_package_share_directory("moveit_resources_panda_moveit_config"))
    carried = yaml.safe_load((control/"config/carried_planning.yaml").read_text())
    ompl = yaml.safe_load((moveit/"config/ompl_planning.yaml").read_text())
    ompl["planning_plugins"] = ["sorting_control/CarriedMarginPlanner"]
    ompl["panda_arm"].update(carried["group"])
    ompl["planner_configs"][carried["planner_id"]] = carried["profile"]
    ompl["panda_arm"]["planner_configs"].append(carried["planner_id"])
    ompl["totg"] = carried["totg"]
    planning = {"planning_pipelines": ["ompl"], "default_planning_pipeline": "ompl",
                "ompl": ompl,
                "allow_trajectory_execution": False, "moveit_manage_controllers": False,
                "moveit_controller_manager": "sorting_control/PlanningOnlyControllers",
                "disable_capabilities": "move_group/MoveGroupExecuteTrajectoryAction move_group/MoveGroupMoveAction move_group/MoveGroupExecuteService",
                "publish_planning_scene": True, "publish_geometry_updates": True,
                "publish_state_updates": True, "publish_transforms_updates": True,
                "monitor_dynamics": True}
    return planning
