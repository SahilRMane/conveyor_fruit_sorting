#pragma once
// Keep an interior planning envelope; the pipeline retains the original request.
#include <moveit/planning_interface/planning_interface.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <set>
#include <cmath>
#include <stdexcept>
namespace sorting_control {
inline constexpr double carried_position_margin_m=0.0005;
inline constexpr double carried_orientation_margin_rad=0.001;
inline planning_interface::MotionPlanRequest carried_interior_request(const planning_interface::MotionPlanRequest& outer){
  auto inner=outer;
  if(outer.planner_id!="CarriedRRTConnect")return inner;
  const std::set<std::string> expected={"handle_bar","handle_post_minus","handle_post_plus","tray_base","tray_wall_x_minus","tray_wall_x_plus","tray_wall_y_minus","tray_wall_y_plus"};
  std::set<std::string> parts;for(const auto& item:outer.start_state.attached_collision_objects)parts.insert(item.object.id);
  if(outer.group_name!="panda_arm" || parts!=expected || outer.start_state.attached_collision_objects.size()!=8)
    throw std::invalid_argument("Carried interior planning requires all eight Panda payload parts");
  auto& path=inner.path_constraints;
  if(path.position_constraints.size()!=1 || path.orientation_constraints.size()!=1)
    throw std::invalid_argument("Carried interior planning requires the position and orientation bounds");
  auto& position=path.position_constraints.front();auto& region=position.constraint_region;
  if(position.header.frame_id!="world" || position.link_name!="panda_hand" || region.primitives.size()!=1 ||
     region.primitive_poses.size()!=1 || !region.meshes.empty() || region.primitives[0].type!=shape_msgs::msg::SolidPrimitive::BOX || region.primitives[0].dimensions.size()!=3)
    throw std::invalid_argument("Unsupported carried position corridor");
  for(auto& width:region.primitives[0].dimensions){
    if(!std::isfinite(width) || width<=2*carried_position_margin_m)throw std::invalid_argument("Carried position corridor has no interior margin");
    width-=2*carried_position_margin_m;
  }
  auto& orientation=path.orientation_constraints.front();
  if(orientation.header.frame_id!="world" || orientation.link_name!="panda_hand")throw std::invalid_argument("Unsupported carried orientation frame");
  for(auto* bound:{&orientation.absolute_x_axis_tolerance,&orientation.absolute_y_axis_tolerance,&orientation.absolute_z_axis_tolerance}){
    if(!std::isfinite(*bound) || *bound<=carried_orientation_margin_rad)throw std::invalid_argument("Carried orientation bound has no interior margin");
    *bound-=carried_orientation_margin_rad;
  }
  return inner;
}
inline planning_interface::PlanningContextPtr carried_interior_context(const planning_interface::PlannerManager& planner,
  const planning_scene::PlanningSceneConstPtr& scene,const planning_interface::MotionPlanRequest& outer,
  moveit_msgs::msg::MoveItErrorCodes& error){
  try{return planner.getPlanningContext(scene,carried_interior_request(outer),error);}
  catch(const std::invalid_argument& e){error.val=moveit_msgs::msg::MoveItErrorCodes::INVALID_MOTION_PLAN;error.message=e.what();error.source="sorting_control/CarriedMarginPlanner";return {};}
}
} // namespace sorting_control
