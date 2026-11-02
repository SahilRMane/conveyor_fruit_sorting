// Node-free OMPL joint motion + installed MoveIt time parameterization and validation.
#include <moveit/planning_scene/planning_scene.hpp>
#include "sorting_control/carried_margin.hpp"
#include <pluginlib/class_loader.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <rclcpp/parameter.hpp>
#include <boost/lexical_cast.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/util/RandomNumbers.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::string read(const char* p){std::ifstream f(p,std::ios::binary);need(bool(f),"Missing check fixture");std::ostringstream s;s<<f.rdbuf();return s.str();}
template<class T>T message(const std::string& p){auto bytes=read(p.c_str());rclcpp::SerializedMessage b(bytes.size());auto& r=b.get_rcl_serialized_message();std::memcpy(r.buffer,bytes.data(),bytes.size());r.buffer_length=bytes.size();T m;rclcpp::Serialization<T>().deserialize_message(&b,&m);return m;}
struct CapturedContext final:planning_interface::PlanningContext{
 CapturedContext():PlanningContext("capture","panda_arm"){}
 void solve(planning_interface::MotionPlanResponse&)override{}
 void solve(planning_interface::MotionPlanDetailedResponse&)override{}
 bool terminate()override{return true;}void clear()override{}
};
struct CapturedPlanner final:planning_interface::PlannerManager{
 mutable planning_interface::MotionPlanRequest captured;
 std::string getDescription()const override{return "node-free capture";}
 bool canServiceRequest(const planning_interface::MotionPlanRequest&)const override{return true;}
 planning_interface::PlanningContextPtr getPlanningContext(const planning_scene::PlanningSceneConstPtr& scene,const planning_interface::MotionPlanRequest& req,moveit_msgs::msg::MoveItErrorCodes& error)const override{
  captured=req;auto context=std::make_shared<CapturedContext>();context->setPlanningScene(scene);context->setMotionPlanRequest(req);error.val=1;return context;
 }
};
int main(int argc,char** argv){
 if(argc!=7){std::cerr<<"Usage: verify_carried_motion URDF SRDF JOINT_LIMITS CASES_YAML PROFILE OUTPUT_JSON\n";return 2;}
 try{
  auto u=urdf::parseURDF(read(argv[1]));need(bool(u),"Invalid URDF");auto semantic=std::make_shared<srdf::Model>();need(semantic->initString(*u,read(argv[2])),"Invalid SRDF");auto model=std::make_shared<moveit::core::RobotModel>(u,semantic);auto group=model->getJointModelGroup("panda_arm");need(group && group->getVariableCount()==7,"Invalid arm group");
  const auto names=group->getVariableNames();auto limits=YAML::LoadFile(argv[3])["joint_limits"];
  for(auto& name:names){auto b=model->getVariableBounds(name);b.acceleration_bounded_=true;b.max_acceleration_=limits[name]["max_acceleration"].as<double>();b.min_acceleration_=-b.max_acceleration_;model->getJointModel(name)->setVariableBounds(name,b);}
  auto profile=YAML::LoadFile(argv[5]);
  // Exercise the installed ROS parameter formatting and the MoveIt bool parser.
  // MoveIt 2.12.4 uses value_to_string for planner overrides, but to_string
  // for the typed group bool. A planner override replaces the group value.
  const rclcpp::Parameter flag("enforce_joint_model_state_space",true);
  const auto planner_text=flag.value_to_string();
  bool old_rejected=false;
  try{(void)boost::lexical_cast<bool>(planner_text);}catch(const boost::bad_lexical_cast&){old_rejected=true;}
  need(old_rejected,"Installed bool conversion changed; review the planner parameter check");
  need(!profile["profile"]["enforce_joint_model_state_space"],"Remove planner-level bool: its text overrides the typed group value");
  need(profile["group"]["enforce_joint_model_state_space"].as<bool>(),"Joint planning is not enabled at group level");
  const rclcpp::Parameter group_flag("enforce_joint_model_state_space",profile["group"]["enforce_joint_model_state_space"].as<bool>());
  const auto group_text=std::to_string(group_flag.as_bool());
  need(boost::lexical_cast<bool>(group_text),"Group bool conversion failed");
  std::cout<<"PASS: installed parameter conversion; planner text="<<planner_text<<" rejected, group text="<<group_text<<" accepted\n";
  KDL::Tree tree;need(kdl_parser::treeFromUrdfModel(*u,tree),"KDL tree failed");KDL::Chain chain;need(tree.getChain("panda_link0","panda_hand",chain),"KDL chain missing");KDL::JntArray lo(7),hi(7);for(int j=0;j<7;++j){const auto& b=model->getVariableBounds(names[j]);lo(j)=b.min_position_;hi(j)=b.max_position_;}
  KDL::ChainFkSolverPos_recursive fk(chain);KDL::ChainIkSolverVel_pinv velocity(chain);KDL::ChainIkSolverPos_NR_JL ik(chain,lo,hi,fk,velocity,150,1e-7);
  pluginlib::ClassLoader<planning_interface::PlannerManager> loader("moveit_core","planning_interface::PlannerManager");
  auto plugin=loader.createSharedInstance("sorting_control/CarriedMarginPlanner");need(bool(plugin),"Carried margin plugin did not load");
  ompl::RNG::setSeed(173);unsigned total=0,negative=0,all_negative=0,negative_cases=0,margin_checks=0;std::ostringstream reports;
  for(auto fixture:YAML::LoadFile(argv[4])["cases"]){
   auto msg=message<moveit_msgs::msg::PlanningScene>(fixture["scene"].as<std::string>());auto req=message<moveit_msgs::msg::MotionPlanRequest>(fixture["request"].as<std::string>());
   planning_scene::PlanningScene scene(model);need(scene.setPlanningSceneDiffMsg(msg),"Scene rejected");auto start=scene.getCurrentState();start.update();std::vector<const moveit::core::AttachedBody*> bodies;start.getAttachedBodies(bodies);need(bodies.size()==8,"Payload parts missing");
   if(fixture["bad"]){auto bad=message<trajectory_msgs::msg::JointTrajectory>(fixture["bad"].as<std::string>());moveit_msgs::msg::RobotTrajectory m;m.joint_trajectory=bad;robot_trajectory::RobotTrajectory old(model,"panda_arm");old.setRobotTrajectoryMsg(start,m);std::vector<std::size_t> invalid;
    need(!scene.isPathValid(old,req.path_constraints,req.goal_constraints,"panda_arm",false,&invalid),"Recorded bad path was not rejected");all_negative+=invalid.size();++negative_cases;
    if(fixture["label"].as<std::string>()=="red_failure_pose")negative=invalid.size();
    need(!invalid.empty(),"Missing invalid states");}
   need(scene.isStateValid(start,req.path_constraints,"panda_arm"),"Recorded start invalid");
   const auto original=req;auto inner=sorting_control::carried_interior_request(req);
   need(req==original,"Interior guard modified the original request");
   need(inner.start_state==req.start_state && inner.goal_constraints==req.goal_constraints,"Interior guard modified payload or goals");
   for(unsigned j=0;j<3;++j)need(std::abs(req.path_constraints.position_constraints[0].constraint_region.primitives[0].dimensions[j]-inner.path_constraints.position_constraints[0].constraint_region.primitives[0].dimensions[j]-.001)<1e-12,"Interior position margin differs");
   need(std::abs(inner.path_constraints.orientation_constraints[0].absolute_x_axis_tolerance-.009)<1e-12,"Interior orientation margin differs");
   CapturedPlanner capture;moveit_msgs::msg::MoveItErrorCodes error;
   auto context=sorting_control::carried_interior_context(capture,{},req,error);
   need(context && error.val==1 && capture.captured==inner && req==original,"Planner delegation changed the outer request");
   auto ordinary=req;ordinary.planner_id="RRTConnectkConfigDefault";
   need(sorting_control::carried_interior_request(ordinary)==ordinary,"Non-carried request changed");
   auto missing=req;missing.start_state.attached_collision_objects.pop_back();
   need(!sorting_control::carried_interior_context(capture,{},missing,error) && error.val==-2,"Incomplete payload accepted");
   auto narrow=req;narrow.path_constraints.position_constraints[0].constraint_region.primitives[0].dimensions[2]=.0005;
   need(!sorting_control::carried_interior_context(capture,{},narrow,error) && error.val==-2,"Too-small corridor accepted");
   auto narrow_attitude=req;narrow_attitude.path_constraints.orientation_constraints[0].absolute_x_axis_tolerance=.0005;
   need(!sorting_control::carried_interior_context(capture,{},narrow_attitude,error) && error.val==-2,"Too-small orientation bound accepted");
   need(scene.isStateValid(start,inner.path_constraints,"panda_arm"),"Recorded start outside interior guard");margin_checks+=8;
   KDL::JntArray q(7),next(7);for(int j=0;j<7;++j)q(j)=start.getVariablePosition(names[j]);KDL::Frame first;need(fk.JntToCart(q,first)>=0,"FK failed");
   auto goal=req.goal_constraints[0];auto p=goal.position_constraints[0].constraint_region.primitive_poses[0].position;auto quat=goal.orientation_constraints[0].orientation;
   for(int step=1;step<=10;++step){auto target=first;double t=step/10.;target.p=first.p*(1-t)+KDL::Vector(p.x,p.y,p.z)*t;target.M=KDL::Rotation::Quaternion(quat.x,quat.y,quat.z,quat.w);need(ik.CartToJnt(q,target,next)>=0,"Continuation IK failed");q=next;}
   auto space=std::make_shared<ompl::base::RealVectorStateSpace>(7);ompl::base::RealVectorBounds bounds(7);for(int j=0;j<7;++j){bounds.setLow(j,lo(j));bounds.setHigh(j,hi(j));}space->setBounds(bounds);
   for(int repeat=0;repeat<fixture["native_repeats"].as<int>();++repeat){
    ompl::geometric::SimpleSetup setup(space);setup.getSpaceInformation()->setStateValidityCheckingResolution(profile["profile"]["longest_valid_segment_fraction"].as<double>());
    setup.setStateValidityChecker([&](const ompl::base::State* raw){auto state=start;state.setJointGroupPositions(group,raw->as<ompl::base::RealVectorStateSpace::StateType>()->values);state.update();return state.satisfiesBounds(group)&&scene.isStateValid(state,inner.path_constraints,"panda_arm");});
    ompl::base::ScopedState<> begin(space),end(space);for(int j=0;j<7;++j){begin[j]=start.getVariablePosition(names[j]);end[j]=q(j);}setup.setStartAndGoalStates(begin,end);auto planner=std::make_shared<ompl::geometric::RRTConnect>(setup.getSpaceInformation());planner->setRange(profile["profile"]["range"].as<double>());setup.setPlanner(planner);
    need(setup.solve(2.)==ompl::base::PlannerStatus::EXACT_SOLUTION,"Joint-space carried plan failed");auto path=setup.getSolutionPath();path.interpolate();robot_trajectory::RobotTrajectory trajectory(model,"panda_arm");
    for(const auto* raw:path.getStates()){auto state=start;state.setJointGroupPositions(group,raw->as<ompl::base::RealVectorStateSpace::StateType>()->values);state.update();trajectory.addSuffixWayPoint(state,0.);}
    trajectory_processing::TimeOptimalTrajectoryGeneration timing(profile["totg"]["path_tolerance"].as<double>(),profile["totg"]["resample_dt"].as<double>(),profile["totg"]["min_angle_change"].as<double>());
    need(timing.computeTimeStamps(trajectory,req.max_velocity_scaling_factor,req.max_acceleration_scaling_factor),"Time parameterization failed");need(scene.isPathValid(trajectory,req.path_constraints,req.goal_constraints,"panda_arm"),"Retimed carried path violates constraints");need(trajectory.getDuration()<20.,"Carried path too long");++total;
   }
   std::cout<<"PASS: "<<fixture["label"].as<std::string>()<<" "<<fixture["native_repeats"].as<int>()<<" guarded OMPL/TOTG paths\n";
  }
  need(total==120 && negative==723 && negative_cases==3 && all_negative==729 && margin_checks==32,"Incomplete regression");std::ofstream out(argv[6]);out<<"{\"status\":\"PASS\",\"parameter_conversion\":{\"status\":\"PASS\",\"planner_bool_text\":\""<<planner_text<<"\",\"planner_bool_rejected\":true,\"group_bool_text\":\""<<group_text<<"\"},\"corrected_paths\":"<<total<<",\"old_invalid_states\":"<<negative<<",\"recorded_invalid_states\":"<<all_negative<<",\"recorded_failed_paths\":"<<negative_cases<<",\"interior_margin_checks\":"<<margin_checks<<",\"margin_plugin_load\":\"PASS\",\"scope\":\"Guarded native joint-space OMPL plus installed MoveIt TOTG and original outer scene validation; full live pipeline requires planning-only probe\"}\n";need(out.good(),"Report write failed");return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
