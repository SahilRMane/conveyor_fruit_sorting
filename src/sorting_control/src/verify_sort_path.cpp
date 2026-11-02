// Production payload scenes at measured Panda poses. No ROS participant.
#include <moveit/planning_scene/planning_scene.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/util/RandomNumbers.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <set>
std::string read(const char* p){std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("Missing fixture");std::ostringstream out;out<<in.rdbuf();return out.str();}
int main(int argc,char** argv){
 try{
  if(argc<5)throw std::runtime_error("Usage: verify_sort_path URDF SRDF SCENE...");
  auto u=urdf::parseURDF(read(argv[1]));auto s=std::make_shared<srdf::Model>();if(!u||!s->initString(*u,read(argv[2])))throw std::runtime_error("Invalid Panda");
  auto model=std::make_shared<moveit::core::RobotModel>(u,s);ompl::RNG::setSeed(173);unsigned carried=0,placed=0,planned=0;moveit_msgs::msg::PlanningScene seed;std::string previous_phase;unsigned since_plan=0;double minimum_world=1e6,minimum_self=1e6;
  for(int i=3;i<argc;++i){
   auto data=read(argv[i]);rclcpp::SerializedMessage buffer(data.size());auto& bytes=buffer.get_rcl_serialized_message();std::memcpy(bytes.buffer,data.data(),data.size());bytes.buffer_length=data.size();
   moveit_msgs::msg::PlanningScene msg;rclcpp::Serialization<moveit_msgs::msg::PlanningScene>().deserialize_message(&buffer,&msg);
   bool carry=std::string(argv[i]).find("-carry.cdr")!=std::string::npos;
   if(!msg.robot_state.is_diff||msg.robot_state.attached_collision_objects.size()!=(carry?8u:0u))throw std::runtime_error("Payload start state incomplete");
   planning_scene::PlanningScene scene(model);if(!scene.setPlanningSceneDiffMsg(msg))throw std::runtime_error("Scene rejected");scene.getCurrentStateNonConst().update();
   collision_detection::CollisionRequest request;request.group_name="panda_arm";request.distance=true;collision_detection::CollisionResult world,self;
   scene.getCollisionEnv()->checkRobotCollision(request,world,scene.getCurrentState(),scene.getAllowedCollisionMatrix());scene.getCollisionEnvUnpadded()->checkSelfCollision(request,self,scene.getCurrentState(),scene.getAllowedCollisionMatrix());
   std::cout<<argv[i]<<" world="<<world.collision<<"/"<<world.distance<<" self="<<self.collision<<"/"<<self.distance<<'\n';
   if(world.collision||self.collision||world.distance<.02||self.distance<.01)throw std::runtime_error("Payload path collides or triggers Servo proximity stop");
   std::string phase=std::string(argv[i]).find("-transfer_retract-")!=std::string::npos?"retract":(std::string(argv[i]).find("-transfer-")!=std::string::npos?"transfer":"other");
   if(phase!="other"){
    if(phase!=previous_phase){seed=msg;since_plan=0;}
    else if(++since_plan>=20){
     planning_scene::PlanningScene plan_scene(model);plan_scene.setPlanningSceneDiffMsg(seed);auto start=plan_scene.getCurrentState();auto goal=scene.getCurrentState();auto group=model->getJointModelGroup("panda_arm");
     const auto& names=group->getVariableNames();auto space=std::make_shared<ompl::base::RealVectorStateSpace>(7);ompl::base::RealVectorBounds bounds(7);
     ompl::base::ScopedState<> begin(space),end(space);
     for(unsigned j=0;j<7;++j){const auto& b=model->getVariableBounds(names[j]);bounds.setLow(j,b.min_position_);bounds.setHigh(j,b.max_position_);begin[j]=start.getVariablePosition(names[j]);end[j]=goal.getVariablePosition(names[j]);}space->setBounds(bounds);
     auto start_tf=start.getGlobalLinkTransform("panda_hand"),end_tf=goal.getGlobalLinkTransform("panda_hand");double low=std::min(start_tf.translation().z(),end_tf.translation().z())-.005,high=std::max(start_tf.translation().z(),end_tf.translation().z())+.005;
     ompl::geometric::SimpleSetup setup(space);setup.setStateValidityChecker([&](const ompl::base::State* raw){
      auto state=start;state.setJointGroupPositions(group,raw->as<ompl::base::RealVectorStateSpace::StateType>()->values);state.update();auto tf=state.getGlobalLinkTransform("panda_hand");
      return state.satisfiesBounds(group)&&Eigen::AngleAxisd(start_tf.rotation().transpose()*tf.rotation()).angle()<.01&&tf.translation().z()>=low&&tf.translation().z()<=high&&!plan_scene.isStateColliding(state,"panda_arm");
     });setup.getSpaceInformation()->setStateValidityCheckingResolution(.0005);setup.setStartAndGoalStates(begin,end);auto planner=std::make_shared<ompl::geometric::RRTConnect>(setup.getSpaceInformation());planner->setRange(.02);setup.setPlanner(planner);
     if(setup.solve(2.)!=ompl::base::PlannerStatus::EXACT_SOLUTION)throw std::runtime_error("Native constrained carried OMPL segment failed");
     auto path=setup.getSolutionPath();path.interpolate(100);for(const auto* raw:path.getStates())if(!setup.getSpaceInformation()->isValid(raw))throw std::runtime_error("Constrained carried path invalid");
     ++planned;seed=msg;since_plan=0;
    }
   }
   previous_phase=phase;
   minimum_world=std::min(minimum_world,world.distance);minimum_self=std::min(minimum_self,self.distance);if(carry)++carried;else ++placed;
  }
  if(!carried||!placed||!planned)throw std::runtime_error("Missing carried/placed phases");
  std::cout<<"PASS constrained_OMPL_segments="<<planned<<" carried="<<carried<<" placed="<<placed<<" minimum_world_m="<<minimum_world<<" minimum_self_m="<<minimum_self<<'\n';
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;
}
