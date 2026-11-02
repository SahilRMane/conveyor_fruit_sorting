// Complete carried geometry and Servo collision/distance queries. No ROS node.
#include <moveit/planning_scene/planning_scene.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <set>
std::string read_file(const char* path){std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot read fixture");std::ostringstream out;out<<in.rdbuf();return out.str();}
const std::set<std::string> tray={"tray_base","tray_wall_x_minus","tray_wall_x_plus","tray_wall_y_minus","tray_wall_y_plus","handle_post_minus","handle_post_plus","handle_bar"};
bool allowed(const planning_scene::PlanningScene& scene,const std::string& a,const std::string& b){collision_detection::AllowedCollision::Type type;return scene.getAllowedCollisionMatrix().getAllowedCollision(a,b,type)&&type==collision_detection::AllowedCollision::ALWAYS;}
int main(int argc,char** argv){
  if(argc<8){std::cerr<<"Usage: verify_lift_support URDF SRDF LEGACY_INTERNAL LEGACY_BASE_ONLY CONTACT CLEAR [PATH...] OUTPUT_JSON\n";return 2;}
  try{
    auto u=urdf::parseURDF(read_file(argv[1]));auto s=std::make_shared<srdf::Model>();if(!u||!s->initString(*u,read_file(argv[2])))throw std::runtime_error("Invalid robot model");
    auto model=std::make_shared<moveit::core::RobotModel>(u,s);std::ostringstream rows;double min_world=1e6,min_self=1e6;bool clear_permission=false;
    for(int i=0;i<argc-4;++i){
      planning_scene::PlanningScene scene(model);auto data=read_file(argv[i+3]);rclcpp::SerializedMessage serialized(data.size());auto& buffer=serialized.get_rcl_serialized_message();std::memcpy(buffer.buffer,data.data(),data.size());buffer.buffer_length=data.size();
      rclcpp::Serialization<moveit_msgs::msg::PlanningScene> serializer;moveit_msgs::msg::PlanningScene msg;serializer.deserialize_message(&serialized,&msg);
      if(msg.robot_state.attached_collision_objects.size()!=8||!scene.setPlanningSceneDiffMsg(msg))throw std::runtime_error("Complete carried fixture did not apply");scene.getCurrentStateNonConst().update();
      collision_detection::CollisionRequest request;request.distance=true;request.group_name=i<4?"":"panda_arm";
      collision_detection::CollisionResult world,self;
      scene.getCollisionEnv()->checkRobotCollision(request,world,scene.getCurrentState(),scene.getAllowedCollisionMatrix());
      scene.getCollisionEnvUnpadded()->checkSelfCollision(request,self,scene.getCurrentState(),scene.getAllowedCollisionMatrix());
      collision_detection::DistanceRequest distance;distance.acm=&scene.getAllowedCollisionMatrix();distance.group_name=request.group_name;distance.enableGroup(model);
      collision_detection::DistanceResult dw,ds;scene.getCollisionEnv()->distanceRobot(distance,dw,scene.getCurrentState());scene.getCollisionEnvUnpadded()->distanceSelf(distance,ds,scene.getCurrentState());
      std::cout<<"Fixture "<<i<<" world_collision="<<world.collision<<" world_distance="<<world.distance<<" pair="<<dw.minimum_distance.link_names[0]<<'/'<<dw.minimum_distance.link_names[1]
               <<" self_collision="<<self.collision<<" self_distance="<<self.distance<<" pair="<<ds.minimum_distance.link_names[0]<<'/'<<ds.minimum_distance.link_names[1]<<'\n';
      if(i==0){if(self.collision||self.distance>=.01||!tray.count(ds.minimum_distance.link_names[0])||!tray.count(ds.minimum_distance.link_names[1]))throw std::runtime_error("Original rigid-part distance slowdown not reproduced");}
      else if(i==1){if(world.collision||world.distance<=0||world.distance>=.02)throw std::runtime_error("Original wall/post support proximity slowdown not reproduced");}
      else{
        if(world.collision||self.collision||world.distance<.02||self.distance<.01)throw std::runtime_error("Complete carried fixture would trigger Servo collision scaling");
        min_world=std::min(min_world,world.distance);min_self=std::min(min_self,self.distance);
        if(i>=3&&!allowed(scene,"tray_base","support_surface"))clear_permission=true;
      }
      if(i==3){for(const auto& part:tray)if(allowed(scene,part,"support_surface"))throw std::runtime_error("Clear-lift support permission remains enabled");}
      if(i)rows<<",\n";
      rows<<"    {\"index\": "<<i<<", \"world_collision\": "<<(world.collision?"true":"false")<<", \"self_collision\": "<<(self.collision?"true":"false")
          <<", \"world_distance_m\": "<<world.distance<<", \"self_distance_m\": "<<self.distance<<", \"nearest_world_pair\": [\""<<dw.minimum_distance.link_names[0]<<"\",\""<<dw.minimum_distance.link_names[1]
          <<"\"], \"nearest_self_pair\": [\""<<ds.minimum_distance.link_names[0]<<"\",\""<<ds.minimum_distance.link_names[1]<<"\"]}";
    }
    if(!clear_permission||argc-4<=4)throw std::runtime_error("Native carried path or revocation proof missing");
    std::ofstream out(argv[argc-1]);if(!out)throw std::runtime_error("Cannot write collision report");
    out<<"{\n  \"status\": \"PASS\",\n  \"legacy_internal_distance_slowdown_reproduced\": true,\n  \"legacy_base_only_proximity_slowdown_reproduced\": true,\n"
       <<"  \"complete_carried_path_fixtures\": "<<argc-8<<",\n  \"minimum_corrected_world_distance_m\": "<<min_world<<",\n  \"minimum_corrected_self_distance_m\": "<<min_self
       <<",\n  \"scope\": \"Eight carried parts, robot/world and self collision distances, native camera/encoder lift path; live ROS NOT_RUN\",\n  \"fixtures\": [\n"<<rows.str()<<"\n  ]\n}\n";
    std::cout<<"PASS: complete carried geometry and Servo distance thresholds\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;
}
