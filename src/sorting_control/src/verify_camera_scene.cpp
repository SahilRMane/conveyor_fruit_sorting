// PlanningScene message replay. No ROS context or node is initialized.
#include <moveit/planning_scene/planning_scene.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
std::string read(const char* file){std::ifstream in(file,std::ios::binary);if(!in)throw std::runtime_error("Cannot read fixture");std::ostringstream out;out<<in.rdbuf();return out.str();}
int main(int argc,char** argv){
  if(argc!=9){std::cerr<<"Usage: verify_camera_scene URDF SRDF HOME MOVE ALLOW HELD DETACH RESET_CDR\n";return 2;}
  try{
    auto u=urdf::parseURDF(read(argv[1]));auto s=std::make_shared<srdf::Model>();if(!u||!s->initString(*u,read(argv[2])))throw std::runtime_error("Invalid robot model");
    auto model=std::make_shared<moveit::core::RobotModel>(u,s);planning_scene::PlanningScene scene(model);
    rclcpp::Serialization<moveit_msgs::msg::PlanningScene> serialization;
    for(int i=3;i<argc;++i){auto data=read(argv[i]);rclcpp::SerializedMessage serialized(data.size());auto& buffer=serialized.get_rcl_serialized_message();std::memcpy(buffer.buffer,data.data(),data.size());buffer.buffer_length=data.size();
      moveit_msgs::msg::PlanningScene msg;serialization.deserialize_message(&serialized,&msg);
      bool applied=scene.setPlanningSceneDiffMsg(msg);moveit_msgs::msg::PlanningScene result;moveit_msgs::msg::PlanningSceneComponents comp;
      comp.components=comp.WORLD_OBJECT_GEOMETRY|comp.ROBOT_STATE_ATTACHED_OBJECTS|comp.ALLOWED_COLLISION_MATRIX;scene.getPlanningSceneMsg(result,comp);
      const int expected_world[]={24,24,24,16,24,24},expected_attached[]={0,0,0,8,0,0};
      const bool expected_permission[]={false,false,true,true,true,false};
      if(!applied || result.world.collision_objects.size()!=size_t(expected_world[i-3]) || result.robot_state.attached_collision_objects.size()!=size_t(expected_attached[i-3]))throw std::runtime_error("Scene transition or shape count failed");
      for(const char* finger:{"panda_leftfinger","panda_rightfinger"}){
        collision_detection::AllowedCollision::Type type;bool found=scene.getAllowedCollisionMatrix().getAllowedCollision("handle_bar",finger,type);
        if((found && type==collision_detection::AllowedCollision::ALWAYS)!=expected_permission[i-3])throw std::runtime_error("Effective handle collision permission failed");
      }
      if(i==4){
        auto object=std::find_if(result.world.collision_objects.begin(),result.world.collision_objects.end(),[](const auto& o){return o.id=="tray_base";});
        if(object==result.world.collision_objects.end() || object->primitive_poses.empty() || std::abs(object->pose.position.x+object->primitive_poses[0].position.x-.62)>1e-8)throw std::runtime_error("MOVE failed to update tray position");
      }
      std::cout<<"Fixture "<<argv[i]<<" applied="<<applied<<" world="<<result.world.collision_objects.size()<<" attached="<<result.robot_state.attached_collision_objects.size()<<'\n';
      for(const char* finger:{"panda_leftfinger","panda_rightfinger"}){collision_detection::AllowedCollision::Type allowed;
        bool found=scene.getAllowedCollisionMatrix().getEntry("handle_bar",finger,allowed);std::cout<<"ACM "<<finger<<" found="<<found<<" allowed="<<(found?int(allowed):-1)<<'\n';}
      std::cout<<"WORLD ";for(auto& o:result.world.collision_objects)std::cout<<o.id<<',';std::cout<<'\n';
      std::cout<<"ACM_NAMES ";for(auto& n:result.allowed_collision_matrix.entry_names)std::cout<<n<<',';std::cout<<'\n';
    }
    std::cout<<"PASS: camera scene MOVE/attach/detach/reset and effective collision permissions\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;
}
