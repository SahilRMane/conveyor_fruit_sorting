// Check measured native sweep states with the installed MoveIt model. No ROS node.
#include <moveit/planning_scene/planning_scene.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
std::string read_file(const char* path) {
  std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("Missing inspection fixture");
  std::ostringstream out;out<<input.rdbuf();return out.str();
}
int main(int argc,char** argv) {
  try {
    if(argc!=7)throw std::runtime_error("Usage: verify_m4a_scene URDF SRDF SCENE_CDR VIEWS_YAML OUTPUT_JSON PARAMETERS_YAML");
    // Exercise the installed ROS parser without creating a node or DDS participant.
    auto parameters=rclcpp::parameter_map_from_yaml_file(argv[6]);
    if(parameters.empty())throw std::runtime_error("Launch parameters are empty");
    auto urdf=urdf::parseURDF(read_file(argv[1]));auto srdf=std::make_shared<srdf::Model>();
    if(!urdf||!srdf->initString(*urdf,read_file(argv[2])))throw std::runtime_error("Invalid inspection robot model");
    auto model=std::make_shared<moveit::core::RobotModel>(urdf,srdf);
    auto raw=read_file(argv[3]);rclcpp::SerializedMessage buffer(raw.size());auto& bytes=buffer.get_rcl_serialized_message();
    std::memcpy(bytes.buffer,raw.data(),raw.size());bytes.buffer_length=raw.size();
    moveit_msgs::msg::PlanningScene message;rclcpp::Serialization<moveit_msgs::msg::PlanningScene>().deserialize_message(&buffer,&message);
    planning_scene::PlanningScene scene(model);if(!scene.setPlanningSceneDiffMsg(message))throw std::runtime_error("Static scene rejected");
    auto state=scene.getCurrentState();auto path=YAML::LoadFile(argv[4]);unsigned count=0;
    for(const auto& sample:path["path_joint_positions"]) {
      if(sample.size()!=7)throw std::runtime_error("Incomplete native joint state");
      for(int j=0;j<7;++j)state.setVariablePosition("panda_joint"+std::to_string(j+1),sample[j].as<double>());
      state.setVariablePosition("panda_finger_joint1",.04);state.setVariablePosition("panda_finger_joint2",.04);state.update();
      if(!state.satisfiesBounds(model->getJointModelGroup("panda_arm"))||scene.isStateColliding(state))
        throw std::runtime_error("Native inspection path violates MoveIt bounds or collision geometry");
      ++count;
    }
    if(count<100)throw std::runtime_error("Too few native inspection samples");
    std::ofstream out(argv[5]);if(!out)throw std::runtime_error("Cannot write collision report");
    out<<"{\"status\":\"PASS\",\"native_path_states_checked\":"<<count<<",\"ROS_nodes_started\":0}\n";
    std::cout<<"PASS: installed MoveIt collision geometry and ROS YAML parser\n";
  }catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
