#include <moveit/controller_manager/controller_manager.hpp>
#include <pluginlib/class_loader.hpp>
#include <iostream>
int main(){
  try{
    pluginlib::ClassLoader<moveit_controller_manager::MoveItControllerManager> loader("moveit_core","moveit_controller_manager::MoveItControllerManager");
    auto manager=loader.createSharedInstance("sorting_control/PlanningOnlyControllers");manager->initialize(nullptr);
    std::vector<std::string> names={"panda_arm_controller"};manager->getControllersList(names);if(!names.empty())return 1;
    names={"panda_arm_controller"};manager->getActiveControllers(names);if(!names.empty())return 1;
    names={"panda_joint1"};manager->getControllerJoints("panda_arm_controller",names);if(!names.empty())return 1;
    if(manager->getControllerHandle("panda_arm_controller"))return 1;
    if(manager->switchControllers({"panda_arm_controller"},{}))return 1;
    auto state=manager->getControllerState("panda_arm_controller");if(state.active_ || state.default_)return 1;
    std::cout<<"PASS: installed planning-only plugin exposes no arm command handle\n";
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
  return 0;
}
