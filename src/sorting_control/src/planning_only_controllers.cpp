// MoveItCpp constructs an execution manager even when execution is disabled.
// This plugin supplies zero controller handles and creates no command clients.
#include <moveit/controller_manager/controller_manager.hpp>
#include <pluginlib/class_list_macros.hpp>
namespace sorting_control {
class PlanningOnlyControllers : public moveit_controller_manager::MoveItControllerManager {
 public:
  void initialize(const rclcpp::Node::SharedPtr&) override {}
  moveit_controller_manager::MoveItControllerHandlePtr getControllerHandle(const std::string&) override {return nullptr;}
  void getControllersList(std::vector<std::string>& names) override {names.clear();}
  void getActiveControllers(std::vector<std::string>& names) override {names.clear();}
  void getControllerJoints(const std::string&,std::vector<std::string>& joints) override {joints.clear();}
  ControllerState getControllerState(const std::string&) override {return {};}
  bool switchControllers(const std::vector<std::string>& activate,const std::vector<std::string>& deactivate) override {
    return activate.empty() && deactivate.empty();
  }
};
}
PLUGINLIB_EXPORT_CLASS(sorting_control::PlanningOnlyControllers,moveit_controller_manager::MoveItControllerManager)
