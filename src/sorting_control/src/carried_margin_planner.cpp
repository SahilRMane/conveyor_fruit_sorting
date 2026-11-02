#include "sorting_control/carried_margin.hpp"
#include <pluginlib/class_loader.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
namespace sorting_control {
class CarriedMarginPlanner final:public planning_interface::PlannerManager {
public:
 bool initialize(const moveit::core::RobotModelConstPtr& model,const rclcpp::Node::SharedPtr& node,const std::string& parameter_namespace)override{
  // These diagnostic parameters identify the compiled margins. They are not tuning inputs.
  for(const auto& entry:std::vector<std::pair<std::string,double>>{{"position_m",carried_position_margin_m},{"orientation_rad",carried_orientation_margin_rad}}){
   const auto key=parameter_namespace+".carried_margin."+entry.first;
   if(node->has_parameter(key)){
    if(node->get_parameter(key).get_type()!=rclcpp::ParameterType::PARAMETER_DOUBLE || node->get_parameter(key).as_double()!=entry.second)return false;
   }else{rcl_interfaces::msg::ParameterDescriptor descriptor;descriptor.read_only=true;node->declare_parameter<double>(key,entry.second,descriptor);}
  }
  loader_=std::make_unique<pluginlib::ClassLoader<planning_interface::PlannerManager>>("moveit_core","planning_interface::PlannerManager");
  delegate_=loader_->createSharedInstance("ompl_interface/OMPLPlanner");
  if(!delegate_->initialize(model,node,parameter_namespace))return false;
  config_settings_=delegate_->getPlannerConfigurations();return true;
 }
 std::string getDescription()const override{return "OMPL with carried-path interior margin";}
 void getPlanningAlgorithms(std::vector<std::string>& algorithms)const override{delegate_->getPlanningAlgorithms(algorithms);}
 bool canServiceRequest(const planning_interface::MotionPlanRequest& request)const override{return delegate_ && delegate_->canServiceRequest(request);}
 planning_interface::PlanningContextPtr getPlanningContext(const planning_scene::PlanningSceneConstPtr& scene,const planning_interface::MotionPlanRequest& outer,moveit_msgs::msg::MoveItErrorCodes& error)const override{
  return carried_interior_context(*delegate_,scene,outer,error);
 }
 void setPlannerConfigurations(const planning_interface::PlannerConfigurationMap& configurations)override{
  planning_interface::PlannerManager::setPlannerConfigurations(configurations);if(delegate_)delegate_->setPlannerConfigurations(configurations);
 }
private:
 // The delegate must be destroyed before its plugin loader.
 std::unique_ptr<pluginlib::ClassLoader<planning_interface::PlannerManager>> loader_;
 planning_interface::PlannerManagerPtr delegate_;
};
}
PLUGINLIB_EXPORT_CLASS(sorting_control::CarriedMarginPlanner,planning_interface::PlannerManager)
