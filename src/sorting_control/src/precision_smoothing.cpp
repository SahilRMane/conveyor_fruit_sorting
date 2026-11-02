#include <sorting_control/precision_smoothing.hpp>
#include <moveit/online_signal_smoothing/smoothing_base_class.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <pluginlib/class_loader.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

namespace sorting_control {
class PrecisionAccelerationLimitedPlugin : public online_signal_smoothing::SmoothingBaseClass {
  using Base=online_signal_smoothing::SmoothingBaseClass;
  pluginlib::ClassLoader<Base> loader_{"moveit_core","online_signal_smoothing::SmoothingBaseClass"};
  std::shared_ptr<Base> original_;
  PrecisionStep step_;
  rclcpp::Node::SharedPtr node_;
public:
  bool initialize(rclcpp::Node::SharedPtr node,moveit::core::RobotModelConstPtr model,std::size_t count) override {
    node_=node;original_=loader_.createSharedInstance("online_signal_smoothing::AccelerationLimitedPlugin");
    if(!original_->initialize(node,model,count))return false;
    step_.dt=node->get_parameter("update_period").as_double();
    const auto group=node->get_parameter("planning_group_name").as_string();auto joints=model->getJointModelGroup(group);
    if(!joints || joints->getActiveJointModels().size()!=count || !(step_.dt>0 && step_.dt<=.02))return false;
    for(auto* vector:{&step_.qmin,&step_.qmax,&step_.vmin,&step_.vmax,&step_.amin,&step_.amax})vector->resize(count);
    std::size_t i=0;
    for(auto* joint:joints->getActiveJointModels()){
      const auto& b=joint->getVariableBounds(joint->getName());
      if(!b.position_bounded_ || !b.velocity_bounded_ || !b.acceleration_bounded_)return false;
      step_.qmin[i]=b.min_position_;step_.qmax[i]=b.max_position_;step_.vmin[i]=b.min_velocity_;step_.vmax[i]=b.max_velocity_;
      step_.amin[i]=b.min_acceleration_;step_.amax[i]=b.max_acceleration_;++i;
    }
    return true;
  }
  bool reset(const Eigen::VectorXd& p,const Eigen::VectorXd& v,const Eigen::VectorXd& a) override {
    step_.reset(p,v);return original_->reset(p,v,a);
  }
  bool doSmoothing(Eigen::VectorXd& p,Eigen::VectorXd& v,Eigen::VectorXd& a) override {
    if(!step_.ready)throw std::runtime_error("Precision smoother was not reset from a measured state");
    try{
      const auto requested_p=p,requested_v=v;
      if(!original_->doSmoothing(p,v,a))throw std::runtime_error("Installed acceleration smoother failed");
      if(step_.repair(requested_p,requested_v,p,v,a))original_->reset(p,v,a);
      return true;
    }catch(const std::exception& error){
      // Servo ignores the boolean return from smoothing. Supply a hold explicitly.
      p=step_.position;v=Eigen::VectorXd::Zero(p.size());a=v;
      step_.reset(p,v);original_->reset(p,v,a);
      RCLCPP_ERROR(node_->get_logger(),"Precision smoother stopped: %s",error.what());return false;
    }
  }
};
}
PLUGINLIB_EXPORT_CLASS(sorting_control::PrecisionAccelerationLimitedPlugin,online_signal_smoothing::SmoothingBaseClass)
