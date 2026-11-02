#include "sorting_simulation/contact_state.hpp"
#include "sorting_simulation/reset_record.hpp"
#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/msg/string.hpp>
#include <memory>
#include <atomic>

namespace sorting_simulation {
class ContactMonitor : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase {
  std::unique_ptr<SceneIds> ids_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr reset_publisher_;
  const mjModel* model_=nullptr;
  std::atomic<std::uint64_t> reset_seq_{0};
public:
  bool init(rclcpp::Node::SharedPtr node,const mjModel* model,mjData*) override {
    try {
      ids_=std::make_unique<SceneIds>(model);
      publisher_=node->create_publisher<std_msgs::msg::String>("/sorting/contact_state",10);
      reset_publisher_=node->create_publisher<std_msgs::msg::String>(
          "/sorting/reset_state",rclcpp::QoS(1).reliable().transient_local());
      model_=model;
      return true;
    } catch(const std::exception& e) { RCLCPP_ERROR(node->get_logger(),"%s",e.what()); return false; }
  }
  void update(const mjModel* model,mjData* data) override {
    std_msgs::msg::String msg;msg.data=state_json(sample(model,data,*ids_));
    msg.data.pop_back();msg.data+=",\"reset_seq\":"+std::to_string(reset_seq_.load())+"}";
    publisher_->publish(msg);
  }
  void on_reset(mjData* data) override {
    auto sequence=reset_seq_.fetch_add(1)+1;
    std_msgs::msg::String msg;msg.data=reset_json(model_,data,sequence);reset_publisher_->publish(msg);
  }
  void cleanup() override { publisher_.reset();reset_publisher_.reset();ids_.reset();model_=nullptr; }
};
}
PLUGINLIB_EXPORT_CLASS(sorting_simulation::ContactMonitor,
                      mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
