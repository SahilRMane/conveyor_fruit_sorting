#include "sorting_simulation/conveyor_transport.hpp"
#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <atomic>
#include <chrono>
#include <memory>

namespace sorting_simulation {
class ConveyorPlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase {
  std::unique_ptr<ConveyorTransport> transport_;
  std::unique_ptr<SceneIds> ids_;
  std::atomic<double> speed_{0};
  std::atomic<std::int64_t> received_{0};
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr command_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_;
  static std::int64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
public:
  bool init(rclcpp::Node::SharedPtr node,const mjModel* model,mjData*) override {
    try {
      transport_=std::make_unique<ConveyorTransport>(model);ids_=std::make_unique<SceneIds>(model);
      state_=node->create_publisher<std_msgs::msg::String>("/sorting/ground_truth/conveyor_state",10);
      command_=node->create_subscription<std_msgs::msg::Float64>("/sorting/conveyor_command",1,
        [this,node](std_msgs::msg::Float64::ConstSharedPtr msg) {
          bool valid=std::isfinite(msg->data) && msg->data>=0 && msg->data<=0.10;
          if(!valid) RCLCPP_ERROR(node->get_logger(),"Conveyor command rejected; use 0..0.10 m/s");
          speed_.store(valid?msg->data:0);received_.store(now());
        });
      return true;
    } catch(const std::exception& e) {RCLCPP_ERROR(node->get_logger(),"%s",e.what());return false;}
  }
  void pre_step(mjData* data) override {
    bool fresh=received_.load()>0 && now()-received_.load()<500000000;
    transport_->apply(data,speed_.load(),fresh);
  }
  void update(const mjModel* model,mjData* data) override {
    int joint=mj_name2id(model,mjOBJ_JOINT,"tray_free_joint");
    double force=data->xfrc_applied[6*ids_->tray];
    auto contact=sample(model,data,*ids_);
    std::ostringstream out;out<<std::setprecision(12)
      <<"{\"sim_time\":"<<data->time<<",\"mode\":\"bounded_transport_force\",\"oracle_only\":true"
      <<",\"requested_speed_m_s\":"<<speed_.load()<<",\"tray_vx_m_s\":"<<data->qvel[model->jnt_dofadr[joint]]
      <<",\"force_x_N\":"<<force<<",\"force_limit_N\":8,\"support_force_N\":"<<contact.support_N
      <<",\"tray_xyz\":";array_json(out,contact.tray);
    out<<",\"tray_quat_wxyz\":";array_json(out,contact.tray_quat);
    out<<",\"forbidden_contacts\":"<<contact.forbidden
       <<",\"command_fresh\":"<<((received_.load()>0 && now()-received_.load()<500000000)?"true":"false")<<'}';
    std_msgs::msg::String message;message.data=out.str();state_->publish(message);
  }
  void on_reset(mjData*) override {transport_->reset();speed_.store(0);received_.store(0);}
  void cleanup() override {command_.reset();state_.reset();transport_.reset();ids_.reset();}
};
}
PLUGINLIB_EXPORT_CLASS(sorting_simulation::ConveyorPlugin,mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
