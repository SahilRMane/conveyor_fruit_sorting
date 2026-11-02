#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <sorting_interfaces/msg/camera_epoch.hpp>
#include <sorting_interfaces/msg/camera_truth.hpp>
#include <atomic>
#include <cmath>
#include <array>

namespace sorting_simulation {
class CameraEvidence : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase {
  rclcpp::Publisher<sorting_interfaces::msg::CameraEpoch>::SharedPtr epoch_pub_;
  rclcpp::Publisher<sorting_interfaces::msg::CameraTruth>::SharedPtr truth_pub_;
  std::atomic<uint64_t> episode_{0};
  int marker_=-1,handle_=-1;
  std::array<int,2> cameras_{};
  static builtin_interfaces::msg::Time stamp(double seconds){
    return rclcpp::Time(static_cast<int64_t>(seconds*1e9),RCL_ROS_TIME);
  }
  void epoch(double time){
    sorting_interfaces::msg::CameraEpoch msg;msg.header.frame_id="world";msg.header.stamp=stamp(time);
    msg.episode_id=episode_.load();epoch_pub_->publish(msg);
  }
public:
  bool init(rclcpp::Node::SharedPtr node,const mjModel* model,mjData* data) override {
    marker_=mj_name2id(model,mjOBJ_GEOM,"neutral_localization_marker");
    handle_=mj_name2id(model,mjOBJ_SITE,"handle_grasp_site");
    cameras_={mj_name2id(model,mjOBJ_CAMERA,"overhead"),mj_name2id(model,mjOBJ_CAMERA,"wrist")};
    if(marker_<0 || handle_<0 || cameras_[0]<0 || cameras_[1]<0)return false;
    epoch_pub_=node->create_publisher<sorting_interfaces::msg::CameraEpoch>("/sorting/episode",rclcpp::QoS(1).reliable().transient_local());
    const std::string parameter="mujoco_plugins."+node->get_sub_namespace()+".enable_evaluator";
    if(!node->has_parameter(parameter))node->declare_parameter(parameter,false);
    if(node->get_parameter(parameter).as_bool())
      truth_pub_=node->create_publisher<sorting_interfaces::msg::CameraTruth>("/sorting/ground_truth/camera_sample",rclcpp::QoS(200).reliable());
    epoch(data->time);return true;
  }
  void update(const mjModel* model,mjData* data) override {
    if(!truth_pub_)return;
    for(int view=0;view<2;++view){
      int cam=cameras_[view];sorting_interfaces::msg::CameraTruth msg;
      msg.header.frame_id="world";msg.header.stamp=stamp(data->time);msg.episode_id=episode_.load();
      msg.view_id=view==0?"overhead":"wrist";msg.camera_frame=msg.view_id+"_camera_optical_frame";
      auto* p=data->cam_xpos+3*cam;msg.camera_pose.position.x=p[0];msg.camera_pose.position.y=p[1];msg.camera_pose.position.z=p[2];
      // MuJoCo camera looks along -z, with +y up. ROS optical is +z forward, +y down.
      mjtNum optical[9],q[4];for(int r=0;r<3;++r)for(int c=0;c<3;++c)
        optical[3*r+c]=data->cam_xmat[9*cam+3*r+c]*(c==0?1:-1);
      mju_mat2Quat(q,optical);
      msg.camera_pose.orientation.w=q[0];msg.camera_pose.orientation.x=q[1];msg.camera_pose.orientation.y=q[2];msg.camera_pose.orientation.z=q[3];
      auto* g=data->geom_xpos+3*marker_;auto* matrix=data->geom_xmat+9*marker_;
      double dz=model->geom_size[3*marker_+2];
      msg.marker_position.x=g[0]+matrix[2]*dz;msg.marker_position.y=g[1]+matrix[5]*dz;msg.marker_position.z=g[2]+matrix[8]*dz;
      auto* h=data->site_xpos+3*handle_;msg.handle_position.x=h[0];msg.handle_position.y=h[1];msg.handle_position.z=h[2];
      truth_pub_->publish(msg);
    }
  }
  void on_reset(mjData* data) override {episode_.fetch_add(1);epoch(data->time);}
  void cleanup() override {truth_pub_.reset();epoch_pub_.reset();}
};
}
PLUGINLIB_EXPORT_CLASS(sorting_simulation::CameraEvidence,mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
