// Read-only component test. A temporary ROS node supplies parameters only.
// No robot command publisher, action client, controller or simulator is created.
#include <moveit/online_signal_smoothing/smoothing_base_class.hpp>
#include <moveit/kinematics_base/kinematics_base.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <pluginlib/class_loader.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <array>
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::string read(const char* file){std::ifstream in(file);need(bool(in),"Cannot read component fixture");std::stringstream out;out<<in.rdbuf();return out.str();}
int main(int argc,char** argv){
 if(argc!=5){std::cerr<<"Usage: verify_servo_pipeline URDF SRDF JOINT_LIMITS OUTPUT_JSON\n";return 2;}
 try{
  rclcpp::init(0,nullptr);
  auto description=urdf::parseURDF(read(argv[1]));need(bool(description),"Invalid URDF");auto semantic=std::make_shared<srdf::Model>();need(semantic->initString(*description,read(argv[2])),"Invalid SRDF");
  auto model=std::make_shared<moveit::core::RobotModel>(description,semantic);auto group=model->getJointModelGroup("panda_arm");need(group && group->getActiveJointModelNames().size()==7,"Invalid Panda group");
  auto limits=YAML::LoadFile(argv[3])["joint_limits"];
  for(const auto& name:group->getActiveJointModelNames()){
    auto b=model->getVariableBounds(name);auto l=limits[name];need(l && l["has_acceleration_limits"].as<bool>(),"Acceleration limits missing");
    b.acceleration_bounded_=true;b.max_acceleration_=l["max_acceleration"].as<double>();b.min_acceleration_=-b.max_acceleration_;model->getJointModel(name)->setVariableBounds(name,b);
  }
  auto options=rclcpp::NodeOptions().enable_rosout(false).start_parameter_services(false).start_parameter_event_publisher(false).automatically_declare_parameters_from_overrides(true)
    .parameter_overrides({rclcpp::Parameter("update_period",.01),rclcpp::Parameter("planning_group_name","panda_arm"),rclcpp::Parameter("robot_description_kinematics.panda_arm.epsilon",1e-7)});
  auto node=std::make_shared<rclcpp::Node>("m3d_component_check",options);
  pluginlib::ClassLoader<kinematics::KinematicsBase> ik_loader("moveit_core","kinematics::KinematicsBase");
  auto ik=ik_loader.createSharedInstance("kdl_kinematics_plugin/KDLKinematicsPlugin");need(ik->initialize(node,*model,"panda_arm","panda_link0",{"panda_hand"},.005),"Installed KDL plugin initialization failed");
  pluginlib::ClassLoader<online_signal_smoothing::SmoothingBaseClass> filter_loader("moveit_core","online_signal_smoothing::SmoothingBaseClass");
  const std::array<std::array<double,7>,2> seeds={{{{.405006,.183100,.384295,-1.591311,-.069561,1.760882,.011239}},{{-.405651,.181179,-.381565,-1.604210,.068543,1.772181,-1.580676}}}};
  std::ostringstream report;report<<"{\n\"status\":\"PASS\",\"scope\":\"Installed KDL plus original and corrected acceleration smoothing; no actuation\",\n\"cases\":[\n";bool comma=false;
  for(const auto& seed:seeds)for(double speed:{.0003,.00025506,.000063765,.000038259}){
    double old_descent=0,new_descent=0;
    for(int variant=0;variant<2;++variant){
      auto filter=filter_loader.createSharedInstance(variant==0?"online_signal_smoothing::AccelerationLimitedPlugin":"sorting_control/PrecisionAccelerationLimitedPlugin");
      need(filter->initialize(node,model,7),"Smoothing plugin initialization failed");
      Eigen::VectorXd q=Eigen::Map<const Eigen::VectorXd>(seed.data(),7),v=Eigen::VectorXd::Zero(7),a=v;need(filter->reset(q,v,a),"Filter reset failed");
      moveit::core::RobotState state(model);state.setToDefaultValues();state.setJointGroupPositions(group,q);state.update();double start_z=state.getGlobalLinkTransform("panda_hand").translation().z();
      for(int cycle=0;cycle<150;++cycle){
        state.setJointGroupPositions(group,q);state.update();auto target=state.getGlobalLinkTransform("panda_hand");target.translation().z()-=speed*.01;
        Eigen::Quaterniond rotation(target.rotation());geometry_msgs::msg::Pose pose;pose.position.x=target.translation().x();pose.position.y=target.translation().y();pose.position.z=target.translation().z();pose.orientation.x=rotation.x();pose.orientation.y=rotation.y();pose.orientation.z=rotation.z();pose.orientation.w=rotation.w();
        std::vector<double> before(q.data(),q.data()+7),solution;moveit_msgs::msg::MoveItErrorCodes error;kinematics::KinematicsQueryOptions query;query.return_approximate_solution=true;
        need(ik->searchPositionIK(pose,before,.005,solution,error,query),"Installed KDL small-step solve failed");need(solution.size()==7,"Installed KDL returned the wrong joint count");
        Eigen::VectorXd requested=Eigen::Map<Eigen::VectorXd>(solution.data(),7),velocity=(requested-q)/.01,acceleration=Eigen::VectorXd::Zero(7);
        need(filter->doSmoothing(requested,velocity,acceleration),"Installed smoothing failed");
        need(requested.allFinite() && velocity.allFinite(),"Nonfinite component output");
        if(variant==1)for(int j=0;j<7;++j){
          const auto& b=model->getVariableBounds(group->getActiveJointModelNames()[j]);
          need(requested[j]>=b.min_position_-1e-10 && requested[j]<=b.max_position_+1e-10,"Component position limit failed");
          need(velocity[j]>=b.min_velocity_-1e-10 && velocity[j]<=b.max_velocity_+1e-10,"Component velocity limit failed");
          const double acceleration=(velocity[j]-v[j])/.01;
          need(acceleration>=b.min_acceleration_-1e-5 && acceleration<=b.max_acceleration_+1e-5,"Component acceleration limit failed");
        }
        q=requested;v=velocity;
      }
      state.setJointGroupPositions(group,q);state.update();double descent=start_z-state.getGlobalLinkTransform("panda_hand").translation().z();
      if(variant==0)old_descent=descent;else new_descent=descent;
    }
    std::cout<<"Component speed="<<speed<<" original_descent="<<old_descent<<" corrected_descent="<<new_descent<<std::endl;
    need(old_descent<speed*1.5*.25,"Installed filter did not reproduce the small-command stall; review version");
    need(new_descent>speed*1.5*.8 && new_descent<speed*1.5*1.1,"Corrected component pipeline did not resolve the commanded descent");
    if(comma)report<<",\n";comma=true;report<<"{\"speed_m_s\":"<<speed<<",\"original_descent_m\":"<<old_descent<<",\"corrected_descent_m\":"<<new_descent<<"}";
  }
  report<<"\n]}\n";std::ofstream saved(argv[4]);saved<<report.str();need(saved.good(),"Could not save component report");node.reset();rclcpp::shutdown();
  std::cout<<"PASS: eight installed KDL/acceleration-filter pipeline cases; no actuation\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(rclcpp::ok())rclcpp::shutdown();return 1;}
}
