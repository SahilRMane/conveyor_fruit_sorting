// Standalone URDF/MuJoCo optical-frame agreement. No ROS node.
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <iostream>
void require(bool ok,const std::string& text){if(!ok)throw std::runtime_error(text);}
std::string read(const char* file){std::ifstream in(file);require(bool(in),"Cannot read model");std::ostringstream out;out<<in.rdbuf();return out.str();}
int main(int argc,char** argv){
  if(argc!=6){std::cerr<<"Usage: verify_camera_model URDF SRDF MJCF RENDER_YAML OUTPUT_JSON\n";return 2;}
  mjModel* m=nullptr;mjData* d=nullptr;
  try{
    auto urdf=urdf::parseURDF(read(argv[1]));require(bool(urdf),"Invalid URDF");auto srdf=std::make_shared<srdf::Model>();
    require(srdf->initString(*urdf,read(argv[2])),"Invalid SRDF");auto model=std::make_shared<moveit::core::RobotModel>(urdf,srdf);
    require(model->getModelFrame()=="world","Use the fixed world frame");
    char error[2048]={};m=mj_loadXML(argv[3],nullptr,error,sizeof(error));require(m,error);d=mj_makeData(m);
    auto fixtures=YAML::LoadFile(argv[4]);double max_position=0,max_angle=0;int count=0;
    for(const auto& frame:fixtures["frames"]){
      mj_resetDataKeyframe(m,d,mj_name2id(m,mjOBJ_KEY,"home"));moveit::core::RobotState state(model);state.setToDefaultValues();
      for(int i=0;i<7;++i){double q=frame["joint_positions"][i].as<double>();d->qpos[i]=q;state.setVariablePosition("panda_joint"+std::to_string(i+1),q);}
      state.setVariablePosition("panda_finger_joint1",0.04);state.setVariablePosition("panda_finger_joint2",0.04);state.update();mj_forward(m,d);
      auto view=frame["view_id"].as<std::string>();int camera=mj_name2id(m,mjOBJ_CAMERA,view.c_str());require(camera>=0,"Missing MuJoCo camera");
      const auto& tf=state.getGlobalLinkTransform(view+"_camera_optical_frame");
      Eigen::Matrix3d gl=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(d->cam_xmat+9*camera);
      Eigen::Matrix3d conversion=Eigen::Vector3d(1,-1,-1).asDiagonal();
      double position=(tf.translation()-Eigen::Map<Eigen::Vector3d>(d->cam_xpos+3*camera)).norm();
      double angle=Eigen::AngleAxisd(tf.rotation()*(gl*conversion).transpose()).angle();
      max_position=std::max(max_position,position);max_angle=std::max(max_angle,angle);++count;
      require(position<0.001 && angle<0.005,"URDF/MuJoCo optical-frame mismatch");
    }
    std::ofstream output(argv[5]);require(bool(output),"Cannot write frame report");
    output<<"{\"status\":\"PASS\",\"checked_camera_poses\":"<<count<<",\"maximum_position_error_m\":"<<max_position<<",\"maximum_orientation_error_rad\":"<<max_angle<<"}\n";
    std::cout<<"PASS: URDF/MuJoCo camera optical frames\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(d)mj_deleteData(d);if(m)mj_deleteModel(m);return 1;}
  mj_deleteData(d);mj_deleteModel(m);return 0;
}
