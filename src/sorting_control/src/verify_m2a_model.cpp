#include <moveit/robot_model/robot_model.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

void require(bool ok,const std::string& why){if(!ok) throw std::runtime_error(why);}
std::string read(const char* path){std::ifstream in(path);require(bool(in),"Cannot read input");std::ostringstream out;out<<in.rdbuf();return out.str();}
int main(int argc,char** argv) {
  if(argc!=6){std::cerr<<"Usage: verify_m2a_model URDF SRDF FIXTURES_YAML MJCF OUTPUT_JSON\n";return 2;}
  std::ofstream output(argv[5]);if(!output)return 2;
  mjModel* mj=nullptr;mjData* d=nullptr;
  try {
    auto description=urdf::parseURDF(read(argv[1]));require(bool(description),"URDF parse failed");
    auto semantic=std::make_shared<srdf::Model>();require(semantic->initString(*description,read(argv[2])),"SRDF parse failed");
    auto robot=std::make_shared<moveit::core::RobotModel>(description,semantic);
    const auto* group=robot->getJointModelGroup("panda_arm");require(group,"Missing panda_arm group");
    std::vector<std::string> names;for(int i=1;i<=7;++i)names.push_back("panda_joint"+std::to_string(i));
    require(group->getActiveJointModelNames()==names,"Unexpected Servo joint order");
    require(robot->getModelFrame()=="world","Expected world planning frame");
    planning_scene::PlanningScene scene(robot);
    auto fixtures=YAML::LoadFile(argv[3]);
    require(fixtures["frame"].as<std::string>()=="world","Unexpected fixture frame");
    for(const auto& box:fixtures["boxes"]) {
      moveit_msgs::msg::CollisionObject obj;obj.id=box["id"].as<std::string>();obj.header.frame_id="world";
      obj.operation=moveit_msgs::msg::CollisionObject::ADD;
      shape_msgs::msg::SolidPrimitive primitive;primitive.type=shape_msgs::msg::SolidPrimitive::BOX;
      for(const auto& v:box["dimensions"])primitive.dimensions.push_back(v.as<double>());
      geometry_msgs::msg::Pose pose;pose.orientation.w=1;
      pose.position.x=box["position"][0].as<double>();pose.position.y=box["position"][1].as<double>();pose.position.z=box["position"][2].as<double>();
      obj.primitives.push_back(primitive);obj.primitive_poses.push_back(pose);
      require(scene.processCollisionObjectMsg(obj),"Fixture insertion failed");
    }
    scene.getAllowedCollisionMatrixNonConst().setEntry("static_floor","panda_link0",true);
    char error[2048]={};mj=mj_loadXML(argv[4],nullptr,error,sizeof(error));require(mj,error);
    d=mj_makeData(mj);int home=mj_name2id(mj,mjOBJ_KEY,"home");require(home>=0,"Missing home keyframe");
    mj_resetDataKeyframe(mj,d,home);mj_forward(mj,d);
    int hand=mj_name2id(mj,mjOBJ_BODY,"hand");require(hand>=0,"Missing MuJoCo hand");
    double maximum_position_error=0,maximum_angle_error=0,maximum_speed=0;
    int samples=0;
    auto check=[&] {
      auto& state=scene.getCurrentStateNonConst();state.setToDefaultValues();
      for(int i=0;i<7;++i)state.setVariablePosition(names[i],d->qpos[i]);
      for(int i=1;i<=2;++i) {
        std::string name="panda_finger_joint"+std::to_string(i);
        int joint=mj_name2id(mj,mjOBJ_JOINT,name.c_str());require(joint>=0,"Missing finger joint");
        state.setVariablePosition(name,d->qpos[mj->jnt_qposadr[joint]]);
      }
      state.update();
      collision_detection::CollisionRequest request;collision_detection::CollisionResult result;
      request.contacts=true;request.max_contacts=100;scene.checkCollision(request,result,state);
      if(result.collision){for(const auto& pair:result.contacts)std::cerr<<pair.first.first<<" / "<<pair.first.second<<'\n';}
      require(!result.collision,"MoveIt collision in stationary probe");
      const auto& transform=state.getGlobalLinkTransform("panda_hand");
      Eigen::Vector3d position(d->xpos[3*hand],d->xpos[3*hand+1],d->xpos[3*hand+2]);
      Eigen::Matrix3d rotation=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(d->xmat+9*hand);
      double position_error=(transform.translation()-position).norm();
      double angle_error=Eigen::AngleAxisd(transform.rotation().transpose()*rotation).angle();
      maximum_position_error=std::max(maximum_position_error,position_error);
      maximum_angle_error=std::max(maximum_angle_error,angle_error);
      require(position_error<0.001 && angle_error<0.001,"URDF and MuJoCo hand frames differ");
      for(int i=0;i<7;++i){maximum_speed=std::max(maximum_speed,std::fabs(d->qvel[i]));require(std::fabs(d->qvel[i])<0.25,"Probe joint speed exceeded bound");}
      ++samples;
    };
    auto advance=[&](double seconds){int count=std::lround(seconds/mj->opt.timestep);for(int i=0;i<count;++i){mj_step(mj,d);if(i%10==0)check();}mj_forward(mj,d);check();};
    advance(2);
    for(int cycle=0;cycle<3;++cycle) {
      double begin=d->ctrl[0];
      for(int i=0;i<1000;++i){double u=static_cast<double>(i)/1000;d->ctrl[0]=begin+0.03*u*u*(3-2*u);advance(mj->opt.timestep);}
      advance(0.3);
      // Match handoff seeding from measured joint positions. This is a physics
      // probe of the target envelope, not a substitute for live MoveIt Servo.
      for(int i=0;i<7;++i)d->ctrl[i]=d->qpos[i];
      advance(0.5);
      for(int i=0;i<400;++i){d->ctrl[0]+=0.04*mj->opt.timestep;advance(mj->opt.timestep);}
      advance(0.3);
      for(int i=0;i<7;++i)d->ctrl[i]=d->qpos[i];
      advance(0.5);
    }
    output<<"{\"status\":\"PASS\",\"scope\":\"URDF/MJCF frame and collision check with small target motions\","
          <<"\"samples\":"<<samples<<",\"fixture_boxes\":"<<fixtures["boxes"].size()
          <<",\"maximum_hand_position_error_m\":"<<maximum_position_error
          <<",\"maximum_hand_orientation_error_rad\":"<<maximum_angle_error
          <<",\"maximum_joint_speed_rad_s\":"<<maximum_speed<<",\"live_ros_ownership\":\"NOT_RUN\"}\n";
    std::cout<<"PASS: MoveIt/MuJoCo hand frames, fixture collisions and small-motion envelope\n";
  } catch(const std::exception& e){output<<"{\"status\":\"FAIL\"}\n";std::cerr<<"FAIL: "<<e.what()<<'\n';if(d)mj_deleteData(d);if(mj)mj_deleteModel(mj);return 1;}
  mj_deleteData(d);mj_deleteModel(mj);return 0;
}
