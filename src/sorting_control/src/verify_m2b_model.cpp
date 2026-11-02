// Standalone model/OMPL check. No ROS node or DDS participant is created.
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/util/RandomNumbers.h>
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
std::string read(const char* path){std::ifstream in(path);require(bool(in),"Cannot read input");std::ostringstream out;out<<in.rdbuf();return out.str();}
using Q=Eigen::Matrix<double,7,1>;
int main(int argc,char** argv){
  if(argc!=7){std::cerr<<"Usage: verify_m2b_model URDF SRDF FIXTURES MJCF CONFIG OUTPUT\n";return 2;}
  std::ofstream output(argv[6]);if(!output)return 2;
  mjModel* m=nullptr;mjData* d=nullptr;mjData* k=nullptr;
  try{
    auto urdf=urdf::parseURDF(read(argv[1]));require(bool(urdf),"Invalid URDF");
    auto srdf=std::make_shared<srdf::Model>();require(srdf->initString(*urdf,read(argv[2])),"Invalid SRDF");
    auto model=std::make_shared<moveit::core::RobotModel>(urdf,srdf);
    const auto* group=model->getJointModelGroup("panda_arm");require(group && model->getModelFrame()=="world","Invalid group/frame");
    planning_scene::PlanningScene scene(model);
    auto fixture=YAML::LoadFile(argv[3]), config=YAML::LoadFile(argv[5]);
    for(const auto& b:fixture["boxes"]){
      moveit_msgs::msg::CollisionObject o;o.id=b["id"].as<std::string>();o.header.frame_id="world";
      o.pose.orientation.w=1;o.operation=moveit_msgs::msg::CollisionObject::ADD;
      shape_msgs::msg::SolidPrimitive shape;shape.type=shape.BOX;
      for(const auto& size:b["dimensions"])shape.dimensions.push_back(size.as<double>());
      geometry_msgs::msg::Pose pose;pose.orientation.w=1;
      pose.position.x=b["position"][0].as<double>();pose.position.y=b["position"][1].as<double>();pose.position.z=b["position"][2].as<double>();
      o.primitives={shape};o.primitive_poses={pose};require(scene.processCollisionObjectMsg(o),"Invalid fixture");
    }
    scene.getAllowedCollisionMatrixNonConst().setEntry("static_floor","panda_link0",true);
    char error[1024]={};m=mj_loadXML(argv[4],nullptr,error,sizeof(error));require(m,error);d=mj_makeData(m);k=mj_makeData(m);
    mj_resetDataKeyframe(m,d,mj_name2id(m,mjOBJ_KEY,"home"));
    int hand=mj_name2id(m,mjOBJ_BODY,"hand"),handle=mj_name2id(m,mjOBJ_SITE,"handle_grasp_site");require(hand>=0 && handle>=0,"Missing frames");
    for(int i=0;i<1000;++i){mj_step(m,d);}
    mj_forward(m,d);mj_copyData(k,m,d);
    Eigen::Vector3d target=Eigen::Map<Eigen::Vector3d>(d->site_xpos+3*handle);
    target.z()+=config["hand_above_handle_m"].as<double>()+config["pregrasp_clearance_m"].as<double>();
    auto qc=config["grasp_quaternion_xyzw"];
    Eigen::Quaterniond desired(qc[3].as<double>(),qc[0].as<double>(),qc[1].as<double>(),qc[2].as<double>());
    Eigen::Matrix3d rotation=desired.toRotationMatrix();
    Q lower,upper;
    auto names=group->getActiveJointModelNames();require(names.size()==7,"Unexpected arm joints");
    for(int i=0;i<7;++i){const auto& bounds=model->getVariableBounds(names[i]);lower[i]=bounds.min_position_;upper[i]=bounds.max_position_;}
    auto solve=[&](const Eigen::Vector3d& p,const Eigen::Matrix3d& r,Q seed){
      std::vector<mjtNum> jp(3*m->nv),jr(3*m->nv);
      for(int step=0;step<1500;++step){
        for(int i=0;i<7;++i){k->qpos[i]=seed[i];}
        mj_forward(m,k);
        Eigen::Vector3d here=Eigen::Map<Eigen::Vector3d>(k->xpos+3*hand);
        Eigen::Matrix3d current=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(k->xmat+9*hand);
        Eigen::AngleAxisd angle(r*current.transpose());Eigen::Matrix<double,6,1> e;e<<p-here,angle.axis()*angle.angle();
        if(e.head<3>().norm()<1e-6 && e.tail<3>().norm()<1e-5)return seed;
        mj_jacBody(m,k,jp.data(),jr.data(),hand);Eigen::Matrix<double,6,7> j;
        for(int row=0;row<3;++row)for(int col=0;col<7;++col){j(row,col)=jp[row*m->nv+col];j(row+3,col)=jr[row*m->nv+col];}
        Q delta=j.transpose()*(j*j.transpose()+Eigen::Matrix<double,6,6>::Identity()*1e-5).ldlt().solve(e);
        if(delta.norm()>0.08){delta*=0.08/delta.norm();}
        seed=(seed+delta).cwiseMax(lower).cwiseMin(upper);
      }
      throw std::runtime_error("Pregrasp/axis inverse kinematics failed");
    };
    double maximum_frame_error=0,maximum_rotation_error=0;int checked=0;
    auto valid=[&](const Q& q){
      moveit::core::RobotState state(model);state.setToDefaultValues();std::vector<double> values(q.data(),q.data()+7);
      state.setJointGroupPositions(group,values);state.setVariablePosition("panda_finger_joint1",0.04);state.setVariablePosition("panda_finger_joint2",0.04);state.update();
      if(!state.satisfiesBounds(group) || scene.isStateColliding(state,"panda_arm"))return false;
      for(int i=0;i<7;++i){k->qpos[i]=q[i];}
      mj_forward(m,k);
      auto tf=state.getGlobalLinkTransform("panda_hand");
      double pe=(tf.translation()-Eigen::Map<Eigen::Vector3d>(k->xpos+3*hand)).norm();
      Eigen::Matrix3d r=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(k->xmat+9*hand);
      double re=Eigen::AngleAxisd(tf.rotation()*r.transpose()).angle();
      maximum_frame_error=std::max(maximum_frame_error,pe);maximum_rotation_error=std::max(maximum_rotation_error,re);++checked;
      require(pe<0.001 && re<0.002,"MoveIt/MuJoCo frame mismatch");return true;
    };
    Q start=Eigen::Map<Q>(d->qpos),goal=solve(target,rotation,start);require(valid(start)&&valid(goal),"Start or pregrasp collision");
    ompl::RNG::setSeed(17);auto space=std::make_shared<ompl::base::RealVectorStateSpace>(7);
    ompl::base::RealVectorBounds bounds(7);for(int i=0;i<7;++i){bounds.setLow(i,lower[i]);bounds.setHigh(i,upper[i]);}space->setBounds(bounds);
    ompl::geometric::SimpleSetup setup(space);setup.setStateValidityChecker([&](const ompl::base::State* state){
      return valid(Eigen::Map<const Q>(state->as<ompl::base::RealVectorStateSpace::StateType>()->values));});
    setup.getSpaceInformation()->setStateValidityCheckingResolution(0.001);
    ompl::base::ScopedState<> begin(space),end(space);for(int i=0;i<7;++i){begin[i]=start[i];end[i]=goal[i];}
    setup.setStartAndGoalStates(begin,end);auto planner=std::make_shared<ompl::geometric::RRTConnect>(setup.getSpaceInformation());planner->setRange(0.10);setup.setPlanner(planner);
    require(setup.solve(5.0)==ompl::base::PlannerStatus::EXACT_SOLUTION,"Native OMPL pregrasp path failed");
    auto path=setup.getSolutionPath();path.interpolate(300);
    for(const auto* point:path.getStates())require(valid(Eigen::Map<const Q>(point->as<ompl::base::RealVectorStateSpace::StateType>()->values)),"Native path collision");
    Q seed=goal;Eigen::Vector3d p=target;
    for(int axis=0;axis<3;++axis){
      for(int step=1;step<=20;++step){Eigen::Vector3d probe=p;probe[axis]+=config["servo_axis_speed_m_s"].as<double>()*config["servo_axis_duration_s"].as<double>()*step/20.0;
        seed=solve(probe,rotation,seed);require(valid(seed),"Cartesian axis envelope collision");}
      p[axis]+=config["servo_axis_speed_m_s"].as<double>()*config["servo_axis_duration_s"].as<double>();
    }
    seed=solve(p,Eigen::AngleAxisd(0.012,Eigen::Vector3d::UnitZ()).toRotationMatrix()*rotation,seed);require(valid(seed),"Yaw envelope collision");
    // Follow the native planned path with the same position actuators. This
    // quantifies compliance; it is not a run of MoveIt trajectory execution.
    for(int step=0;step<1500;++step){int index=std::min(299,step*300/1500);const auto* state=path.getState(index)->as<ompl::base::RealVectorStateSpace::StateType>();
      for(int i=0;i<7;++i){d->ctrl[i]=state->values[i];}
      mj_step(m,d);
      if(step%10==0){mj_forward(m,d);require(valid(Eigen::Map<Q>(d->qpos)),"Physical path collision");}
    }
    for(int i=0;i<7;++i){d->ctrl[i]=goal[i];}
    for(int step=0;step<500;++step){mj_step(m,d);}
    mj_forward(m,d);
    double physical_error=(Eigen::Map<Eigen::Vector3d>(d->xpos+3*hand)-target).norm();
    require(physical_error<config["planned_position_tolerance_m"].as<double>(),"Physical pregrasp position exceeds compliance allowance");
    require(valid(Eigen::Map<Q>(d->qpos)),"Physical pregrasp state collision");
    output<<"{\"status\":\"PASS\",\"scope\":\"native OMPL/FCL geometry and frame envelope; live MoveIt service and Servo NOT_RUN\","
          <<"\"checked_states\":"<<checked<<",\"planned_path_samples\":300,\"maximum_frame_error_m\":"<<maximum_frame_error
          <<",\"maximum_rotation_error_rad\":"<<maximum_rotation_error<<",\"physical_pregrasp_position_error_m\":"<<physical_error
          <<",\"pregrasp_xyz\":["<<target.x()<<','<<target.y()<<','<<target.z()<<"]}\n";
    std::cout<<"PASS: native pregrasp path, Cartesian envelope and frame agreement\n";
  }catch(const std::exception& e){output<<"{\"status\":\"FAIL\"}\n";std::cerr<<e.what()<<'\n';if(k)mj_deleteData(k);if(d)mj_deleteData(d);if(m)mj_deleteModel(m);return 1;}
  mj_deleteData(k);mj_deleteData(d);mj_deleteModel(m);return 0;
}
