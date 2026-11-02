#pragma once
#include "sorting_simulation/contact_state.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <vector>

namespace sorting_simulation {
using ArmVector = Eigen::Matrix<double,7,1>;
inline void check(bool ok,const std::string& message) {
  if(!ok) throw std::runtime_error(message);
}
inline double distance3(const std::array<double,3>& a,const std::array<double,3>& b) {
  double sum=0;for(int i=0;i<3;++i) sum+=(a[i]-b[i])*(a[i]-b[i]);return std::sqrt(sum);
}
struct StaticMotion {
  mjModel* model;
  mjData* data;
  mjData* kinematics;
  SceneIds ids;
  std::function<void(const ContactState&)> guard;

  explicit StaticMotion(mjModel* m):model(m),data(mj_makeData(m)),kinematics(mj_makeData(m)),ids(m) {
    check(model->nq==16 && model->nv==15 && model->nu==8,"Unexpected scene dimensions");
    for(int i=0;i<7;++i) {
      std::string name="panda_joint"+std::to_string(i+1);
      int joint=mj_name2id(model,mjOBJ_JOINT,name.c_str());
      check(joint>=0 && model->jnt_qposadr[joint]==i && model->jnt_dofadr[joint]==i &&
            mj_name2id(model,mjOBJ_ACTUATOR,name.c_str())==i,"Unexpected Panda joint or actuator order");
    }
    check(ids.gripper==7 && model->actuator_forcelimited[7] &&
          std::fabs(model->actuator_forcerange[14]+20)<1e-9 &&
          std::fabs(model->actuator_forcerange[15]-20)<1e-9,"Expected gripper actuator force limit +/-20 N");
    int home=mj_name2id(model,mjOBJ_KEY,"home");
    check(home>=0,"Missing home keyframe");
    mj_resetDataKeyframe(model,data,home);mj_forward(model,data);
  }
  ~StaticMotion() { mj_deleteData(kinematics);mj_deleteData(data); }
  ContactState state() const { return sample(model,data,ids); }

  void step(double seconds) {
    double end=data->time+seconds;
    while(data->time<end) {
      mj_step(model,data);
      if(guard) guard(state());
      for(int i=0;i<mjNWARNING;++i) check(!data->warning[i].number,"MuJoCo physics warning");
    }
    mj_forward(model,data);
  }

  ArmVector solve(const std::array<double,3>& target) {
    mj_copyData(kinematics,model,data);mju_zero(kinematics->qvel,model->nv);
    Eigen::Matrix3d rotation;rotation<<0,1,0, 1,0,0, 0,0,-1;
    Eigen::Vector3d position=Eigen::Map<const Eigen::Vector3d>(target.data());
    ArmVector q;for(int i=0;i<7;++i) q[i]=data->qpos[i];
    bool solved=false;
    for(int iteration=0;iteration<400;++iteration) {
      for(int i=0;i<7;++i) kinematics->qpos[i]=q[i];
      mj_forward(model,kinematics);
      Eigen::Vector3d p=Eigen::Map<Eigen::Vector3d>(kinematics->xpos+3*ids.hand);
      Eigen::Matrix3d current=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(kinematics->xmat+9*ids.hand);
      Eigen::Vector3d angle=Eigen::Vector3d::Zero();
      for(int i=0;i<3;++i) angle+=0.5*current.col(i).cross(rotation.col(i));
      if((position-p).norm()<1e-6 && (current-rotation).norm()<1e-5) {solved=true;break;}
      Eigen::Matrix<double,6,1> error;error<<position-p,angle;
      std::vector<mjtNum> jp(3*model->nv),jr(3*model->nv);
      mj_jacBody(model,kinematics,jp.data(),jr.data(),ids.hand);
      Eigen::Matrix<double,6,7> jacobian;
      for(int r=0;r<3;++r) for(int c=0;c<7;++c) {
        jacobian(r,c)=jp[r*model->nv+c];jacobian(r+3,c)=jr[r*model->nv+c];
      }
      ArmVector delta=jacobian.transpose()*(jacobian*jacobian.transpose()+
          0.0001*Eigen::Matrix<double,6,6>::Identity()).ldlt().solve(error);
      if(delta.norm()>0.1) delta*=0.1/delta.norm();
      q+=delta;
      for(int i=0;i<7;++i) q[i]=std::clamp(q[i],model->actuator_ctrlrange[2*i]+0.02,
                                         model->actuator_ctrlrange[2*i+1]-0.02);
    }
    check(solved,"Inverse kinematics did not converge");
    ArmVector command=q;
    for(int i=0;i<7;++i) {
      double kp=model->actuator_gainprm[i*mjNGAIN];
      check(kp>0 && std::fabs(model->actuator_biasprm[i*mjNBIAS+1]+kp)<1e-6,"Expected Panda position servos");
      command[i]+=kinematics->qfrc_bias[i]/kp;
      check(command[i]>=model->actuator_ctrlrange[2*i] &&
            command[i]<=model->actuator_ctrlrange[2*i+1],"Arm command exceeds actuator range");
    }
    return command;
  }

  void move(const ArmVector& command,double duration,double settle) {
    // Match interpolate_from_desired_state=true: preserve the last servo targets.
    ArmVector start;for(int i=0;i<7;++i) start[i]=data->ctrl[i];
    double begin=data->time;
    while(data->time-begin<duration) {
      double u=std::clamp((data->time-begin)/duration,0.0,1.0),blend=u*u*(3-2*u);
      for(int i=0;i<7;++i) data->ctrl[i]=start[i]+blend*(command[i]-start[i]);
      step(model->opt.timestep);
    }
    for(int i=0;i<7;++i) data->ctrl[i]=command[i];
    step(settle);
  }
};
}
