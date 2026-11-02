// Solve the approach, then validate it through actuators and contact physics.
#include "sorting_simulation/contact_state.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace sorting_simulation;
using Q = Eigen::Matrix<double,7,1>;
struct Phase { std::string name; Q command; std::array<double,3> hand_target; ContactState measured; };
static void require(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }

int main(int argc,char** argv) {
  if(argc!=3) { std::cerr<<"Usage: plan_static_contact SCENE_XML OUTPUT_JSON\n";return 2; }
  std::ofstream output(argv[2]);
  if(!output) { std::cerr<<"Cannot write output\n";return 2; }
  char error[2048]={}; mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m) { output<<"{\"status\":\"FAIL\"}\n";std::cerr<<error<<'\n';return 1; }
  mjData* d=mj_makeData(m); mjData* kin=mj_makeData(m);
  try {
    require(m->nq==16 && m->nv==15 && m->nu==8,"Unexpected scene dimensions");
    SceneIds ids(m);
    for(int i=0;i<7;++i) {
      const std::string name="panda_joint"+std::to_string(i+1);
      int joint=mj_name2id(m,mjOBJ_JOINT,name.c_str());
      require(joint>=0 && m->jnt_qposadr[joint]==i && m->jnt_dofadr[joint]==i &&
              mj_name2id(m,mjOBJ_ACTUATOR,name.c_str())==i,"Unexpected arm joint or actuator order");
    }
    require(ids.gripper==7,"Unexpected gripper actuator order");
    int home=mj_name2id(m,mjOBJ_KEY,"home"), site=mj_name2id(m,mjOBJ_SITE,"handle_grasp_site");
    require(home>=0 && site>=0,"Missing home keyframe or grasp site");
    require(m->actuator_forcelimited[7] && std::fabs(m->actuator_forcerange[14]+20)<1e-9 &&
            std::fabs(m->actuator_forcerange[15]-20)<1e-9,"Run the updated scene preparation script: gripper limit must be +/-20 N");
    mj_resetDataKeyframe(m,d,home); mj_forward(m,d);
    auto step=[&](double seconds,bool closing) {
      const double end=d->time+seconds;
      while(d->time<end) {
        mj_step(m,d); ContactState state=sample(m,d,ids);
        require(state.forbidden==0,"Unexpected robot contact with tray or support");
        require(closing || state.left_N+state.right_N<0.1,"Finger contact before closure");
        for(int i=0;i<mjNWARNING;++i) require(!d->warning[i].number,"MuJoCo physics warning");
      }
      mj_forward(m,d);
    };
    step(2.0,false);
    ContactState initial=sample(m,d,ids);
    Eigen::Vector3d handle=Eigen::Map<Eigen::Vector3d>(d->site_xpos+3*site);
    Eigen::Matrix3d rotation; rotation<<0,1,0, 1,0,0, 0,0,-1;
    Q seed; for(int i=0;i<7;++i) seed[i]=d->qpos[i];
    std::vector<Phase> phases;
    for(int index=0;index<4;++index) {
      Eigen::Vector3d target=handle+Eigen::Vector3d(0,0,0.1029+0.06-0.02*index);
      mj_copyData(kin,m,d); mju_zero(kin->qvel,m->nv);
      Q q=seed;
      bool solved=false;
      for(int iteration=0;iteration<400;++iteration) {
        for(int i=0;i<7;++i) kin->qpos[i]=q[i];
        mj_forward(m,kin);
        Eigen::Vector3d p=Eigen::Map<Eigen::Vector3d>(kin->xpos+3*ids.hand);
        Eigen::Matrix3d current=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(kin->xmat+9*ids.hand);
        Eigen::Vector3d angle=Eigen::Vector3d::Zero();
        for(int i=0;i<3;++i) angle+=0.5*current.col(i).cross(rotation.col(i));
        Eigen::Matrix<double,6,1> e; e<<target-p,angle;
        if((target-p).norm()<1e-6 && (current-rotation).norm()<1e-5) { solved=true;break; }
        std::vector<mjtNum> jp(3*m->nv),jr(3*m->nv);
        mj_jacBody(m,kin,jp.data(),jr.data(),ids.hand);
        Eigen::Matrix<double,6,7> jac;
        for(int r=0;r<3;++r) for(int c=0;c<7;++c) { jac(r,c)=jp[r*m->nv+c];jac(r+3,c)=jr[r*m->nv+c]; }
        Q delta=jac.transpose()*(jac*jac.transpose()+0.0001*Eigen::Matrix<double,6,6>::Identity()).ldlt().solve(e);
        if(delta.norm()>0.1) delta*=0.1/delta.norm();
        q+=delta;
        for(int i=0;i<7;++i) q[i]=std::clamp(q[i],m->actuator_ctrlrange[2*i]+0.02,m->actuator_ctrlrange[2*i+1]-0.02);
      }
      require(solved,"Inverse kinematics did not converge");
      seed=q;
      // Offset the position servo target to balance gravity at this static pose.
      Q command=q;
      for(int i=0;i<7;++i) {
        double kp=m->actuator_gainprm[i*mjNGAIN];
        require(kp>0 && std::fabs(m->actuator_biasprm[i*mjNBIAS+1]+kp)<1e-6,"Expected Panda position servos");
        command[i]+=kin->qfrc_bias[i]/kp;
        require(command[i]>=m->actuator_ctrlrange[2*i] && command[i]<=m->actuator_ctrlrange[2*i+1],"Command exceeds actuator range");
      }
      Q start;for(int i=0;i<7;++i) start[i]=d->qpos[i];
      double begin=d->time;
      while(d->time-begin<3.0) {
        double u=std::clamp((d->time-begin)/3.0,0.0,1.0), blend=u*u*(3-2*u);
        for(int i=0;i<7;++i) d->ctrl[i]=start[i]+blend*(command[i]-start[i]);
        step(m->opt.timestep,false);
      }
      for(int i=0;i<7;++i) d->ctrl[i]=command[i];
      step(1.0,false);
      ContactState state=sample(m,d,ids);
      Eigen::Vector3d measured=Eigen::Map<Eigen::Vector3d>(state.hand.data());
      require((target-measured).norm()<0.003,"Hand position error exceeds 3 mm");
      require(std::hypot(state.tray[0]-initial.tray[0],state.tray[1]-initial.tray[1])<0.002,"Tray moved during approach");
      phases.push_back({index==0 ? "approach" : "descend_"+std::to_string(index),command,{target.x(),target.y(),target.z()},state});
      std::cout<<phases.back().name<<": position error="<<(target-measured).norm()<<" m\n";
    }
    for(double opening:{0.025,0.015}) { d->ctrl[7]=opening;step(1.0,false); }
    d->ctrl[7]=0; step(3.0,true);
    double min_left=1e9,min_right=1e9,max_z_change=0,max_xy_change=0;
    double end=d->time+1.0;
    while(d->time<end) {
      step(m->opt.timestep,true);auto s=sample(m,d,ids);
      min_left=std::min(min_left,s.left_N);min_right=std::min(min_right,s.right_N);
      max_z_change=std::max(max_z_change,std::fabs(s.tray[2]-initial.tray[2]));
      max_xy_change=std::max(max_xy_change,std::hypot(s.tray[0]-initial.tray[0],s.tray[1]-initial.tray[1]));
      require(s.left_N<15 && s.right_N<15,"Finger force exceeds test bound");
    }
    auto final=sample(m,d,ids);
    require(min_left>0.2 && min_right>0.2,"No sustained bilateral handle contact");
    require(max_z_change<0.003 && max_xy_change<0.003 && final.support_N>0.5,"Tray must remain on support");
    // Release before moving the arm away. This check stops at the contact pose.
    d->ctrl[7]=0.04;step(2.0,true);
    auto released=sample(m,d,ids);require(released.left_N+released.right_N<0.1,"Handle contact did not release");
    output<<std::setprecision(12)<<"{\"status\":\"PASS\",\"schema_version\":1,\"duration_s\":3.0,\"settle_s\":1.0,\"finger_force_limit_N\":20.0,\"initial_state\":"<<state_json(initial)<<",\"phases\":[";
    for(size_t i=0;i<phases.size();++i) {
      if(i) output<<',';
      auto& p=phases[i];
      output<<"{\"name\":\""<<p.name<<"\",\"positions\":[";
      for(int j=0;j<7;++j) { if(j) output<<',';output<<p.command[j]; }
      output<<"],\"hand_target_xyz\":";array_json(output,p.hand_target);
      output<<",\"offline_state\":"<<state_json(p.measured)<<'}';
    }
    output<<"],\"offline_contact_state\":"<<state_json(final)<<",\"minimum_left_force_N\":"<<min_left
          <<",\"minimum_right_force_N\":"<<min_right<<",\"released_state\":"<<state_json(released)<<"}\n";
    std::cout<<"PASS: both fingers contact the handle; tray remains on support; release succeeds\n"<<state_json(final)<<'\n';
  } catch(const std::exception& e) { output<<"{\"status\":\"FAIL\"}\n";std::cerr<<"FAIL: "<<e.what()<<'\n';mj_deleteData(kin);mj_deleteData(d);mj_deleteModel(m);return 1; }
  mj_deleteData(kin);mj_deleteData(d);mj_deleteModel(m);return 0;
}
