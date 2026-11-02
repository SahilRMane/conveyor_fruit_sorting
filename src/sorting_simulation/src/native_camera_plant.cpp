// Standalone test plant. ROS is never initialized. Truth is returned only by
// nc_truth, separate from encoder, camera and bounded actuator interfaces.
#include "sorting_simulation/native_camera_render.hpp"
#include "sorting_simulation/conveyor_transport.hpp"
using V3=Eigen::Vector3d;
struct Plant {
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model{nullptr,mj_deleteModel};
  std::unique_ptr<StaticMotion> sim;std::unique_ptr<ConveyorTransport> belt;
  Offscreen renderer;ArmVector command;double condition=0,max_speed=0;
  std::array<double,2> floor_peaks{{0,0}};
  double kp=4.,kr=3.,linear_limit=.049,angular_limit=.08,joint_limit=.5,warn=17.,hard=30.;
  std::string result;
  explicit Plant(const char* scene){char error[2048]={};model.reset(mj_loadXML(scene,nullptr,error,sizeof(error)));check(bool(model),error);
    sim=std::make_unique<StaticMotion>(model.get());belt=std::make_unique<ConveyorTransport>(model.get());
    sim->guard=[this](const ContactState& state){
      // Grader-only record: retain impacts between Python 10 ms samples.
      for(int i=0;i<2;++i)floor_peaks[i]=std::max(floor_peaks[i],state.bin_floor_N[i]);
      check(state.forbidden==0,"Native evaluator: forbidden physical contact");
      check(std::max(state.left_N,state.right_N)<15.,"Native evaluator: finger force bound");
      for(int i=0;i<7;++i){max_speed=std::max(max_speed,std::abs(sim->data->qvel[i]));check(std::abs(sim->data->qvel[i])<joint_limit,"Native evaluator: measured joint speed bound");}
    };
    renderer.init(model.get());sim->step(2.);command=Eigen::Map<ArmVector>(sim->data->ctrl);
  }
  void step(double seconds,double speed){auto* d=sim->data;double end=d->time+seconds;
    while(d->time<end-1e-10){belt->apply(d,speed,true);sim->step(model->opt.timestep);
      for(int i=0;i<7;++i)max_speed=std::max(max_speed,std::abs(d->qvel[i]));}}
  void control(const double* target,const double* ff,double speed){auto* m=model.get();auto* d=sim->data;
    mj_forward(m,d);V3 p=Eigen::Map<V3>(d->xpos+3*sim->ids.hand);
    Eigen::Matrix3d current=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(d->xmat+9*sim->ids.hand);
    Eigen::Matrix3d desired;desired<<0,1,0,1,0,0,0,0,-1;
    V3 angular=V3::Zero();for(int i=0;i<3;++i)angular+=.5*current.col(i).cross(desired.col(i));angular*=kr;
    V3 linear=Eigen::Map<const V3>(ff)+kp*(Eigen::Map<const V3>(target)-p);
    if(linear.norm()>linear_limit){linear*=linear_limit/linear.norm();}
    if(angular.norm()>angular_limit){angular*=angular_limit/angular.norm();}
    mj_copyData(sim->kinematics,m,d);for(int i=0;i<7;++i)sim->kinematics->qpos[i]=command[i];mj_forward(m,sim->kinematics);
    std::vector<mjtNum> jp(3*m->nv),jr(3*m->nv);mj_jacBody(m,sim->kinematics,jp.data(),jr.data(),sim->ids.hand);
    Eigen::Matrix<double,6,7> j;for(int r=0;r<3;++r)for(int c=0;c<7;++c){j(r,c)=jp[r*m->nv+c];j(r+3,c)=jr[r*m->nv+c];}
    auto singular=j.jacobiSvd().singularValues();condition=singular[0]/singular[5];check(condition<hard,"Native Servo singularity bound");
    double scale=condition>warn?(hard-condition)/(hard-warn):1.;Eigen::Matrix<double,6,1> twist;twist<<scale*linear,scale*angular;
    ArmVector velocity=j.transpose()*(j*j.transpose()+.0001*Eigen::Matrix<double,6,6>::Identity()).ldlt().solve(twist);
    for(int i=0;i<7;++i){check(std::abs(velocity[i])<joint_limit,"Native command speed bound");command[i]+=.01*velocity[i];d->ctrl[i]=command[i];}
    step(.01,speed);
  }
};
static thread_local std::string failure;
#define SAFE_BEGIN try {
#define SAFE_END } catch(const std::exception& e){failure=e.what();return -1;} return 0;
extern "C" {
const char* nc_error(){return failure.c_str();}
void* nc_create(const char* scene){try{return new Plant(scene);}catch(const std::exception& e){failure=e.what();return nullptr;}}
void nc_destroy(void* p){delete static_cast<Plant*>(p);}
int nc_settings(void* ptr,const double* v){auto& p=*static_cast<Plant*>(ptr);p.kp=v[0];p.kr=v[1];p.linear_limit=v[2];p.angular_limit=v[3];p.joint_limit=v[4];p.warn=v[5];p.hard=v[6];return 0;}
int nc_step(void* ptr,double seconds,double speed){SAFE_BEGIN static_cast<Plant*>(ptr)->step(seconds,speed);SAFE_END}
int nc_pregrasp(void* ptr,const double* target){SAFE_BEGIN auto& p=*static_cast<Plant*>(ptr);std::array<double,3> goal;std::copy_n(target,3,goal.begin());
  p.sim->move(p.sim->solve(goal),6.,.5);p.command=Eigen::Map<ArmVector>(p.sim->data->ctrl);SAFE_END}
int nc_control(void* ptr,const double* target,const double* ff,double speed){SAFE_BEGIN static_cast<Plant*>(ptr)->control(target,ff,speed);SAFE_END}
int nc_close(void* ptr){static_cast<Plant*>(ptr)->sim->data->ctrl[7]=0.;return 0;}
int nc_gripper(void* ptr,double target){static_cast<Plant*>(ptr)->sim->data->ctrl[7]=target;return 0;}
int nc_open(void* ptr){static_cast<Plant*>(ptr)->sim->data->ctrl[7]=.04;return 0;}
int nc_resume(void* ptr){auto& p=*static_cast<Plant*>(ptr);p.command=Eigen::Map<ArmVector>(p.sim->data->ctrl);return 0;}
int nc_hold_reference(void*){return 0;}
int nc_hold(void* ptr){auto& p=*static_cast<Plant*>(ptr);for(int i=0;i<7;++i)p.sim->data->ctrl[i]=p.sim->data->qpos[i];return 0;}
int nc_feedback(void* ptr,double* values){SAFE_BEGIN auto& p=*static_cast<Plant*>(ptr);auto* d=p.sim->data;
  values[0]=d->time;std::copy_n(d->xpos+3*p.sim->ids.hand,3,values+1);
  std::copy_n(d->xquat+4*p.sim->ids.hand,4,values+4);
  std::copy_n(d->qpos,9,values+8);std::copy_n(d->qvel,9,values+17);values[26]=p.condition;values[27]=p.max_speed;SAFE_END}
const char* nc_truth(void* ptr){auto& p=*static_cast<Plant*>(ptr);p.result=state_json(p.sim->state());return p.result.c_str();}
int nc_floor_peaks(void* ptr,double* values){auto& p=*static_cast<Plant*>(ptr);
  std::copy_n(p.floor_peaks.begin(),2,values);p.floor_peaks={0,0};return 0;}
int nc_wrist_wrench(void* ptr,double* values){SAFE_BEGIN
  auto& p=*static_cast<Plant*>(ptr);auto* m=p.model.get();auto* d=p.sim->data;
  int force=mj_name2id(m,mjOBJ_SENSOR,"panda_hand_wrench_force");
  int torque=mj_name2id(m,mjOBJ_SENSOR,"panda_hand_wrench_torque");
  int site=mj_name2id(m,mjOBJ_SITE,"panda_hand_wrench_site");
  check(force>=0 && torque>=0 && site>=0,"Missing physical wrist sensor");
  values[0]=d->time;
  mju_mulMatVec(values+1,d->site_xmat+9*site,d->sensordata+m->sensor_adr[force],3,3);
  mju_mulMatVec(values+4,d->site_xmat+9*site,d->sensordata+m->sensor_adr[torque],3,3);
  SAFE_END}
int nc_shape(void* ptr,const char* name,int* shape){SAFE_BEGIN auto& p=*static_cast<Plant*>(ptr);int id=mj_name2id(p.model.get(),mjOBJ_CAMERA,name);check(id>=0,"Missing camera");
  shape[0]=p.model->cam_resolution[2*id];shape[1]=p.model->cam_resolution[2*id+1];SAFE_END}
int nc_render(void* ptr,const char* name,unsigned char* rgb,double* metadata){SAFE_BEGIN auto& p=*static_cast<Plant*>(ptr);auto* m=p.model.get();auto* d=p.sim->data;
  int id=mj_name2id(m,mjOBJ_CAMERA,name);check(id>=0,"Missing native camera");int w=m->cam_resolution[2*id],h=m->cam_resolution[2*id+1];
  mjrRect rect{0,0,w,h};p.renderer.camera.type=mjCAMERA_FIXED;p.renderer.camera.fixedcamid=id;
  mjv_updateScene(m,d,&p.renderer.options,nullptr,&p.renderer.camera,mjCAT_ALL,&p.renderer.scene);mjr_render(rect,&p.renderer.scene,&p.renderer.render);
  std::vector<unsigned char> bottom(w*h*3);mjr_readPixels(bottom.data(),nullptr,rect,&p.renderer.render);
  for(int row=0;row<h;++row)std::copy_n(bottom.data()+(h-row-1)*w*3,w*3,rgb+row*w*3);
  metadata[0]=d->time;metadata[1]=w;metadata[2]=h;metadata[3]=m->cam_fovy[id];std::copy_n(d->cam_xpos+3*id,3,metadata+4);
  for(int r=0;r<3;++r)for(int c=0;c<3;++c)metadata[7+r*3+c]=d->cam_xmat[9*id+r*3+c]*(c==0?1:-1);SAFE_END}
}
