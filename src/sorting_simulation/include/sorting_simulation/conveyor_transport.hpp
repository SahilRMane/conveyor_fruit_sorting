#pragma once
#include "sorting_simulation/contact_state.hpp"
#include <cmath>

namespace sorting_simulation {
struct ConveyorParameters {
  double speed_limit=0.10,force_limit=8.0,kp=100.0,ki=400.0;
  double x_min=0.42,x_max=0.88,y_limit=0.16,surface_z=0.30,rest_z=0.306;
};
struct ConveyorState {
  double speed=0,vx=0,force=0,torque_y=0,support=0;
  bool supported=false,in_region=false,active=false,fresh=false;
};
class ConveyorTransport {
  const mjModel* model_;
  SceneIds ids_;
  ConveyorParameters p_;
  int velocity_address_,position_address_;
  double integral_=0;
  double unsupported_seconds_=0;
public:
  explicit ConveyorTransport(const mjModel* m,ConveyorParameters p={}):model_(m),ids_(m),p_(p) {
    int joint=mj_name2id(m,mjOBJ_JOINT,"tray_free_joint");
    if(joint<0 || m->jnt_type[joint]!=mjJNT_FREE) throw std::runtime_error("Tray needs its free joint");
    velocity_address_=m->jnt_dofadr[joint];position_address_=m->jnt_qposadr[joint];
  }
  void reset() {integral_=0;unsupported_seconds_=0;}
  ConveyorState apply(mjData* d,double speed,bool fresh) {
    ConveyorState s;s.fresh=fresh;
    s.speed=(fresh && std::isfinite(speed) && speed>=0 && speed<=p_.speed_limit)?speed:0;
    s.vx=d->qvel[velocity_address_];
    double x=d->qpos[position_address_],y=d->qpos[position_address_+1],z=d->qpos[position_address_+2];
    s.in_region=x>=p_.x_min && x<=p_.x_max && std::fabs(y)<=p_.y_limit;
    s.support=sample(model_,d,ids_).support_N;
    s.supported=s.support>0.5 && std::fabs(z-p_.rest_z)<0.002;
    s.active=s.speed>0 && s.in_region && s.supported;
    if(s.active) {
      unsupported_seconds_=0;
      double error=s.speed-s.vx;
      double candidate=std::clamp(integral_+p_.ki*error*model_->opt.timestep,-p_.force_limit,p_.force_limit);
      double raw=p_.kp*error+candidate;
      s.force=std::clamp(raw,-p_.force_limit,p_.force_limit);
      if(std::fabs(raw-s.force)<1e-12 || error*raw<=0) integral_=candidate;
      // Apply traction at surface height, not at the tray's centre of mass.
      s.torque_y=(p_.surface_z-d->xipos[3*ids_.tray+2])*s.force;
    } else {
      unsupported_seconds_+=model_->opt.timestep;
      if(!fresh || s.speed==0 || !s.in_region || unsupported_seconds_>0.05) integral_=0;
    }
    // This plugin owns these two applied-force channels in the conveyor scene.
    d->xfrc_applied[6*ids_.tray]=s.force;
    d->xfrc_applied[6*ids_.tray+4]=s.torque_y;
    return s;
  }
};
}
