#pragma once
#include "sorting_simulation/static_motion.hpp"

namespace sorting_simulation {
constexpr double TRANSFER_LIFT_M=0.10;
constexpr double VERTICAL_STEP_M=0.02;
constexpr double TRANSFER_STEP_M=0.06;
constexpr double BIN_WALL_THICKNESS_M=0.008;
constexpr double TRAY_HALF_X_M=0.12, TRAY_HALF_Y_M=0.09, TRAY_HALF_Z_M=0.006;

inline void validate_bin_fixture(const mjModel* m,const SceneIds& ids) {
  int base=mj_name2id(m,mjOBJ_GEOM,"tray_base");
  check(base>=0 && m->geom_type[base]==mjGEOM_BOX,"Expected a box tray base");
  const double expected[3]={TRAY_HALF_X_M,TRAY_HALF_Y_M,TRAY_HALF_Z_M};
  for(int i=0;i<3;++i) check(std::fabs(m->geom_size[3*base+i]-expected[i])<1e-9,
                            "Tray dimensions changed; update the footprint checks");
  for(int i=0;i<m->neq;++i) check(m->eq_type[i]!=mjEQ_WELD,"No weld attachments are allowed in physical placement");
  for(int bin=0;bin<2;++bin) {
    check(ids.bins[bin]>=0,"Both bins must exist");
    int body=ids.bins[bin],floor=ids.bin_floors[bin];
    check(m->body_jntnum[body]==0 && m->body_parentid[body]==0,"Bins must be fixed to world");
    check(m->geom_type[floor]==mjGEOM_BOX,"Bin floor must be a box");
    check(std::fabs(m->body_quat[4*body]-1)<1e-9 && std::fabs(m->geom_quat[4*floor]-1)<1e-9,
          "Footprint checks require unrotated bins");
    std::string prefix=bin==0?"bin_a":"bin_b";
    for(int axis=0;axis<2;++axis) for(int sign:{-1,1}) {
      std::string name=prefix+"_wall_"+(axis==0?"x":"y")+(sign<0?"_minus":"_plus");
      int wall=mj_name2id(m,mjOBJ_GEOM,name.c_str());
      check(wall>=0 && m->geom_bodyid[wall]==body && m->geom_type[wall]==mjGEOM_BOX,"Missing box bin wall");
      double half=m->geom_size[3*wall+axis];
      check(std::fabs(2*half-BIN_WALL_THICKNESS_M)<1e-9 &&
            std::fabs(m->geom_pos[3*wall+axis]-m->geom_pos[3*floor+axis]-sign*(m->geom_size[3*floor+axis]-half))<1e-9,
            "Bin wall thickness or position differs from the footprint check");
      check(std::fabs(m->geom_quat[4*wall]-1)<1e-9,"Bin walls must be unrotated");
    }
  }
}

inline bool footprint_inside(const ContactState& s,int bin) {
  mjtNum rotation[9];mju_quat2Mat(rotation,s.tray_quat.data());
  for(double x:{-TRAY_HALF_X_M,TRAY_HALF_X_M})
    for(double y:{-TRAY_HALF_Y_M,TRAY_HALF_Y_M})
      for(double z:{-TRAY_HALF_Z_M,TRAY_HALF_Z_M}) {
        double world_x=s.tray[0]+rotation[0]*x+rotation[1]*y+rotation[2]*z;
        double world_y=s.tray[1]+rotation[3]*x+rotation[4]*y+rotation[5]*z;
        if(std::fabs(world_x-s.bin_xyz[bin][0])>s.bin_half_size[bin][0]-BIN_WALL_THICKNESS_M-0.003 ||
           std::fabs(world_y-s.bin_xyz[bin][1])>s.bin_half_size[bin][1]-BIN_WALL_THICKNESS_M-0.003) return false;
      }
  return true;
}

struct PlacementGuard {
  int bin;
  std::string stage="startup";
  bool reference_set=false,grasp_set=false;
  ContactState reference,grasp;
  double maximum_slip=0,maximum_bin_force=0,minimum_transfer_rise=1e9;
  explicit PlacementGuard(int destination):bin(destination) {}
  std::array<double,3> target_tray(const ContactState& s) const {
    return {s.bin_xyz[bin][0],s.bin_xyz[bin][1],s.bin_top_z[bin]+TRAY_HALF_Z_M};
  }
  void operator()(const ContactState& s) {
    check(s.bins_present[0] && s.bins_present[1],"Use scene_bins.xml with both bins");
    check(s.forbidden==0,"Unexpected robot contact");
    check(s.tray_floor_N<0.1,"Tray touched the ground plane");
    check(s.bin_wall_N[0]+s.bin_wall_N[1]<0.1,"Tray touched a bin wall");
    check(s.left_N<15 && s.right_N<15,"Finger force exceeded 15 N");
    check(std::fabs(s.external_force_x_N)<1e-9 && std::fabs(s.external_torque_y_Nm)<1e-9,
          "Stationary placement must not use conveyor force");
    for(double v:s.tray) check(std::isfinite(v),"Invalid tray position");
    if(!reference_set) return;
    double dot=0;for(int i=0;i<4;++i) dot+=s.tray_quat[i]*reference.tray_quat[i];
    check(std::fabs(dot)>std::cos(3*M_PI/360),"Tray tilt exceeded 3 degrees");
    check(s.tray[2]>=std::min(reference.tray[2],target_tray(s)[2])-0.003 &&
          s.tray[2]<=reference.tray[2]+TRANSFER_LIFT_M+0.01,"Tray height outside transfer bounds");
    check(s.bin_floor_N[1-bin]<0.1,"Tray touched the wrong bin floor");
    if(stage=="approach" || stage=="grip") {
      check(distance3(s.tray,reference.tray)<0.003 && s.support_N>0.5,"Tray left its initial support");
      check(s.bin_floor_N[bin]<0.1,"Unexpected bin contact before transfer");
    }
    if(grasp_set && (stage=="lift" || stage=="transfer" || stage=="lower")) {
      check(s.left_N>0.2 && s.right_N>0.2,"Physical handle contact was lost");
      std::array<double,3> relative{},baseline{};
      for(int i=0;i<3;++i) {relative[i]=s.hand[i]-s.tray[i];baseline[i]=grasp.hand[i]-grasp.tray[i];}
      maximum_slip=std::max(maximum_slip,distance3(relative,baseline));
      check(maximum_slip<0.003,"Tray slip exceeded 3 mm");
    }
    if(stage=="lift" && s.tray[2]-reference.tray[2]>0.005)
      check(s.support_N<0.1,"Lifted tray still has source support force");
    if(stage=="transfer") {
      double rise=s.tray[2]-reference.tray[2];
      minimum_transfer_rise=std::min(minimum_transfer_rise,rise);
      check(rise>TRANSFER_LIFT_M-0.005,"Insufficient clearance during transfer");
      check(s.support_N+s.bin_floor_N[bin]<0.1,"Tray must be airborne during transfer");
    }
    if(stage=="lower" || stage=="release" || stage=="retreat" || stage=="return" || stage=="home" || stage=="verify") {
      maximum_bin_force=std::max(maximum_bin_force,s.bin_floor_N[bin]);
      check(s.support_N<0.1,"Tray remained on its initial support");
      check(footprint_inside(s,bin),"Tray footprint is outside the selected bin");
      check(s.bin_floor_N[bin]<2*reference.support_N,"Bin-floor force exceeded twice tray weight");
    }
    if(stage=="release" || stage=="retreat" || stage=="return" || stage=="home" || stage=="verify") {
      check(distance3(s.tray,target_tray(s))<0.003,"Placed tray moved from the bin centre");
      check(s.bin_floor_N[bin]>0.5,"Placed tray lost its bin-floor support");
    }
    if(stage=="retreat" || stage=="return" || stage=="home" || stage=="verify")
      check(s.left_N+s.right_N<0.1,"Finger contact persisted after release");
  }
};
}
