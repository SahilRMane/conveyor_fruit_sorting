#pragma once
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace sorting_simulation {
struct SceneIds {
  int tray, support, root, left, right, hand, bar, floor, finger1, finger2, gripper;
  std::array<int,2> bins{{-1,-1}}, bin_floors{{-1,-1}};
  int id(const mjModel* m, mjtObj type, const char* name) {
    int result = mj_name2id(m, type, name);
    if (result < 0) throw std::runtime_error(std::string("Missing scene name: ") + name);
    return result;
  }
  explicit SceneIds(const mjModel* m) {
    tray=id(m,mjOBJ_BODY,"tray"); support=id(m,mjOBJ_BODY,"stationary_support");
    root=id(m,mjOBJ_BODY,"link0"); left=id(m,mjOBJ_BODY,"left_finger");
    right=id(m,mjOBJ_BODY,"right_finger"); hand=id(m,mjOBJ_BODY,"hand");
    bar=id(m,mjOBJ_GEOM,"handle_bar");
    floor=id(m,mjOBJ_GEOM,"static_floor");
    finger1=id(m,mjOBJ_JOINT,"panda_finger_joint1");
    finger2=id(m,mjOBJ_JOINT,"panda_finger_joint2");
    gripper=id(m,mjOBJ_ACTUATOR,"panda_finger_joint1");
    for(int i=0;i<2;++i) {
      std::string name=i==0 ? "bin_a" : "bin_b";
      bins[i]=mj_name2id(m,mjOBJ_BODY,name.c_str());
      if(bins[i]>=0) bin_floors[i]=id(m,mjOBJ_GEOM,(name+"_floor").c_str());
    }
  }
  bool robot(const mjModel* m, int body) const {
    while (body > 0) { if (body==root) return true; body=m->body_parentid[body]; }
    return false;
  }
};
struct ContactState {
  double time=0, left_N=0, right_N=0, support_N=0, gripper_limit_N=0;
  double friction_impedance_ratio=0, tray_mass_kg=0;
  int friction_cone=0;
  int forbidden=0;
  std::array<double,3> tray{}, hand{};
  std::array<double,4> tray_quat{}, hand_quat{};
  std::array<double,2> fingers{};
  std::array<double,2> bin_floor_N{}, bin_wall_N{}, bin_top_z{};
  std::array<std::array<double,3>,2> bin_xyz{}, bin_half_size{};
  std::array<double,3> tray_linear_velocity{}, tray_angular_velocity{};
  std::array<double,3> hand_linear_velocity{};
  double tray_floor_N=0, external_force_x_N=0, external_torque_y_Nm=0;
  std::array<int,2> bins_present{};
};
inline ContactState sample(const mjModel* m, const mjData* d, const SceneIds& ids) {
  ContactState s; s.time=d->time;
  s.friction_cone=static_cast<int>(m->opt.cone);
  s.friction_impedance_ratio=m->opt.impratio;
  s.tray_mass_kg=m->body_mass[ids.tray];
  std::copy_n(d->xpos+3*ids.tray,3,s.tray.begin());
  std::copy_n(d->xquat+4*ids.tray,4,s.tray_quat.begin());
  std::copy_n(d->xquat+4*ids.hand,4,s.hand_quat.begin());
  s.gripper_limit_N=m->actuator_forcelimited[ids.gripper] ? m->actuator_forcerange[2*ids.gripper+1] : -1;
  std::copy_n(d->xpos+3*ids.hand,3,s.hand.begin());
  s.fingers={d->qpos[m->jnt_qposadr[ids.finger1]],d->qpos[m->jnt_qposadr[ids.finger2]]};
  mjtNum velocity[6];mj_objectVelocity(m,d,mjOBJ_BODY,ids.tray,velocity,0);
  std::copy_n(velocity,3,s.tray_angular_velocity.begin());
  std::copy_n(velocity+3,3,s.tray_linear_velocity.begin());
  mj_objectVelocity(m,d,mjOBJ_BODY,ids.hand,velocity,0);
  std::copy_n(velocity+3,3,s.hand_linear_velocity.begin());
  s.external_force_x_N=d->xfrc_applied[6*ids.tray];
  s.external_torque_y_Nm=d->xfrc_applied[6*ids.tray+4];
  for(int i=0;i<2;++i) if(ids.bins[i]>=0) {
    s.bins_present[i]=1;
    int geom=ids.bin_floors[i];
    std::copy_n(d->geom_xpos+3*geom,3,s.bin_xyz[i].begin());
    std::copy_n(m->geom_size+3*geom,3,s.bin_half_size[i].begin());
    s.bin_top_z[i]=d->geom_xpos[3*geom+2]+m->geom_size[3*geom+2];
  }
  for (int i=0;i<d->ncon;++i) {
    const auto& c=d->contact[i];
    if (c.geom[0]<0 || c.geom[1]<0) continue;
    int a=m->geom_bodyid[c.geom[0]], b=m->geom_bodyid[c.geom[1]];
    mjtNum force[6]; mj_contactForce(m,d,i,force);
    bool left=(c.geom[0]==ids.bar && b==ids.left)||(c.geom[1]==ids.bar && a==ids.left);
    bool right=(c.geom[0]==ids.bar && b==ids.right)||(c.geom[1]==ids.bar && a==ids.right);
    if (left) s.left_N+=std::max(0.0,force[0]);
    if (right) s.right_N+=std::max(0.0,force[0]);
    if ((a==ids.tray && b==ids.support)||(b==ids.tray && a==ids.support))
      s.support_N+=std::max(0.0,force[0]);
    bool env_a=a==ids.tray || a==ids.support, env_b=b==ids.tray || b==ids.support;
    for(int bin=0;bin<2;++bin) if(ids.bins[bin]>=0) {
      env_a=env_a || a==ids.bins[bin];env_b=env_b || b==ids.bins[bin];
      if((a==ids.tray && b==ids.bins[bin]) || (b==ids.tray && a==ids.bins[bin])) {
        bool on_floor=c.geom[0]==ids.bin_floors[bin] || c.geom[1]==ids.bin_floors[bin];
        (on_floor ? s.bin_floor_N[bin] : s.bin_wall_N[bin])+=std::max(0.0,force[0]);
      }
    }
    if((a==ids.tray && c.geom[1]==ids.floor) || (b==ids.tray && c.geom[0]==ids.floor))
      s.tray_floor_N+=std::max(0.0,force[0]);
    if (((env_a && ids.robot(m,b))||(env_b && ids.robot(m,a))) && !left && !right && force[0]>0.1)
      ++s.forbidden;
    if (ids.robot(m,a) && ids.robot(m,b) && force[0]>0.1) ++s.forbidden;
    if (((c.geom[0]==ids.floor && ids.robot(m,b) && b!=ids.root)||
         (c.geom[1]==ids.floor && ids.robot(m,a) && a!=ids.root)) && force[0]>0.1) ++s.forbidden;
  }
  return s;
}
template<size_t N,typename T> inline void array_json(std::ostream& out,const std::array<T,N>& a) {
  out << '['; for(size_t i=0;i<N;++i) { if(i) out<<','; out<<a[i]; } out<<']';
}
inline std::string state_json(const ContactState& s) {
  std::ostringstream out; out<<std::setprecision(12);
  out<<"{\"sim_time\":"<<s.time<<",\"left_force_N\":"<<s.left_N
     <<",\"right_force_N\":"<<s.right_N<<",\"support_force_N\":"<<s.support_N
     <<",\"gripper_force_limit_N\":"<<s.gripper_limit_N
     <<",\"friction_cone\":"<<s.friction_cone<<",\"friction_impedance_ratio\":"<<s.friction_impedance_ratio
     <<",\"tray_mass_kg\":"<<s.tray_mass_kg
     <<",\"forbidden_contacts\":"<<s.forbidden<<",\"tray_xyz\":";
  array_json(out,s.tray); out<<",\"tray_quat_wxyz\":";array_json(out,s.tray_quat);
  out<<",\"hand_xyz\":";array_json(out,s.hand);out<<",\"hand_quat_wxyz\":";array_json(out,s.hand_quat);
  out<<",\"finger_positions_m\":";array_json(out,s.fingers);
  out<<",\"contact_schema_version\":2,\"bins_present\":";array_json(out,s.bins_present);
  out<<",\"bin_names\":[\"bin_a\",\"bin_b\"],\"bin_floor_force_N\":";array_json(out,s.bin_floor_N);
  out<<",\"bin_wall_force_N\":";array_json(out,s.bin_wall_N);
  out<<",\"bin_floor_top_z\":";array_json(out,s.bin_top_z);
  out<<",\"bin_floor_xyz\":[";array_json(out,s.bin_xyz[0]);out<<',';array_json(out,s.bin_xyz[1]);out<<']';
  out<<",\"bin_floor_half_size\":[";array_json(out,s.bin_half_size[0]);out<<',';array_json(out,s.bin_half_size[1]);out<<']';
  out<<",\"tray_linear_velocity_m_s\":";array_json(out,s.tray_linear_velocity);
  out<<",\"hand_linear_velocity_m_s\":";array_json(out,s.hand_linear_velocity);
  out<<",\"tray_angular_velocity_rad_s\":";array_json(out,s.tray_angular_velocity);
  out<<",\"tray_floor_force_N\":"<<s.tray_floor_N
     <<",\"external_force_x_N\":"<<s.external_force_x_N
     <<",\"external_torque_y_Nm\":"<<s.external_torque_y_Nm;
  out<<'}'; return out.str();
}
}
