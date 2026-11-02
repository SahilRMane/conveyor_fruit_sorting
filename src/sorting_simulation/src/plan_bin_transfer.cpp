#include "sorting_simulation/placement_checks.hpp"
#include <fstream>
#include <iostream>

using namespace sorting_simulation;
constexpr double SEGMENT_SECONDS=3.0,SETTLE_SECONDS=1.0;
struct Phase {
  std::string name,stage,operation;
  ArmVector command=ArmVector::Zero();
  std::array<double,3> target{};
  double opening=0,seconds=0;
  bool contact=false;
  ContactState measured;
};
inline void emit_phase(std::ostream& out,const Phase& p) {
  out<<"{\"name\":\""<<p.name<<"\",\"stage\":\""<<p.stage<<"\",\"operation\":\""<<p.operation<<"\"";
  if(p.operation=="arm") {
    out<<",\"positions\":[";for(int j=0;j<7;++j) {if(j) out<<',';out<<p.command[j];}
    out<<"],\"hand_target_xyz\":";array_json(out,p.target);
  } else if(p.operation=="gripper")
    out<<",\"opening_m\":"<<p.opening<<",\"expect_contact\":"<<(p.contact ? "true" : "false");
  else out<<",\"seconds\":"<<p.seconds;
  out<<",\"offline_state\":"<<state_json(p.measured)<<'}';
}

void make_plan(mjModel* model,int bin,std::ostream& out) {
  StaticMotion motion(model);PlacementGuard guard(bin);
  validate_bin_fixture(model,motion.ids);
  check(model->opt.cone==mjCONE_ELLIPTIC && std::fabs(model->opt.impratio-100)<1e-9,"Use the bin scene friction settings");
  motion.guard=[&](const ContactState& s){guard(s);};
  motion.step(2);guard.reference=motion.state();guard.reference_set=true;
  const auto initial=guard.reference;
  ArmVector home;for(int i=0;i<7;++i) home[i]=motion.data->ctrl[i];
  int site=mj_name2id(model,mjOBJ_SITE,"handle_grasp_site");check(site>=0,"Missing handle site");
  std::array<double,3> grasp_target{};
  for(int i=0;i<3;++i) grasp_target[i]=motion.data->site_xpos[3*site+i];
  grasp_target[2]+=0.1029;
  const auto destination=guard.target_tray(initial);
  std::array<double,3> destination_hand=grasp_target;
  for(int i=0;i<3;++i) destination_hand[i]+=destination[i]-initial.tray[i];
  std::vector<Phase> phases;
  auto move=[&](const std::string& name,const std::string& stage,const std::array<double,3>& target) {
    guard.stage=stage;Phase phase;phase.name=name;phase.stage=stage;phase.operation="arm";
    phase.command=motion.solve(target);phase.target=target;
    motion.move(phase.command,SEGMENT_SECONDS,SETTLE_SECONDS);phase.measured=motion.state();
    check(distance3(phase.measured.hand,target)<0.003,"Hand position error exceeded 3 mm");
    if(stage=="approach") check(phase.measured.left_N+phase.measured.right_N<0.1,"Contact before closure");
    phases.push_back(phase);
    std::cout<<(bin==0?"bin_a":"bin_b")<<'/'<<name<<" tray="<<phase.measured.tray[0]<<','<<phase.measured.tray[1]<<','<<phase.measured.tray[2]<<'\n';
  };
  auto grip=[&](const std::string& name,double opening,bool contact) {
    guard.stage=contact || opening<0.04 ? "grip" : "release";
    Phase phase;phase.name=name;phase.stage=guard.stage;phase.operation="gripper";
    phase.opening=opening;phase.contact=contact;motion.data->ctrl[7]=opening;
    motion.step(contact ? 3.0 : 2.0);phase.measured=motion.state();phases.push_back(phase);
  };
  auto wait=[&](const std::string& name,const std::string& stage,double seconds) {
    guard.stage=stage;motion.step(seconds);Phase p;p.name=name;p.stage=stage;p.operation="wait";
    p.seconds=seconds;p.measured=motion.state();phases.push_back(p);
  };
  for(int i=0;i<4;++i) {auto target=grasp_target;target[2]+=0.06-0.02*i;move("approach_"+std::to_string(i+1),"approach",target);}
  grip("close_25mm",0.025,false);grip("close_15mm",0.015,false);grip("grip_handle",0.0,true);
  guard.grasp=motion.state();guard.grasp_set=true;
  check(guard.grasp.left_N>0.2 && guard.grasp.right_N>0.2,"Both fingers must contact the handle");
  const int vertical_steps=std::lround(TRANSFER_LIFT_M/VERTICAL_STEP_M);
  for(int i=1;i<=vertical_steps;++i) {auto p=grasp_target;p[2]+=i*VERTICAL_STEP_M;move("lift_"+std::to_string(i),"lift",p);}
  wait("lift_hold","transfer",2.0);
  int travel_steps=std::max(1,static_cast<int>(std::ceil(distance3(destination_hand,grasp_target)/TRANSFER_STEP_M)));
  for(int i=1;i<=travel_steps;++i) {
    auto p=grasp_target;for(int j=0;j<3;++j) p[j]+=(destination_hand[j]-grasp_target[j])*i/travel_steps;
    p[2]+=TRANSFER_LIFT_M;move("transfer_"+std::to_string(i),"transfer",p);
  }
  wait("bin_hold","transfer",2.0);
  for(int i=vertical_steps-1;i>=0;--i) {auto p=destination_hand;p[2]+=i*VERTICAL_STEP_M;move("lower_"+std::to_string(vertical_steps-i),"lower",p);}
  const auto placed=motion.state();
  check(placed.bin_floor_N[bin]>0.8*initial.support_N && distance3(placed.tray,destination)<0.003,
        "Bin floor must support the tray before release");
  grip("release",0.04,false);const auto released=motion.state();
  check(released.left_N+released.right_N<0.1,"Fingers did not release");
  for(int i=1;i<=vertical_steps;++i) {auto p=destination_hand;p[2]+=i*VERTICAL_STEP_M;move("retreat_"+std::to_string(i),"retreat",p);}
  for(int i=1;i<=travel_steps;++i) {
    auto p=destination_hand;for(int j=0;j<3;++j) p[j]+=(grasp_target[j]-destination_hand[j])*i/travel_steps;
    p[2]+=TRANSFER_LIFT_M;move("return_"+std::to_string(i),"return",p);
  }
  guard.stage="home";motion.move(home,SEGMENT_SECONDS,SETTLE_SECONDS);
  Phase home_phase;home_phase.name="return_home";home_phase.stage="home";home_phase.operation="arm";
  home_phase.command=home;home_phase.target=initial.hand;home_phase.measured=motion.state();phases.push_back(home_phase);
  guard.stage="verify";
  std::vector<ContactState> stable;
  motion.guard=[&](const ContactState& s) {
    guard(s);stable.push_back(s);
    check(std::fabs(s.bin_floor_N[bin]-initial.support_N)<0.5,"Released tray weight is not supported by bin floor");
    for(double v:s.tray_linear_velocity) check(std::fabs(v)<0.002,"Released tray did not settle");
    for(double v:s.tray_angular_velocity) check(std::fabs(v)<0.02,"Released tray rotates after release");
  };
  motion.step(2);check(stable.size()>=50,"Insufficient placement samples");
  const auto final=motion.state();
  double peak_linear=0,peak_angular=0;
  for(const auto& s:stable) {
    for(double v:s.tray_linear_velocity) peak_linear=std::max(peak_linear,std::fabs(v));
    for(double v:s.tray_angular_velocity) peak_angular=std::max(peak_angular,std::fabs(v));
  }
  int home_key=mj_name2id(model,mjOBJ_KEY,"home");
  mj_resetDataKeyframe(model,motion.data,home_key);mj_forward(model,motion.data);
  for(int i=0;i<model->nq;++i) check(std::fabs(motion.data->qpos[i]-model->key_qpos[home_key*model->nq+i])<1e-9,"Raw home reset failed");
  for(int i=0;i<model->nv;++i) check(std::fabs(motion.data->qvel[i])<1e-9,"Reset left velocity");
  for(int i=0;i<model->nu;++i) check(std::fabs(motion.data->ctrl[i]-model->key_ctrl[home_key*model->nu+i])<1e-9,"Reset left stale commands");
  for(int i=0;i<6*model->nbody;++i) check(std::fabs(motion.data->xfrc_applied[i])<1e-9,"Reset left external body force");
  for(int i=0;i<model->nv;++i) check(std::fabs(motion.data->qfrc_applied[i])<1e-9,"Reset left external joint force");
  motion.guard=nullptr;motion.step(2);
  check(distance3(motion.state().tray,initial.tray)<0.001,"Tray failed to return from the bin at reset");
  out<<"{\"bin_name\":\""<<(bin==0?"bin_a":"bin_b")<<"\",\"bin_index\":"<<bin
     <<",\"initial_state\":"<<state_json(initial)<<",\"grasp_state\":"<<state_json(guard.grasp)
     <<",\"target_tray_xyz\":";array_json(out,destination);
  out<<",\"phases\":[";for(size_t i=0;i<phases.size();++i){if(i) out<<',';emit_phase(out,phases[i]);}
  out<<"],\"placed_state\":"<<state_json(placed)<<",\"released_state\":"<<state_json(released)
     <<",\"final_state\":"<<state_json(final)<<",\"metrics\":{\"maximum_relative_slip_m\":"<<guard.maximum_slip
     <<",\"maximum_bin_floor_force_N\":"<<guard.maximum_bin_force<<",\"minimum_transfer_rise_m\":"<<guard.minimum_transfer_rise
     <<",\"maximum_settled_linear_velocity_m_s\":"<<peak_linear<<",\"maximum_settled_angular_velocity_rad_s\":"<<peak_angular
     <<"},\"native_reset_status\":\"PASS\"}";
}

int main(int argc,char** argv) {
  if(argc!=3){std::cerr<<"Usage: plan_bin_transfer SCENE_XML OUTPUT_JSON\n";return 2;}
  std::ofstream output(argv[2]);if(!output){std::cerr<<"Cannot write output\n";return 2;}
  char error[2048]={};mjModel* model=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!model){output<<"{\"status\":\"FAIL\"}\n";std::cerr<<error<<'\n';return 1;}
  try {
    std::ostringstream plans;plans<<std::setprecision(12);
    for(int bin=0;bin<2;++bin){if(bin) plans<<',';make_plan(model,bin,plans);}
    output<<std::setprecision(12)<<"{\"status\":\"PASS\",\"schema_version\":5,\"mode\":\"stationary_bin_transfer\","
          <<"\"interpolate_from_desired_state\":true,\"duration_s\":"<<SEGMENT_SECONDS<<",\"settle_s\":"<<SETTLE_SECONDS
          <<",\"lift_height_m\":"<<TRANSFER_LIFT_M<<",\"bin_wall_thickness_m\":"<<BIN_WALL_THICKNESS_M
          <<",\"tray_half_size_m\":[0.12,0.09,0.006],\"placement_verify_s\":2.0,\"bins\":["<<plans.str()<<"]}\n";
    std::cout<<"PASS: physical transfer, placement, release, retreat and native reset in both bins\n";
  } catch(const std::exception& e) {
    output<<"{\"status\":\"FAIL\"}\n";std::cerr<<"FAIL: "<<e.what()<<'\n';mj_deleteModel(model);return 1;
  }
  mj_deleteModel(model);return 0;
}
