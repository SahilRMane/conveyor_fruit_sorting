#include "sorting_simulation/static_motion.hpp"
#include <fstream>
#include <iostream>

using namespace sorting_simulation;
constexpr double LIFT_HEIGHT_M=0.05;
constexpr double LIFT_STEP_M=0.01;
constexpr double SEGMENT_SECONDS=3.0;
constexpr double SETTLE_SECONDS=1.0;
constexpr double HOLD_SECONDS=3.0;

struct LiftPhase {
  std::string name,stage;
  ArmVector command;
  std::array<double,3> target;
  ContactState measured;
};

int main(int argc,char** argv) {
  if(argc!=3) {std::cerr<<"Usage: plan_static_lift SCENE_XML OUTPUT_JSON\n";return 2;}
  std::ofstream output(argv[2]);
  if(!output) {std::cerr<<"Cannot write plan file\n";return 2;}
  char error[2048]={};mjModel* model=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!model) {output<<"{\"status\":\"FAIL\"}\n";std::cerr<<error<<'\n';return 1;}
  try {
    StaticMotion motion(model);
    check(model->opt.cone==mjCONE_ELLIPTIC && std::fabs(model->opt.impratio-100)<1e-9,
          "Use scene_lift.xml with elliptic friction and impratio=100");
    std::string stage="approach";
    bool reference_set=false,grasp_set=false;
    ContactState initial,grasp;
    double maximum_slip=0,maximum_placement_force=0;
    std::vector<ContactState> samples;
    motion.guard=[&](const ContactState& s) {
      check(s.forbidden==0,"Unexpected robot contact");
      check(s.left_N<15 && s.right_N<15,"Finger contact force exceeds 15 N");
      for(double value:s.tray) check(std::isfinite(value),"Non-finite tray position");
      if(reference_set) {
        if(stage=="lower" || stage=="release") {
          maximum_placement_force=std::max(maximum_placement_force,s.support_N);
          check(s.support_N<2*initial.support_N,"Placement support force exceeds twice tray weight");
        }
        check(std::hypot(s.tray[0]-initial.tray[0],s.tray[1]-initial.tray[1])<0.003,"Tray horizontal drift exceeds 3 mm");
        check(s.tray[2]>=initial.tray[2]-0.003 && s.tray[2]<=initial.tray[2]+LIFT_HEIGHT_M+0.01,"Tray height outside test bounds");
        double dot=0;for(int i=0;i<4;++i) dot+=s.tray_quat[i]*initial.tray_quat[i];
        check(std::fabs(dot)>std::cos(3*M_PI/360),"Tray tilt exceeds 3 degrees");
        if(stage=="approach" || stage=="grip" || stage=="release") {
          check(distance3(s.tray,initial.tray)<0.003 && s.support_N>0.5,"Tray must remain supported");
        }
      }
      if(grasp_set && (stage=="lift" || stage=="hold" || stage=="lower")) {
        check(s.left_N>0.2 && s.right_N>0.2,"Finger-to-handle contact was lost: "+state_json(s));
        std::array<double,3> relative{},baseline{};
        for(int i=0;i<3;++i) {relative[i]=s.hand[i]-s.tray[i];baseline[i]=grasp.hand[i]-grasp.tray[i];}
        maximum_slip=std::max(maximum_slip,distance3(relative,baseline));
        check(distance3(relative,baseline)<0.003,"Tray slip relative to hand exceeds 3 mm: "+state_json(s));
        if(s.tray[2]-initial.tray[2]>0.005) check(s.support_N<0.1,"Lifted tray still has support force");
      }
      if(stage=="hold") samples.push_back(s);
    };
    motion.step(2.0);initial=motion.state();reference_set=true;
    int site=mj_name2id(model,mjOBJ_SITE,"handle_grasp_site");check(site>=0,"Missing handle site");
    std::array<double,3> contact_target{};
    for(int i=0;i<3;++i) contact_target[i]=motion.data->site_xpos[3*site+i];
    contact_target[2]+=0.1029;
    std::vector<LiftPhase> phases;
    auto move_to=[&](const std::string& name,const std::string& mode,std::array<double,3> target) {
      stage=mode;
      ArmVector command=motion.solve(target);
      motion.move(command,SEGMENT_SECONDS,SETTLE_SECONDS);
      ContactState measured=motion.state();
      check(distance3(measured.hand,target)<0.003,"Hand position error exceeds 3 mm");
      if(mode=="approach") check(measured.left_N+measured.right_N<0.1,"Handle contact before closure");
      phases.push_back({name,mode,command,target,measured});
      std::cout<<name<<": tray rise="<<measured.tray[2]-initial.tray[2]<<" m, support="<<measured.support_N<<" N\n";
    };
    for(int i=0;i<4;++i) {
      auto target=contact_target;target[2]+=0.06-0.02*i;
      move_to(i==0?"approach":"descend_"+std::to_string(i),"approach",target);
    }
    stage="grip";
    for(double opening:{0.025,0.015}) {motion.data->ctrl[7]=opening;motion.step(1.0);}
    motion.data->ctrl[7]=0;motion.step(3.0);grasp=motion.state();
    check(grasp.left_N>0.2 && grasp.right_N>0.2,"Both fingers must contact the handle before lift");
    grasp_set=true;
    int steps=static_cast<int>(std::lround(LIFT_HEIGHT_M/LIFT_STEP_M));
    check(steps>0 && std::fabs(steps*LIFT_STEP_M-LIFT_HEIGHT_M)<1e-9,"Lift height must be a positive multiple of lift step");
    for(int i=1;i<=steps;++i) {
      auto target=contact_target;target[2]+=i*LIFT_STEP_M;
      move_to("lift_"+std::to_string(i),"lift",target);
    }
    stage="hold";samples.clear();motion.step(HOLD_SECONDS);
    double minimum_rise=1e9,maximum_rise=-1e9,maximum_support=0,minimum_left=1e9,minimum_right=1e9;
    for(const auto& s:samples) {
      minimum_rise=std::min(minimum_rise,s.tray[2]-initial.tray[2]);
      maximum_rise=std::max(maximum_rise,s.tray[2]-initial.tray[2]);
      maximum_support=std::max(maximum_support,s.support_N);
      minimum_left=std::min(minimum_left,s.left_N);minimum_right=std::min(minimum_right,s.right_N);
    }
    check(!samples.empty() && minimum_rise>LIFT_HEIGHT_M-0.005 && maximum_support<0.1,"Tray did not remain lifted");
    check(maximum_rise-minimum_rise<0.002,"Tray height drift exceeds 2 mm during hold");
    ContactState held=motion.state();
    for(int i=steps-1;i>=0;--i) {
      auto target=contact_target;target[2]+=i*LIFT_STEP_M;
      move_to("lower_"+std::to_string(steps-i),"lower",target);
    }
    ContactState placed=motion.state();
    check(std::fabs(placed.tray[2]-initial.tray[2])<0.002 && placed.support_N>0.8*initial.support_N,
          "Tray must return to the support before release");
    stage="release";motion.data->ctrl[7]=0.04;motion.step(2.0);
    ContactState released=motion.state();
    check(released.left_N+released.right_N<0.1,"Finger contact did not release");
    check(std::fabs(released.support_N-initial.support_N)<0.5,"Support force must balance weight after release");
    output<<std::setprecision(12)<<"{\"status\":\"PASS\",\"schema_version\":3,\"lift_height_m\":"<<LIFT_HEIGHT_M
          <<",\"duration_s\":"<<SEGMENT_SECONDS<<",\"settle_s\":"<<SETTLE_SECONDS<<",\"hold_s\":"<<HOLD_SECONDS
          <<",\"interpolate_from_desired_state\":true,\"initial_state\":"<<state_json(initial)<<",\"grasp_state\":"<<state_json(grasp)<<",\"phases\":[";
    for(size_t i=0;i<phases.size();++i) {
      if(i) output<<',';
      const auto& p=phases[i];output<<"{\"name\":\""<<p.name<<"\",\"stage\":\""<<p.stage<<"\",\"positions\":[";
      for(int j=0;j<7;++j) {if(j) output<<',';output<<p.command[j];}
      output<<"],\"hand_target_xyz\":";array_json(output,p.target);
      output<<",\"offline_state\":"<<state_json(p.measured)<<'}';
    }
    output<<"],\"held_state\":"<<state_json(held)<<",\"placed_state\":"<<state_json(placed)
          <<",\"released_state\":"<<state_json(released)<<",\"hold_metrics\":{\"minimum_rise_m\":"<<minimum_rise
          <<",\"maximum_rise_m\":"<<maximum_rise<<",\"maximum_support_force_N\":"<<maximum_support
          <<",\"minimum_left_force_N\":"<<minimum_left<<",\"minimum_right_force_N\":"<<minimum_right
          <<"},\"maximum_relative_slip_m\":"<<maximum_slip<<",\"maximum_placement_support_force_N\":"<<maximum_placement_force<<"}\n";
    std::cout<<"PASS: grip, lift, hold, lower and release\n"<<"Minimum held rise: "<<minimum_rise<<" m\n";
  } catch(const std::exception& e) {output<<"{\"status\":\"FAIL\"}\n";std::cerr<<"FAIL: "<<e.what()<<'\n';mj_deleteModel(model);return 1;}
  mj_deleteModel(model);return 0;
}
