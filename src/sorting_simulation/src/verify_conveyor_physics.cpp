#include "sorting_simulation/static_motion.hpp"
#include "sorting_simulation/conveyor_transport.hpp"
#include <fstream>
#include <iostream>

using namespace sorting_simulation;
int main(int argc,char** argv) {
  if(argc!=3) {std::cerr<<"Usage: verify_conveyor_physics SCENE_XML OUTPUT_JSON\n";return 2;}
  std::ofstream out(argv[2]);if(!out) return 2;
  char error[2048]={};mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m) {std::cerr<<error<<'\n';return 1;}
  try {
    StaticMotion robot(m);ConveyorTransport belt(m);
    double requested=0;bool fresh=true;
    double peak=0,min_v=1e9,max_v=-1e9;
    auto step=[&](double seconds,bool measure=false) {
      double end=robot.data->time+seconds;
      while(robot.data->time<end) {
        auto state=belt.apply(robot.data,requested,fresh);
        peak=std::max(peak,std::fabs(state.force));
        check(std::fabs(state.force)<=8.000001,"Transport force exceeded its bound");
        mj_step(m,robot.data);
        check(robot.state().forbidden==0,"Unexpected robot contact during transport");
        if(measure) {min_v=std::min(min_v,state.vx);max_v=std::max(max_v,state.vx);}
      }
      mj_forward(m,robot.data);
    };
    step(2);double start_x=robot.state().tray[0];
    ConveyorParameters outside;outside.x_min=2;outside.x_max=3;
    ConveyorTransport absent(m,outside);
    auto outside_state=absent.apply(robot.data,0.03,true);
    check(outside_state.supported && !outside_state.in_region && outside_state.force==0,
          "Out-of-region transport was not disabled for a grounded tray");
    auto invalid=belt.apply(robot.data,0.2,true);
    check(!invalid.active && invalid.force==0,"Invalid speed was not rejected");
    requested=0.03;step(2);step(2,true);
    double displacement=robot.state().tray[0]-start_x;
    std::cout<<"Steady vx range="<<min_v<<".."<<max_v<<", displacement="<<displacement<<", peak force="<<peak<<'\n';
    std::cout<<state_json(robot.state())<<'\n';
    check(displacement>0.08 && min_v>0.028 && max_v<0.032,"Belt speed tracking failed");
    requested=0;step(1);
    check(std::fabs(robot.data->qvel[m->jnt_dofadr[mj_name2id(m,mjOBJ_JOINT,"tray_free_joint")]])<0.002,
          "Stop command did not stop the tray");
    requested=0.03;fresh=false;step(0.1);
    check(robot.data->xfrc_applied[6*robot.ids.tray]==0,"Expired command retained force");
    requested=0;fresh=true;
    // Use the physical gripper to lift the transported tray for the release test.
    int site=mj_name2id(m,mjOBJ_SITE,"handle_grasp_site");
    std::array<double,3> target{};for(int i=0;i<3;++i) target[i]=robot.data->site_xpos[3*site+i];
    target[2]+=0.1029;
    auto move=[&](std::array<double,3> pose) {
      auto command=robot.solve(pose);ArmVector start;for(int i=0;i<7;++i) start[i]=robot.data->ctrl[i];
      double begin=robot.data->time;
      while(robot.data->time-begin<3) {
        double u=(robot.data->time-begin)/3,blend=u*u*(3-2*u);
        for(int i=0;i<7;++i) robot.data->ctrl[i]=start[i]+blend*(command[i]-start[i]);
        step(m->opt.timestep);
      }
      for(int i=0;i<7;++i) robot.data->ctrl[i]=command[i];
      step(1);
    };
    for(int i=0;i<4;++i) {auto p=target;p[2]+=0.06-0.02*i;move(p);}
    robot.data->ctrl[7]=0;step(3);
    check(robot.state().left_N>0.2 && robot.state().right_N>0.2,"Physical test grip failed");
    double grounded_z=robot.state().tray[2];requested=0.03;
    for(int i=1;i<=5;++i) {auto p=target;p[2]+=0.01*i;move(p);}
    step(2);
    auto held=robot.state();
    check(held.tray[2]-grounded_z>0.045 && held.support_N<0.1,"Tray did not lift clear");
    check(held.left_N>0.2 && held.right_N>0.2,"Physical gripper did not retain the lifted tray");
    check(robot.data->xfrc_applied[6*robot.ids.tray]==0 && robot.data->xfrc_applied[6*robot.ids.tray+4]==0,
          "Conveyor force persisted after physical lift");
    out<<std::setprecision(12)<<"{\"status\":\"PASS\",\"transport_model\":\"bounded_force_approximation\",\"speed_m_s\":0.03,"
       <<"\"minimum_steady_vx_m_s\":"<<min_v<<",\"maximum_steady_vx_m_s\":"<<max_v
       <<",\"displacement_m\":"<<displacement<<",\"peak_force_N\":"<<peak
       <<",\"lift_clearance_m\":"<<held.tray[2]-grounded_z<<",\"lifted_force_x_N\":0,"
       <<"\"stop_check\":true,\"expired_command_check\":true,\"out_of_region_check\":true,\"invalid_speed_check\":true}\n";
    std::cout<<"PASS: transport, speed bound, stop, expiry and physical lift disengagement\n";
  } catch(const std::exception& e) {out<<"{\"status\":\"FAIL\"}\n";std::cerr<<"FAIL: "<<e.what()<<'\n';mj_deleteModel(m);return 1;}
  mj_deleteModel(m);return 0;
}
