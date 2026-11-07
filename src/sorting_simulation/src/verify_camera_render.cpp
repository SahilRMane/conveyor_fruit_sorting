// Standalone camera calibration fixtures. No ROS node and no grasp claim.
#include "sorting_simulation/native_camera_render.hpp"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <GL/gl.h>
#include <memory>
using namespace sorting_simulation;
YAML::Node vector(const mjtNum* values,int length){YAML::Node out;for(int i=0;i<length;++i)out.push_back(values[i]);return out;}
int main(int argc,char** argv){
  if(argc!=5 && argc!=6){std::cerr<<"Usage: verify_camera_render CAMERA_SCENE BASELINE_SCENE CONFIG OUTPUT_DIR [--benchmark]\n";return 2;}
  mjModel* model=nullptr; mjModel* baseline=nullptr;
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model_owner(nullptr,mj_deleteModel),baseline_owner(nullptr,mj_deleteModel);
  try{
    char error[2048]={};model=mj_loadXML(argv[1],nullptr,error,sizeof(error));model_owner.reset(model);check(model,error);
    baseline=mj_loadXML(argv[2],nullptr,error,sizeof(error));baseline_owner.reset(baseline);check(baseline,error);
    check(model->nq==baseline->nq && model->nv==baseline->nv && model->nu==baseline->nu && model->nbody==baseline->nbody,"Camera scene changed physics dimensions");
    for(int b=0;b<model->nbody;++b){check(model->body_mass[b]==baseline->body_mass[b],"Camera scene changed body mass");
      for(int j=0;j<3;++j)check(model->body_inertia[3*b+j]==baseline->body_inertia[3*b+j],"Camera scene changed inertia");}
    int marker=mj_name2id(model,mjOBJ_GEOM,"neutral_localization_marker"),handle=mj_name2id(model,mjOBJ_SITE,"handle_grasp_site");
    check(marker>=0 && model->geom_contype[marker]==0 && model->geom_conaffinity[marker]==0,"Marker must have no physical collision");
    auto config=YAML::LoadFile(argv[3]);std::filesystem::path folder(argv[4]);std::filesystem::create_directories(folder);
    YAML::Node report;report["status"]="PASS";report["scope"]="Native RGB calibration fixtures; live camera plugin NOT_RUN";
    report["physical_mass_and_inertia_preserved"]=true;report["marker_physical_collision"]=false;
    StaticMotion sim(model);double maximum_speed=0;
    sim.guard=[&](const ContactState& s){check(s.forbidden==0 && s.left_N+s.right_N<0.1,"Calibration path physical contact");
      for(int i=0;i<7;++i){maximum_speed=std::max(maximum_speed,std::fabs(sim.data->qvel[i]));check(std::fabs(sim.data->qvel[i])<0.25,"Calibration path exceeds 0.25 rad/s");}};
    Offscreen render;render.init(model);
    if(argc==6){
      check(std::string(argv[5])=="--benchmark","Unknown native camera option");sim.step(2.0);
      report["render_benchmark"]=render.benchmark(model,sim.data);
      std::ofstream out(folder/"render-benchmark.yaml");out<<report;
      std::cout<<"PASS: native render benchmark\n";
      return 0;
    }
    auto record=[&](const std::string& label,const std::string& view,bool expect_marker){
      mj_forward(model,sim.data);int id=mj_name2id(model,mjOBJ_CAMERA,view.c_str());check(id>=0,"Missing camera");
      YAML::Node frame;frame["label"]=label;frame["view_id"]=view;frame["expect_marker"]=expect_marker;
      frame["image"]=label+".ppm";frame["camera_position"]=vector(sim.data->cam_xpos+3*id,3);
      mjtNum matrix[9],q[4];for(int r=0;r<3;++r)for(int c=0;c<3;++c)matrix[3*r+c]=sim.data->cam_xmat[9*id+3*r+c]*(c==0?1:-1);
      mju_mat2Quat(q,matrix);frame["camera_quat_xyzw"].push_back(q[1]);frame["camera_quat_xyzw"].push_back(q[2]);frame["camera_quat_xyzw"].push_back(q[3]);frame["camera_quat_xyzw"].push_back(q[0]);
      frame["handle_position"]=vector(sim.data->site_xpos+3*handle,3);frame["joint_positions"]=vector(sim.data->qpos,7);
      double z=model->geom_size[3*marker+2];mjtNum position[3];for(int i=0;i<3;++i)position[i]=sim.data->geom_xpos[3*marker+i]+z*sim.data->geom_xmat[9*marker+3*i+2];
      frame["marker_position"]=vector(position,3);frame["width"]=model->cam_resolution[2*id];frame["height"]=model->cam_resolution[2*id+1];frame["fovy_deg"]=model->cam_fovy[id];
      render.save(model,sim.data,id,(folder/(label+".ppm")).string());report["frames"].push_back(frame);
    };
    for(double x:{0.48,0.55,0.68,0.74}){
      mj_resetDataKeyframe(model,sim.data,mj_name2id(model,mjOBJ_KEY,"home"));sim.data->qpos[9]=x;mj_forward(model,sim.data);sim.step(2.0);
      record("overhead-"+std::to_string(x),"overhead",true);
    }
    mj_resetDataKeyframe(model,sim.data,mj_name2id(model,mjOBJ_KEY,"home"));mj_forward(model,sim.data);sim.step(2.0);
    record("wrist-home","wrist",true);
    std::array<double,3> inspection;for(int i=0;i<3;++i)inspection[i]=config["inspection_hand_xyz_m"][i].as<double>();
    ArmVector goal=sim.solve(inspection);report["inspection_joint_positions"]=vector(goal.data(),7);
    sim.move(goal,config["inspection_motion_duration_s"].as<double>(),0.5);record("wrist-inspection","wrist",true);
    goal[0]+=0.04;sim.move(goal,2.0,0.4);record("wrist-rotated","wrist",true);
    report["maximum_calibration_joint_speed_rad_s"]=maximum_speed;
    std::ofstream out(folder/"native-render.yaml");out<<report;std::cout<<"PASS: native RGB fixtures and physical camera path\n";
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
  return 0;
}
