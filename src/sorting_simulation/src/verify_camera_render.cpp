// Standalone camera calibration fixtures. No ROS node and no grasp claim.
#include "sorting_simulation/static_motion.hpp"
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
struct Offscreen {
  EGLDisplay display=EGL_NO_DISPLAY;EGLContext context=EGL_NO_CONTEXT;EGLSurface surface=EGL_NO_SURFACE;
  mjvScene scene{};mjrContext render{};mjvOption options{};mjvCamera camera{};
  bool ready=false;
  void init(const mjModel* m){
    display=eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,nullptr);
    EGLint major,minor;check(eglInitialize(display,&major,&minor),"EGL initialization failed");
    check(eglBindAPI(EGL_OPENGL_API),"EGL OpenGL binding failed");
    EGLint attributes[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_BIT,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_DEPTH_SIZE,24,EGL_NONE};
    EGLConfig config;EGLint count;check(eglChooseConfig(display,attributes,&config,1,&count)&&count>0,"EGL configuration failed");
    context=eglCreateContext(display,config,EGL_NO_CONTEXT,nullptr);check(context!=EGL_NO_CONTEXT,"EGL context failed");
    EGLint size[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};surface=eglCreatePbufferSurface(display,config,size);
    check(surface!=EGL_NO_SURFACE && eglMakeCurrent(display,surface,surface,context),"EGL offscreen surface failed");
    mjv_defaultScene(&scene);mjr_defaultContext(&render);mjv_defaultOption(&options);mjv_defaultCamera(&camera);
    for(int i=0;i<mjNGROUP;++i)options.sitegroup[i]=0;
    mjv_makeScene(m,&scene,2000);mjr_makeContext(m,&render,mjFONTSCALE_100);mjr_setBuffer(mjFB_OFFSCREEN,&render);ready=true;
  }
  void save(const mjModel* m,mjData* d,int id,const std::string& filename){
    int w=m->cam_resolution[2*id],h=m->cam_resolution[2*id+1];mjrRect rectangle{0,0,w,h};
    camera.type=mjCAMERA_FIXED;camera.fixedcamid=id;
    mjv_updateScene(m,d,&options,nullptr,&camera,mjCAT_ALL,&scene);mjr_render(rectangle,&scene,&render);
    std::vector<unsigned char> rgb(w*h*3);mjr_readPixels(rgb.data(),nullptr,rectangle,&render);
    std::ofstream image(filename,std::ios::binary);check(bool(image),"Cannot write rendered image");image<<"P6\n"<<w<<' '<<h<<"\n255\n";
    for(int row=h-1;row>=0;--row)image.write(reinterpret_cast<const char*>(rgb.data()+row*w*3),w*3);
  }
  YAML::Node benchmark(const mjModel* m,mjData* d){
    YAML::Node out;out["gl_renderer"]=reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    out["shadow_map_size"]=m->vis.quality.shadowsize;out["offscreen_samples"]=m->vis.quality.offsamples;
    int shadow_lights=0;for(int i=0;i<m->nlight;++i)shadow_lights+=m->light_castshadow[i]!=0;
    out["shadow_casting_lights"]=shadow_lights;
    int repeats=24;
    for(const char* view:{"overhead","wrist"}){
      int id=mj_name2id(m,mjOBJ_CAMERA,view);int w=m->cam_resolution[2*id],h=m->cam_resolution[2*id+1];
      mjrRect rect{0,0,w,h};std::vector<unsigned char> rgb(w*h*3);std::vector<float> depth(w*h);
      camera.type=mjCAMERA_FIXED;camera.fixedcamid=id;
      std::vector<double> times;
      for(int i=0;i<repeats+2;++i){
        auto start=std::chrono::steady_clock::now();
        mjv_updateScene(m,d,&options,nullptr,&camera,mjCAT_ALL,&scene);mjr_render(rect,&scene,&render);
        mjr_readPixels(rgb.data(),depth.data(),rect,&render);
        double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(i>=2)times.push_back(elapsed);
      }
      std::sort(times.begin(),times.end());double sum=0;for(double t:times)sum+=t;
      out["views"][view]["mean_render_read_wall_s"]=sum/times.size();
      out["views"][view]["p95_render_read_wall_s"]=times[static_cast<size_t>(std::ceil(.95*times.size()))-1];
      out["views"][view]["maximum_render_read_wall_s"]=times.back();
      out["views"][view]["frames"]=repeats;
    }
    out["scope"]="Native render/read only; excludes ROS transport and delivery";
    return out;
  }
  ~Offscreen(){
    if(ready){mjv_freeScene(&scene);mjr_freeContext(&render);}
    if(display!=EGL_NO_DISPLAY){eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
      if(surface!=EGL_NO_SURFACE)eglDestroySurface(display,surface);
      if(context!=EGL_NO_CONTEXT)eglDestroyContext(display,context);
      eglTerminate(display);}
  }
};
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
