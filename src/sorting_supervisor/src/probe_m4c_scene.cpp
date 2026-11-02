// Node-free RGB domain probes and free-body conveyor dynamics. No ROS participant.
#include "sorting_simulation/native_camera_render.hpp"
#include "sorting_simulation/conveyor_transport.hpp"

int main(int argc,char** argv) {
 if(argc!=5){std::cerr<<"Usage: probe_m4c_scene SCENE M4A_VIEWS CONFIG OUTPUT\n";return 2;}
 try {
  char error[2048]={};std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);check(bool(m),error);
  auto poses=YAML::LoadFile(argv[2])["views"],cfg=YAML::LoadFile(argv[3]);
  std::filesystem::path out(argv[4]);check(!std::filesystem::exists(out/"probe.yaml"),"Use a new probe directory");std::filesystem::create_directories(out);
  StaticMotion sim(m.get());Offscreen render;render.init(m.get());YAML::Node report;
  int camera=mj_name2id(m.get(),mjOBJ_CAMERA,"wrist"),overhead=mj_name2id(m.get(),mjOBJ_CAMERA,"overhead");
  int skin=mj_name2id(m.get(),mjOBJ_GEOM,"fruit_class_skin"),cover=mj_name2id(m.get(),mjOBJ_GEOM,"fruit_front_occluder"),handle=mj_name2id(m.get(),mjOBJ_SITE,"handle_grasp_site");
  check(camera>=0&&overhead>=0&&skin>=0&&cover>=0&&handle>=0,"Missing M4C fixtures");
  auto row=[&](int cam,const std::string& image){
   YAML::Node r;r["image"]=image;r["capture_ns"]=static_cast<long long>(std::llround(sim.data->time*1e9));
   for(int j=0;j<3;++j)r["camera_xyz"].push_back(sim.data->cam_xpos[3*cam+j]);
   for(int j=0;j<9;++j)r["camera_optical_rotation"].push_back(sim.data->cam_xmat[9*cam+j]*(j%3==0?1:-1));
   for(int j=0;j<3;++j)r["evaluator_handle_xyz"].push_back(sim.data->site_xpos[3*handle+j]);
   r["width"]=m->cam_resolution[2*cam];r["height"]=m->cam_resolution[2*cam+1];r["fovy_deg"]=m->cam_fovy[cam];return r;
  };
  // Direct position assignment is restricted to this offline image-domain probe.
  for(const char* view:{"assess","inspect_left"})for(int i=0;i<=12;++i) {
   double x=.50+.01*i;mj_resetDataKeyframe(m.get(),sim.data,mj_name2id(m.get(),mjOBJ_KEY,"home"));
   for(int j=0;j<7;++j)sim.data->qpos[j]=poses[view]["joints"][j].as<double>();
   sim.data->qpos[9]=x;sim.data->qpos[10]=0;sim.data->qpos[11]=.3059;
   for(const char* condition:{"clear","partial","covered"}) {
    bool full=std::string(condition)=="covered",clear=std::string(condition)=="clear";
    const double partial_pos[3]={-.0357,.0407,.0776},full_pos[3]={-.04,.042,.055},partial_size[3]={.021,.018,.002},full_size[3]={.040,.040,.002};
    for(int j=0;j<3;++j){m->geom_pos[3*cover+j]=(full?full_pos:partial_pos)[j];m->geom_size[3*cover+j]=(full?full_size:partial_size)[j];}
    m->geom_rgba[4*cover+3]=clear?0.:1.;mj_forward(m.get(),sim.data);
    for(int label=0;label<2;++label){
     for(int c=0;c<3;++c)m->geom_rgba[4*skin+c]=c==label?.8:.08;
     std::string name=std::string(view)+"-x"+std::to_string(i)+"-"+condition+"-c"+std::to_string(label)+".ppm";
     render.save(m.get(),sim.data,camera,(out/name).string());auto r=row(camera,name);r["pose"]=view;r["x_m"]=x;r["condition"]=condition;r["label"]=label;report["domain_frames"].push_back(r);
    }
   }
  }
  // The dynamics test changes the initial state only through the XML keyframe.
  mj_resetDataKeyframe(m.get(),sim.data,mj_name2id(m.get(),mjOBJ_KEY,"home"));mj_forward(m.get(),sim.data);
  sim.step(2.);
  ConveyorTransport transport(m.get());double begin=sim.data->time,next_capture=begin;
  double camera_period=1./cfg["camera_rate_hz"].as<double>();ArmVector previous=Eigen::Map<ArmVector>(sim.data->ctrl);
  double speed=cfg["belt_speed_m_s"].as<double>();check(speed>0&&speed<=.03,"Invalid M4C engineering speed");
  for(const char* pose:{"wait_entry","assess","inspect_left","assess"}) {
   bool waiting=std::string(pose)=="wait_entry";ArmVector target=previous;
   if(!waiting){for(int j=0;j<7;++j)target[j]=poses[pose]["joints"][j].as<double>();}
   double started=sim.data->time;
   double duration=waiting?std::max(0.,(cfg["assess_trigger_x_m"].as<double>()-cfg["upstream_x_m"].as<double>())/speed-5.):14.;
   check(duration<120.,"Native upstream wait exceeds its bound");
   while(sim.data->time-started<duration) {
    double u=std::clamp((sim.data->time-started)/12.,0.,1.),blend=u*u*(3.-2.*u);
    for(int j=0;j<7;++j)sim.data->ctrl[j]=previous[j]+blend*(target[j]-previous[j]);
    auto belt=transport.apply(sim.data,cfg["belt_speed_m_s"].as<double>(),true);sim.step(m->opt.timestep);auto state=sim.state();
    check(!state.forbidden&&state.support_N>.5&&state.left_N+state.right_N<.1,"Moving inspection lost support or made contact");
    check(std::min(state.fingers[0],state.fingers[1])>.039,"Moving inspection closed fingers");
    for(int j=0;j<7;++j)check(std::fabs(sim.data->qvel[j])<.25,"Native inspection joint speed exceeded bound");
    if(sim.data->time+1e-8>=next_capture) {
     std::string image="overhead-"+std::to_string(report["moving_frames"].size())+".ppm";
     render.save(m.get(),sim.data,overhead,(out/image).string());auto r=row(overhead,image);r["pose_transition"]=pose;r["vx_m_s"]=belt.vx;r["force_N"]=belt.force;
     report["moving_frames"].push_back(r);YAML::Node q;for(int j=0;j<7;++j)q.push_back(sim.data->qpos[j]);report["path_joint_positions"].push_back(q);next_capture+=camera_period;
    }
   }
   previous=target;
  }
  report["physics_duration_s"]=sim.data->time-begin;report["transport_mode"]="bounded_force; free tray; no pose overwrite during stepping";
  report["belt_speed_m_s"]=cfg["belt_speed_m_s"].as<double>();
  report["native_scope"]="Physical position actuators and RGB; live MoveIt Servo NOT_RUN";
  report["benchmark"]=render.benchmark(m.get(),sim.data);
  std::vector<double> pair_times;
  for(int repeat=0;repeat<26;++repeat){
   auto started=std::chrono::steady_clock::now();
   for(int cam:{overhead,camera}){
    int w=m->cam_resolution[2*cam],h=m->cam_resolution[2*cam+1];std::vector<unsigned char> rgb(w*h*3);std::vector<float> depth(w*h);
    render.camera.type=mjCAMERA_FIXED;render.camera.fixedcamid=cam;mjv_updateScene(m.get(),sim.data,&render.options,nullptr,&render.camera,mjCAT_ALL,&render.scene);
    mjrRect rect{0,0,w,h};mjr_render(rect,&render.scene,&render.render);mjr_readPixels(rgb.data(),depth.data(),rect,&render.render);
   }
   if(repeat>=2)pair_times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
  }
  std::sort(pair_times.begin(),pair_times.end());report["pair_render_p95_wall_s"]=pair_times[22];
  std::ofstream file(out/"probe.yaml");file<<report;
  std::cout<<"PASS: M4C RGB probes and free-body moving inspection\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
