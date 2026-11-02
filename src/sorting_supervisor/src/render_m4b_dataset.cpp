// Offline grouped, counterfactual RGB. Encoder configurations come from M4A.
// Direct state assignment is for dataset synthesis only, never live control.
#include "sorting_simulation/native_camera_render.hpp"
#include <random>
int main(int argc,char** argv) {
 if(argc!=6){std::cerr<<"Usage: render_m4b_dataset SCENE POSES_YAML CONFIG OUTPUT GROUP_OFFSET\n";return 2;}
 try {
  char error[2048]={};std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);check(bool(m),error);
  StaticMotion sim(m.get());Offscreen renderer;renderer.init(m.get());
  auto poses=YAML::LoadFile(argv[2])["views"],cfg=YAML::LoadFile(argv[3]);
  int groups=cfg["dataset_groups"].as<int>(),offset=std::stoi(argv[5]);
  std::filesystem::path out(argv[4]);check(!std::filesystem::exists(out/"frames.yaml"),"Use a new dataset directory");std::filesystem::create_directories(out);
  int camera=mj_name2id(m.get(),mjOBJ_CAMERA,"wrist"),skin=mj_name2id(m.get(),mjOBJ_GEOM,"fruit_class_skin"),cover=mj_name2id(m.get(),mjOBJ_GEOM,"fruit_front_occluder");
  check(camera>=0&&skin>=0&&cover>=0,"Missing class fixtures");
  YAML::Node manifest;manifest["scope"]="Stationary grouped M4B synthetic-class views; not a physical motion test";
  for(int g=0;g<groups;++g) {
   int seed=cfg["dataset_seed"].as<int>()+(g+offset)*7919;std::mt19937 rng(seed);std::uniform_real_distribution<double> u(0.,1.);
   double light=.55+.2*u(rng),high=.68+.2*u(rng),low=.04+.07*u(rng),shade=.38+.08*u(rng);
   double dx=-.002+.004*u(rng),dy=-.001+.002*u(rng);
   std::array<double,7> jitter{};jitter[0]=-.003+.006*u(rng);jitter[1]=-.003+.006*u(rng);jitter[6]=-.005+.01*u(rng);
   for(int l=0;l<m->nlight;++l)for(int c=0;c<3;++c)m->light_diffuse[3*l+c]=light;
   for(int c=0;c<3;++c)m->geom_rgba[4*cover+c]=shade;
   for(const char* view:{"assess","inspect_left","inspect_right"}) {
    mj_resetDataKeyframe(m.get(),sim.data,mj_name2id(m.get(),mjOBJ_KEY,"home"));
    for(int j=0;j<7;++j)sim.data->qpos[j]=poses[view]["joints"][j].as<double>()+jitter[j];
    sim.data->qpos[9]=.55+dx;sim.data->qpos[10]=dy;sim.data->qpos[11]=.3059;mj_forward(m.get(),sim.data);
    for(const char* condition:{"clear","partial","covered"}) {
     bool covered=std::string(condition)=="covered",clear=std::string(condition)=="clear";
     const double partial_pos[3]={-.0357,.0407,.0776},full_pos[3]={-.04,.042,.055};
     const double partial_size[3]={.021,.018,.002},full_size[3]={.040,.040,.002};
     for(int j=0;j<3;++j){m->geom_pos[3*cover+j]=(covered?full_pos:partial_pos)[j];m->geom_size[3*cover+j]=(covered?full_size:partial_size)[j];}
     m->geom_rgba[4*cover+3]=clear?0.:1.;mj_forward(m.get(),sim.data);
     for(int label=0;label<2;++label) {
      for(int c=0;c<3;++c)m->geom_rgba[4*skin+c]=c==label?high:low;
      std::string id="g"+std::to_string(g+offset)+"-"+view+"-"+condition+"-c"+std::to_string(label);
      renderer.save(m.get(),sim.data,camera,(out/(id+".ppm")).string());
      YAML::Node row;row["id"]=id;row["image"]=id+".ppm";row["group_id"]=g+offset;row["appearance_instance"]=seed;
      row["label"]=label;row["pose"]=view;row["condition"]=condition;
      for(int j=0;j<3;++j)row["camera_xyz"].push_back(sim.data->cam_xpos[3*camera+j]);
      for(int j=0;j<9;++j)row["camera_optical_rotation"].push_back(sim.data->cam_xmat[9*camera+j]*(j%3==0?1:-1));
      int handle=mj_name2id(m.get(),mjOBJ_SITE,"handle_grasp_site");for(int j=0;j<3;++j)row["evaluator_handle_xyz"].push_back(sim.data->site_xpos[3*handle+j]);
      row["width"]=m->cam_resolution[2*camera];row["height"]=m->cam_resolution[2*camera+1];row["fovy_deg"]=m->cam_fovy[camera];manifest["frames"].push_back(row);
     }
    }
   }
   if(g%20==0)std::cout<<"Rendered group "<<g<<'/'<<groups<<std::endl;
  }
  std::ofstream file(out/"frames.yaml");file<<manifest;std::cout<<"PASS: matched class/view groups\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
