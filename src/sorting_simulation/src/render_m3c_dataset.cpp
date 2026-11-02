// Seeded, grouped RGB data. Labels are offline only. No ROS node is started.
#include "sorting_simulation/native_camera_render.hpp"
#include <random>
#include <iomanip>
int main(int argc,char** argv){
 if(argc!=5){std::cerr<<"Usage: render_m3c_dataset SCENE BASELINE CONFIG OUT\n";return 2;}
 try{
  char error[2048]={};std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);check(bool(m),error);
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> b(mj_loadXML(argv[2],nullptr,error,sizeof(error)),mj_deleteModel);check(bool(b),error);
  check(m->nq==b->nq&&m->nv==b->nv&&m->nu==b->nu&&m->nbody==b->nbody,"Visual classifier fixture changed physics dimensions");
  for(int i=0;i<m->nbody;++i){check(m->body_mass[i]==b->body_mass[i],"Visual fruit changed mass");for(int j=0;j<3;++j)check(m->body_inertia[3*i+j]==b->body_inertia[3*i+j],"Visual fruit changed inertia");}
  auto cfg=YAML::LoadFile(argv[3]);int groups=cfg["dataset_groups"].as<int>(),seed=cfg["dataset_seed"].as<int>();
  std::filesystem::path out(argv[4]);check(!std::filesystem::exists(out/"frames.yaml"),"Dataset exists: use a new output directory");std::filesystem::create_directories(out);
  StaticMotion sim(m.get());sim.step(2.);Offscreen renderer;renderer.init(m.get());
  int skin=mj_name2id(m.get(),mjOBJ_GEOM,"fruit_class_skin"),cover=mj_name2id(m.get(),mjOBJ_GEOM,"fruit_front_occluder"),texture=mj_name2id(m.get(),mjOBJ_TEXTURE,"fruit_instance_texture");
  check(skin>=0&&cover>=0&&texture>=0,"Missing fruit assets");YAML::Node manifest;manifest["scope"]="Synthetic RGB fruit class cue; grouped views and counterfactual classes";
  manifest["physics_mass_and_inertia_preserved"]=true;manifest["dataset_seed"]=seed;
  int boundaries[4],n=0;const char* splits[]={"train","validation","calibration","test"};for(int i=0;i<4;++i){n+=cfg["split_group_counts"][splits[i]].as<int>();boundaries[i]=n;}check(n==groups,"Split group counts differ");
  for(int group=0;group<groups;++group){
   std::mt19937 rng(seed+group*7919);std::uniform_real_distribution<double> u(0.,1.);int part=0;while(group>=boundaries[part])++part;
   auto* pixels=m->tex_data+m->tex_adr[texture];int nc=m->tex_nchannel[texture];
   // Every scene seed owns one texture instance. The pair of class labels shares it.
   for(int y=0;y<m->tex_height[texture];++y)for(int x=0;x<m->tex_width[texture];++x){double value=.30+.30*u(rng);for(int c=0;c<nc;++c)pixels[(y*m->tex_width[texture]+x)*nc+c]=c<3?static_cast<unsigned char>(255*value):255;}
   mjr_uploadTexture(m.get(),&renderer.render,texture);
   // Unique held-out occluder arrangements, within the declared training ranges.
   double dx=-.003+.006*u(rng),tangent=-.003+.006*u(rng),shade=.38+.08*u(rng);
   m->geom_pos[3*cover]=-.04+dx;m->geom_pos[3*cover+1]=.042+tangent;m->geom_pos[3*cover+2]=.055-tangent;
   for(int c=0;c<3;++c)m->geom_rgba[4*cover+c]=shade;
   mj_resetDataKeyframe(m.get(),sim.data,mj_name2id(m.get(),mjOBJ_KEY,"home"));sim.data->qpos[9]=.535+.03*u(rng);sim.data->qpos[10]=-.006+.012*u(rng);
   sim.data->qpos[0]=sim.data->ctrl[0]=-.025+.05*u(rng);sim.data->qpos[1]=sim.data->ctrl[1]=-.025+.05*u(rng);
   sim.data->qpos[6]=sim.data->ctrl[6]=-.7853-.025+.05*u(rng);mj_forward(m.get(),sim.data);sim.step(1.);
   double light=.45+.35*u(rng);for(int l=0;l<m->nlight;++l)for(int c=0;c<3;++c)m->light_diffuse[3*l+c]=light;
   double high=.65+.30*u(rng),low=.03+.10*u(rng);
   for(int kind=0;kind<2;++kind){for(int c=0;c<3;++c)m->geom_rgba[4*skin+c]=(c==kind?high:low);m->geom_rgba[4*skin+3]=1.;
    for(int view=0;view<3;++view){const char* name=view==0?"overhead":"wrist";const char* condition=view==0?"overhead_hidden":(view==1?"wrist_clear":"wrist_occluded");m->geom_rgba[4*cover+3]=view==2?1.:0.;
     mj_forward(m.get(),sim.data);int camera=mj_name2id(m.get(),mjOBJ_CAMERA,name);std::string id="g"+std::to_string(group)+"-c"+std::to_string(kind)+"-"+condition;renderer.save(m.get(),sim.data,camera,(out/(id+".ppm")).string());
     YAML::Node row;row["id"]=id;row["image"]=id+".ppm";row["group_id"]=group;row["appearance_instance"]=seed+group*7919;row["occluder_offset_xyz"].push_back(dx);row["occluder_offset_xyz"].push_back(tangent);row["occluder_offset_xyz"].push_back(-tangent);row["split"]=splits[part];row["label"]=kind;row["view_id"]=name;row["condition"]=condition;
     row["camera_xyz"]=YAML::Node();for(int j=0;j<3;++j)row["camera_xyz"].push_back(sim.data->cam_xpos[3*camera+j]);
     for(int j=0;j<9;++j)row["camera_optical_rotation"].push_back(sim.data->cam_xmat[9*camera+j]*(j%3==0?1:-1));
     int handle=mj_name2id(m.get(),mjOBJ_SITE,"handle_grasp_site");for(int j=0;j<3;++j)row["evaluator_handle_xyz"].push_back(sim.data->site_xpos[3*handle+j]);
     row["fovy_deg"]=m->cam_fovy[camera];row["width"]=m->cam_resolution[2*camera];row["height"]=m->cam_resolution[2*camera+1];manifest["frames"].push_back(row);
    }
   }
   if(group%32==0)std::cout<<"Rendered group "<<group<<'/'<<groups<<std::endl;
  }
  std::ofstream file(out/"frames.yaml");file<<manifest;std::cout<<"PASS: grouped native RGB dataset\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;
}
