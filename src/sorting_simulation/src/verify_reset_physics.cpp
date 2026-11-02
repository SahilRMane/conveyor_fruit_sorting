#include "sorting_simulation/reset_record.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
  if(argc!=3) {std::cerr<<"Usage: verify_reset_physics SCENE_XML OUTPUT_JSON\n";return 2;}
  std::ofstream out(argv[2]);if(!out) return 2;
  char error[2048]={};mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m) {std::cerr<<error<<'\n';out<<"{\"status\":\"FAIL\"}";return 1;}
  mjData* d=mj_makeData(m);
  try {
    int home=mj_name2id(m,mjOBJ_KEY,"home"),tray=mj_name2id(m,mjOBJ_BODY,"tray");
    if(home<0 || tray<0 || m->nq!=16 || m->nu!=8) throw std::runtime_error("Unexpected scene");
    mj_resetDataKeyframe(m,d,home);
    for(int check=0;check<20;++check) {
      // Disturb the state through commands and applied forces, then verify reset.
      d->ctrl[0]=0.05;d->ctrl[7]=0.02;
      d->qfrc_applied[0]=1.0;d->xfrc_applied[6*tray+2]=1.0;
      double end=d->time+0.5;while(d->time<end) mj_step(m,d);
      if(std::fabs(d->qpos[0])<0.005) throw std::runtime_error("Disturbance did not move the arm");
      mj_resetDataKeyframe(m,d,home);
      for(mjtSize i=0;i<m->nq;++i)
        if(std::fabs(d->qpos[i]-m->key_qpos[home*m->nq+i])>1e-10) throw std::runtime_error("Reset position mismatch");
      for(mjtSize i=0;i<m->nu;++i)
        if(std::fabs(d->ctrl[i]-m->key_ctrl[home*m->nu+i])>1e-10) throw std::runtime_error("Reset control mismatch");
      if(sorting_simulation::maximum_absolute(d->qvel,m->nv)>1e-10 ||
         sorting_simulation::maximum_absolute(d->qfrc_applied,m->nv)>1e-10 ||
         sorting_simulation::maximum_absolute(d->xfrc_applied,6*m->nbody)>1e-10)
        throw std::runtime_error("Reset retained velocity or applied force");
    }
    out<<"{\"status\":\"PASS\",\"native_reset_checks\":20,\"last_reset_record\":"
       <<sorting_simulation::reset_json(m,d,20)<<"}\n";
    std::cout<<"PASS: 20 native reset checks; positions, velocities, commands and applied forces restored\n";
  } catch(const std::exception& e) {
    out<<"{\"status\":\"FAIL\"}\n";std::cerr<<e.what()<<'\n';mj_deleteData(d);mj_deleteModel(m);return 1;
  }
  mj_deleteData(d);mj_deleteModel(m);return 0;
}
