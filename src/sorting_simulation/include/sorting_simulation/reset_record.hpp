#pragma once
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace sorting_simulation {
inline double maximum_absolute(const mjtNum* values,mjtSize count) {
  double result=0;for(mjtSize i=0;i<count;++i) result=std::fmax(result,std::fabs(values[i]));return result;
}
inline void vector_json(std::ostream& out,const mjtNum* values,mjtSize count) {
  out<<'[';for(mjtSize i=0;i<count;++i) {if(i) out<<',';out<<values[i];}out<<']';
}
inline std::string reset_json(const mjModel* m,const mjData* d,std::uint64_t sequence) {
  std::ostringstream out;out<<std::setprecision(15);
  out<<"{\"schema_version\":1,\"reset_seq\":"<<sequence<<",\"sim_time\":"<<d->time<<",\"qpos\":";
  vector_json(out,d->qpos,m->nq);out<<",\"qvel\":";vector_json(out,d->qvel,m->nv);
  out<<",\"ctrl\":";vector_json(out,d->ctrl,m->nu);
  out<<",\"act_max_abs\":"<<maximum_absolute(d->act,m->na)
     <<",\"qfrc_applied_max_abs\":"<<maximum_absolute(d->qfrc_applied,m->nv)
     <<",\"xfrc_applied_max_abs\":"<<maximum_absolute(d->xfrc_applied,6*m->nbody)<<'}';
  return out.str();
}
}
