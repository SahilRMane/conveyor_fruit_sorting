#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sorting_control {
// Pure numerical part of the wrapper; it neither starts ROS nor commands hardware.
class PrecisionStep {
public:
  Eigen::VectorXd position,velocity,qmin,qmax,vmin,vmax,amin,amax;
  double dt=.01;
  bool ready=false;
  void reset(const Eigen::VectorXd& p,const Eigen::VectorXd& v){
    if(p.size()==0 || p.size()!=v.size() || !p.allFinite() || !v.allFinite())throw std::runtime_error("Invalid smoother reset");
    position=p;velocity=v;ready=true;
  }
  bool repair(const Eigen::VectorXd& requested_p,const Eigen::VectorXd& requested_v,
              Eigen::VectorXd& filtered_p,Eigen::VectorXd& filtered_v,Eigen::VectorXd& filtered_a){
    const auto n=position.size();
    if(!ready || !(dt>0 && std::isfinite(dt)) || requested_p.size()!=n || requested_v.size()!=n ||
       filtered_p.size()!=n || filtered_v.size()!=n || !requested_p.allFinite() || !requested_v.allFinite() ||
       !filtered_p.allFinite() || !filtered_v.allFinite())throw std::runtime_error("Invalid smoothing state");
    // Match only the installed acceleration filter's constant-output branch.
    // All other output continues through the original filter unchanged.
    const bool froze=(filtered_p-position).norm()<=1e-12 && (filtered_v-velocity).norm()<=1e-12;
    const bool small=(requested_p-position).norm()<1e-4 && (requested_v-velocity).norm()<1e-4;
    const bool change=(requested_p-position).norm()>1e-12 || (requested_v-velocity).norm()>1e-12;
    const bool repair=froze && small && change;
    if(repair){
      if(qmin.size()!=n || qmax.size()!=n || vmin.size()!=n || vmax.size()!=n || amin.size()!=n || amax.size()!=n ||
         !qmin.allFinite() || !qmax.allFinite() || !vmin.allFinite() || !vmax.allFinite() || !amin.allFinite() || !amax.allFinite())
        throw std::runtime_error("Invalid smoother bounds");
      filtered_a.resize(n);
      for(Eigen::Index i=0;i<n;++i){
        const double low=std::max({vmin[i],velocity[i]+amin[i]*dt,(qmin[i]-position[i])/dt});
        const double high=std::min({vmax[i],velocity[i]+amax[i]*dt,(qmax[i]-position[i])/dt});
        if(low>high || amin[i]>0 || amax[i]<0 || vmin[i]>0 || vmax[i]<0)throw std::runtime_error("Infeasible smoother bounds");
        filtered_v[i]=std::clamp((requested_p[i]-position[i])/dt,low,high);
        filtered_p[i]=position[i]+filtered_v[i]*dt;
        filtered_a[i]=(filtered_v[i]-velocity[i])/dt;
      }
    }
    position=filtered_p;velocity=filtered_v;return repair;
  }
};
}
