#include <sorting_control/precision_smoothing.hpp>
#include <moveit/online_signal_smoothing/smoothing_base_class.hpp>
#include <pluginlib/class_loader.hpp>
#include <fstream>
#include <iostream>
#include <vector>
using sorting_control::PrecisionStep;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
PrecisionStep configured(){PrecisionStep s;s.qmin=Eigen::VectorXd::Constant(7,-3.);s.qmax=Eigen::VectorXd::Constant(7,3.);s.vmin=Eigen::VectorXd::Constant(7,-.5);s.vmax=-s.vmin;s.amin=Eigen::VectorXd::Constant(7,-1.);s.amax=-s.amin;s.reset(Eigen::VectorXd::Zero(7),Eigen::VectorXd::Zero(7));return s;}
int main(int argc,char** argv){
 if(argc!=2){std::cerr<<"Usage: verify_precision_smoothing OUTPUT_JSON\n";return 2;}
 try{
  pluginlib::ClassLoader<online_signal_smoothing::SmoothingBaseClass> loader("moveit_core","online_signal_smoothing::SmoothingBaseClass");
  auto plugin=loader.createSharedInstance("sorting_control/PrecisionAccelerationLimitedPlugin");
  need(bool(plugin),"Precision smoothing plugin is not discoverable");
  int cases=0;
  for(double speed:{.00002,.00008,.0003,.001})for(double sign:{-1.,1.}){
    auto s=configured();Eigen::VectorXd v=Eigen::VectorXd::Constant(7,sign*speed);s.reset(Eigen::VectorXd::Zero(7),v);
    for(int i=0;i<1000;++i){auto requested=s.position+v*s.dt;Eigen::VectorXd out=s.position,velocity=s.velocity,a=Eigen::VectorXd::Zero(7);
      need(s.repair(requested,v,out,velocity,a),"Constant small-step repair did not activate");
      need((a.array().abs()<=1.+1e-10).all(),"Acceleration bound failed");
    }
    need((s.position-v*10.).norm()<1e-10,"Small commands did not accumulate");++cases;
  }
  {auto s=configured();Eigen::VectorXd desired=s.position,velocity=Eigen::VectorXd::Constant(7,2e-5),out=s.position,v=s.velocity,a;
   desired+=velocity*s.dt;need(s.repair(desired,velocity,out,v,a),"Tiny start from rest was lost");++cases;}
  {auto s=configured();Eigen::VectorXd requested=Eigen::VectorXd::Constant(7,.001),v=Eigen::VectorXd::Constant(7,.1),out=requested,a=Eigen::VectorXd::Zero(7),velocity=v;
   need(!s.repair(requested,v,out,velocity,a),"Normal original-filter output was changed");need((out-requested).norm()==0,"Normal motion changed");++cases;}
  {auto s=configured();s.amin.setConstant(-.001);s.amax.setConstant(.001);Eigen::VectorXd v=Eigen::VectorXd::Constant(7,5e-6),requested=v*s.dt,out=s.position,actual=s.velocity,a;
   need(s.repair(requested,v,out,actual,a),"Bounded recovery missing");need((a.array().abs()<=.001+1e-12).all(),"Tight acceleration bound failed");++cases;}
  {auto s=configured();s.vmin.setConstant(-1e-6);s.vmax.setConstant(1e-6);Eigen::VectorXd v=Eigen::VectorXd::Constant(7,2e-5),requested=v*s.dt,out=s.position,actual=s.velocity,a;
   s.repair(requested,v,out,actual,a);need((actual.array().abs()<=1e-6+1e-12).all(),"Velocity bound failed");++cases;}
  {auto s=configured();s.qmax.setConstant(1e-8);Eigen::VectorXd v=Eigen::VectorXd::Constant(7,2e-5),requested=v*s.dt,out=s.position,actual=s.velocity,a;
   s.repair(requested,v,out,actual,a);need((out.array()<=1e-8+1e-12).all(),"Position bound failed");++cases;}
  {auto s=configured();s.velocity.setConstant(2e-5);Eigen::VectorXd v=Eigen::VectorXd::Zero(7),requested=s.position,out=s.position,actual=s.velocity,a;
   need(s.repair(requested,v,out,actual,a),"Small stop was lost");need(actual.norm()==0,"Stale small velocity persisted");++cases;}
  {auto s=configured();Eigen::VectorXd requested=s.position,v=s.velocity,out=s.position,a;requested[0]=std::nan("");bool rejected=false;
   try{s.repair(requested,v,out,v,a);}catch(const std::exception&){rejected=true;}need(rejected,"Nonfinite request accepted");++cases;}
  {auto s=configured();s.qmax.setZero();s.velocity.setConstant(.01);s.amin.setConstant(-.001);s.amax.setConstant(.001);Eigen::VectorXd requested=s.position+Eigen::VectorXd::Constant(7,1e-6),v=s.velocity,out=s.position,actual=s.velocity,a;bool rejected=false;
   try{s.repair(requested,v,out,actual,a);}catch(const std::exception&){rejected=true;}need(rejected,"Infeasible bound was accepted");++cases;}
  std::ofstream out(argv[1]);out<<"{\"status\":\"PASS\",\"cases\":"<<cases<<",\"scope\":\"Node-free recovery kernel; installed filter composition checked separately\"}\n";need(out.good(),"Could not save smoothing report");
  std::cout<<"PASS: "<<cases<<" bounded small-step smoothing cases (no ROS node)\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
