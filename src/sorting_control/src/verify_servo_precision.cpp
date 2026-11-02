// Node-free KDL small-step regression. This does not start MoveIt Servo or DDS.
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl_kinematics_parameters.hpp>
#include <urdf/model.h>
#include <yaml-cpp/yaml.h>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>

void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv){
  if(argc!=4){std::cerr<<"Usage: verify_servo_precision URDF EPSILON OUTPUT_JSON\n";return 2;}
  try{
    const double epsilon=std::stod(argv[2]);
    need(std::isfinite(epsilon) && epsilon>0 && epsilon<=1e-7,"Servo IK epsilon must be positive and at most 1e-7");
    std::ifstream input(argv[1]);std::stringstream text;text<<input.rdbuf();
    urdf::Model urdf;need(urdf.initString(text.str()),"URDF parse failed");
    KDL::Tree tree;need(kdl_parser::treeFromUrdfModel(urdf,tree),"KDL tree parse failed");
    KDL::Chain chain;need(tree.getChain("panda_link0","panda_hand",chain),"Arm chain missing");
    need(chain.getNrOfJoints()==7,"Expected seven arm joints");
    KDL::JntArray lo(7),hi(7);unsigned int index=0;
    for(const auto& segment:chain.segments){
      if(segment.getJoint().getType()==KDL::Joint::None)continue;
      auto joint=urdf.getJoint(segment.getJoint().getName());need(joint && joint->limits,"Joint limits missing");
      lo(index)=joint->limits->lower;hi(index)=joint->limits->upper;++index;
    }
    const double original=kdl_kinematics::Params().epsilon;
    need(original==1e-5,"Installed KDL default changed; review the regression");
    const std::array<std::array<double,7>,2> seeds={{
      {{.404723006911706,.183215664893790,.384548382460216,-1.591206340947123,-.069648056649230,1.760867013596325,.011214648643756}},
      {{-.405650795733638,.181179372231751,-.381565319798395,-1.604210342187990,.068542848462563,1.772181254718039,-1.580675974594062}}
    }};
    YAML::Node report;report["status"]="PASS";report["installed_kdl_default_epsilon"]=original;report["servo_epsilon"]=epsilon;
    report["scope"]="Native KDL chain solves; installed MoveIt parameter default. Live plugin, smoothing, middleware and physical Servo execution NOT_RUN.";
    for(unsigned int pose=0;pose<seeds.size();++pose){
      for(double speed:{.0003,.00025506,.000063765,.000038259}){
        YAML::Node row;row["pose"]=pose==0?"red_live_lower":"green_native_lower";row["speed_m_s"]=speed;row["step_m"]=speed*.01;
        for(unsigned int variant=0;variant<2;++variant){
          const double eps=variant==0?original:epsilon;
          KDL::ChainFkSolverPos_recursive fk(chain);KDL::ChainIkSolverVel_pinv velocity(chain);
          KDL::ChainIkSolverPos_NR_JL solve(chain,lo,hi,fk,velocity,100,eps);
          KDL::JntArray q(7),out(7);for(unsigned int i=0;i<7;++i)q(i)=seeds[pose][i];
          KDL::Frame first,current;need(fk.JntToCart(q,first)>=0,"Initial FK failed");
          for(unsigned int step=0;step<100;++step){
            need(fk.JntToCart(q,current)>=0,"FK failed");auto target=current;target.p.z(target.p.z()-speed*.01);
            need(solve.CartToJnt(q,target,out)>=0,"Small-step KDL solve failed");q=out;
          }
          need(fk.JntToCart(q,current)>=0,"Final FK failed");const double descent=first.p.z()-current.p.z();
          row[variant==0?"default_descent_m":"corrected_descent_m"]=descent;
          if(variant==0)need(std::abs(descent)<1e-10,"Default tolerance did not reproduce the stationary seed");
          else{
            need(std::abs(descent-speed)<speed*.05,"Corrected IK did not resolve the slow descent");
            need(KDL::diff(first,current).rot.Norm()<1e-6,"Small-step IK changed hand orientation");
          }
        }
        report["cases"].push_back(row);
      }
    }
    // JSON is also a YAML flow mapping; emit scalar strings as JSON strings.
    std::ofstream output(argv[3]);output<<"{\n  \"status\": \"PASS\",\n  \"installed_kdl_default_epsilon\": "<<original
      <<",\n  \"servo_epsilon\": "<<epsilon<<",\n  \"scope\": \"Native KDL chain solves; live MoveIt Servo NOT_RUN\",\n  \"cases\": [\n";
    for(std::size_t i=0;i<report["cases"].size();++i){auto row=report["cases"][i];if(i)output<<",\n";
      output<<"    {\"pose\": \""<<row["pose"].as<std::string>()<<"\", \"speed_m_s\": "<<row["speed_m_s"].as<double>()
        <<", \"step_m\": "<<row["step_m"].as<double>()<<", \"default_descent_m\": "<<row["default_descent_m"].as<double>()
        <<", \"corrected_descent_m\": "<<row["corrected_descent_m"].as<double>()<<"}";
    }
    output<<"\n  ]\n}\n";need(output.good(),"Could not save precision report");
    std::cout<<"PASS: installed KDL default and eight native sub-tolerance descent cases\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
