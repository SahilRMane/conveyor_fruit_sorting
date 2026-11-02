#include <hardware_interface/component_parser.hpp>
#include <mujoco/mujoco.h>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <utility>
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv){
  if(argc!=4){std::cerr<<"Usage: verify_wrist_model URDF SENSOR_SCENE BASELINE_SCENE\n";return 2;}
  try{
    std::ifstream file(argv[1]);std::stringstream input;input<<file.rdbuf();
    auto systems=hardware_interface::parse_control_resources_from_urdf(input.str());
    need(systems.size()==1,"Expected one hardware system");auto& hw=systems[0];
    need(hw.sensors.size()==1 && hw.sensors[0].name=="panda_hand_wrench_fts","Wrist sensor ROS name mismatch");
    std::set<std::string> names;for(auto& interface:hw.sensors[0].state_interfaces)names.insert(interface.name);
    need(names==std::set<std::string>{"force.x","force.y","force.z","torque.x","torque.y","torque.z"},"Wrist interfaces mismatch");
    need(hw.sensors[0].parameters.at("frame_id")=="panda_hand","Wrist sensor frame mismatch");
    // register_sensors() skips a sensor without mujoco_type. The _fts
    // suffix alone does not select the installed driver's mapping.
    auto& parameters=hw.sensors[0].parameters;
    need(parameters.count("mujoco_type") && parameters.at("mujoco_type")=="fts",
         "Wrist sensor requires mujoco_type=fts; the driver otherwise skips it");
    need(parameters.count("mujoco_sensor_name") && parameters.at("mujoco_sensor_name")=="panda_hand_wrench",
         "Wrist sensor requires mujoco_sensor_name=panda_hand_wrench");
    char error[2048];using Model=std::unique_ptr<mjModel,decltype(&mj_deleteModel)>;
    Model model(mj_loadXML(argv[2],nullptr,error,sizeof(error)),mj_deleteModel);need(bool(model),error);
    Model baseline(mj_loadXML(argv[3],nullptr,error,sizeof(error)),mj_deleteModel);need(bool(baseline),error);
    need(model->nq==baseline->nq && model->nv==baseline->nv && model->nu==baseline->nu && model->nbody==baseline->nbody,"Wrist sensor changed physical dimensions");
    for(int i=0;i<model->nbody;++i){need(model->body_mass[i]==baseline->body_mass[i],"Body mass changed");for(int j=0;j<3;++j)need(model->body_inertia[3*i+j]==baseline->body_inertia[3*i+j],"Body inertia changed");}
    int site=mj_name2id(model.get(),mjOBJ_SITE,"panda_hand_wrench_site");
    need(site>=0 && model->site_bodyid[site]==mj_name2id(model.get(),mjOBJ_BODY,"hand"),"Wrist site is not on hand");
    for(const auto& component:{std::make_pair("force",mjSENS_FORCE),std::make_pair("torque",mjSENS_TORQUE)}){
      const std::string key=std::string(component.first)+"_mjcf_suffix";
      const std::string suffix=hw.hardware_parameters.count(key)?hw.hardware_parameters.at(key):std::string("_")+component.first;
      const std::string name=parameters.at("mujoco_sensor_name")+suffix;
      int sensor=mj_name2id(model.get(),mjOBJ_SENSOR,name.c_str());
      need(sensor>=0 && model->sensor_type[sensor]==component.second && model->sensor_dim[sensor]==3 && model->sensor_objid[sensor]==site,"Physical wrist sensor mapping mismatch");
    }
    std::cout<<"PASS: physical wrist sensor, driver selection and MJCF binding (offline)\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
