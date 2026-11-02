// Full installed MoveIt pipeline in complete saved scenes. No robot command API.
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/planning_pipeline/planning_pipeline.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/parameter_map.hpp>
#include <rclcpp/serialized_message.hpp>
#include <moveit_msgs/msg/motion_plan_response.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iostream>
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
std::string read(const std::string& path){std::ifstream input(path,std::ios::binary);need(bool(input),"Missing fixture: "+path);std::ostringstream text;text<<input.rdbuf();return text.str();}
template<class T>T message(const std::string& path){auto bytes=read(path);rclcpp::SerializedMessage serialized(bytes.size());auto& data=serialized.get_rcl_serialized_message();std::memcpy(data.buffer,bytes.data(),bytes.size());data.buffer_length=bytes.size();T result;rclcpp::Serialization<T>().deserialize_message(&serialized,&result);return result;}
template<class T>void save_message(const T& message,const std::filesystem::path& path){rclcpp::SerializedMessage serialized;rclcpp::Serialization<T>().serialize_message(&message,&serialized);const auto& data=serialized.get_rcl_serialized_message();std::ofstream output(path,std::ios::binary);output.write(reinterpret_cast<const char*>(data.buffer),data.buffer_length);need(output.good(),"Cannot save response");}
int main(int argc,char** argv){
 if(argc==4 && std::string(argv[1])=="--check-parameters"){
  try{
   const auto nodes=rclcpp::parameter_map_from_yaml_file(argv[2]);const auto& values=nodes.at("/m3d_fixture_planner");
   std::map<std::string,rclcpp::Parameter> parameters;for(const auto& value:values)parameters.emplace(value.get_name(),value);
   need(parameters.at("robot_description").as_string().find("<robot")!=std::string::npos,"URDF parameter is not XML text");
   need(parameters.at("robot_description_semantic").as_string().find("tip_link=\"panda_hand\"")!=std::string::npos,"SRDF hand tip missing");
   need(parameters.at("ompl.panda_arm.enforce_joint_model_state_space").as_bool(),"Group bool type differs");
   need(parameters.count("ompl.planner_configs.CarriedRRTConnect.enforce_joint_model_state_space")==0,"Planner bool override present");
   need(parameters.at("ompl.planning_plugins").as_string_array()==std::vector<std::string>{"sorting_control/CarriedMarginPlanner"},"Plugin array differs");
   need(!parameters.at("use_sim_time").as_bool() && !parameters.at("allow_trajectory_execution").as_bool(),"Component must use wall time and disable execution");
   need(parameters.at("ompl.totg.path_tolerance").as_double()==.0001,"Retiming type or value differs");
   std::ofstream report(argv[3]);report<<"{\"status\":\"PASS\",\"scope\":\"Installed rclcpp YAML parser only; no ROS node\",\"parameters\":"<<values.size()<<"}\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
 }

 if(argc!=4){std::cerr<<"Usage: verify_carried_pipeline PARAMETERS_YAML CASES_YAML OUTPUT_DIR\n";return 2;}
 YAML::Node report;report["status"]="FAIL";report["robot_commanded"]=false;report["scene_context"]="private_fixture_scenes";report["cases"]=YAML::Node(YAML::NodeType::Sequence);unsigned successful=0;
 const std::filesystem::path folder(argv[3]);std::filesystem::create_directories(folder);
 try{
  rclcpp::init(argc,argv);rclcpp::NodeOptions options;options.allow_undeclared_parameters(true).automatically_declare_parameters_from_overrides(true);
  options.arguments({"--ros-args","--params-file",argv[1]});auto node=std::make_shared<rclcpp::Node>("m3d_fixture_planner",options);
  need(!node->get_parameter("allow_trajectory_execution").as_bool(),"Fixture checker must disable execution");
  robot_model_loader::RobotModelLoader loader(node,"robot_description",true);auto model=loader.getModel();need(bool(model),"Robot model did not load");
  auto group=model->getJointModelGroup("panda_arm");need(group && group->getSolverInstance(),"Installed arm IK plugin did not load");
  planning_pipeline::PlanningPipeline pipeline(model,node,"ompl");
  const std::string prefix="default_planning_response_adapters/";
  auto adapters=pipeline.getResponseAdapterPluginNames();auto timing=std::find(adapters.begin(),adapters.end(),prefix+"AddTimeOptimalParameterization"),validation=std::find(adapters.begin(),adapters.end(),prefix+"ValidateSolution");
  need(timing!=adapters.end() && validation!=adapters.end() && timing<validation,"Final validation must follow retiming");
  auto request_adapters=pipeline.getRequestAdapterPluginNames();need(std::find(request_adapters.begin(),request_adapters.end(),"default_planning_request_adapters/CheckStartStateCollision")!=request_adapters.end(),"Start collision adapter is missing");
  need(node->get_parameter("ompl.planning_plugins").as_string_array()==std::vector<std::string>{"sorting_control/CarriedMarginPlanner"},"Carried margin planner is missing");
  need(node->get_parameter("ompl.carried_margin.position_m").as_double()==.0005 && node->get_parameter("ompl.carried_margin.orientation_rad").as_double()==.001,"Compiled margin mismatch");
  need(node->get_parameter("ompl.panda_arm.enforce_joint_model_state_space").as_bool(),"Joint-space selection is missing");
  need(node->get_parameter("ompl.planner_configs.CarriedRRTConnect.enforce_joint_model_state_space").get_type()==rclcpp::ParameterType::PARAMETER_NOT_SET,"Unsafe planner-level bool override");
  report["pipeline_configuration"]["status"]="PASS";report["pipeline_configuration"]["planning_plugins"]=node->get_parameter("ompl.planning_plugins").as_string_array();report["pipeline_configuration"]["request_adapters"]=request_adapters;report["pipeline_configuration"]["response_adapters"]=adapters;report["pipeline_configuration"]["position_margin_m"]=.0005;report["pipeline_configuration"]["orientation_margin_rad"]=.001;
  for(const auto& item:YAML::LoadFile(argv[2])["cases"]){
   const auto label=item["label"].as<std::string>();auto fixture=message<moveit_msgs::msg::PlanningScene>(item["scene"].as<std::string>());auto request=message<moveit_msgs::msg::MotionPlanRequest>(item["request"].as<std::string>());
   // Each case owns a new scene. No live PlanningSceneMonitor or scene topic is used.
   auto scene=std::make_shared<planning_scene::PlanningScene>(model);need(scene->setPlanningSceneDiffMsg(fixture),"Saved scene was rejected");
   auto start=scene->getCurrentState();moveit::core::robotStateMsgToRobotState(scene->getTransforms(),request.start_state,start);start.update();
   std::vector<const moveit::core::AttachedBody*> attached;start.getAttachedBodies(attached);need(attached.size()==8,"Saved scene lacks eight payload parts");
   for(const auto* part:attached)need(!scene->getWorld()->hasObject(part->getName()),"Payload is duplicated in the saved world");
   collision_detection::CollisionRequest check;check.group_name=request.group_name;check.contacts=true;check.max_contacts=100;collision_detection::CollisionResult collisions;scene->checkCollision(check,collisions,start);
   auto binding=report["scene_checks"][label];binding["world_objects"]=scene->getWorld()->getObjectIds();binding["attached_parts"]=attached.size();binding["start_collision_free"]=!collisions.collision;
   for(const auto& entry:collisions.contacts)binding["contacts"].push_back(entry.first.first+" - "+entry.first.second);
   need(!collisions.collision,"Saved requested start is in collision: "+label);need(scene->isStateValid(start,request.path_constraints,"panda_arm"),"Saved start violates its path bounds: "+label);
   need(scene->getWorld()->getObjectIds().size()>0,"Saved world is empty");
   for(int repeat=1;repeat<=10;++repeat){
    bool accepted=false;
    for(int attempt=1;attempt<=2;++attempt){
     planning_interface::MotionPlanResponse result;const bool generated=pipeline.generatePlan(scene,request,result,false);moveit_msgs::msg::MotionPlanResponse response;result.getMessage(response);
     YAML::Node record;record["label"]=label;record["repeat"]=repeat;record["attempt"]=attempt;record["code"]=response.error_code.val;record["message"]=response.error_code.message;record["source"]=response.error_code.source;record["points"]=response.trajectory.joint_trajectory.points.size();report["cases"].push_back(record);
     save_message(response,folder/(label+"-"+std::to_string(repeat)+"-"+std::to_string(attempt)+".cdr"));
     if(generated && response.error_code.val==1){
      need(result.trajectory && result.trajectory->getDuration()>0 && result.trajectory->getDuration()<=20.,"Trajectory duration exceeds the existing bound");
      need(scene->isPathValid(*result.trajectory,request.path_constraints,request.goal_constraints,"panda_arm"),"Final path violates original scene or bounds");accepted=true;break;
     }
     if(response.error_code.val!=-1 && response.error_code.val!=-2 && response.error_code.val!=-6 && response.error_code.val!=99999)break;
    }
    need(accepted,"Installed fixture pipeline failed: "+label);++successful;
   }
  }
  need(successful==40,"Incomplete forty-plan fixture coverage");report["status"]="PASS";
 }catch(const std::exception& error){report["error"]=error.what();std::cerr<<error.what()<<'\n';}
 report["successful_plans"]=successful;std::ofstream output(folder/"pipeline-report.yaml");output<<report;output.close();
 if(rclcpp::ok())rclcpp::shutdown();return report["status"].as<std::string>()=="PASS"?0:1;
}
