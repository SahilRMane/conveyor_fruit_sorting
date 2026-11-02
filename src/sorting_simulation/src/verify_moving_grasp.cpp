// Physical feasibility check. Direct bounded native kinematics, not live Servo.
#include "sorting_simulation/static_motion.hpp"
#include "sorting_simulation/conveyor_transport.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
using namespace sorting_simulation;
using V3=Eigen::Vector3d;
int main(int argc,char** argv){
  if(argc!=4){std::cerr<<"Usage: verify_moving_grasp SCENE CONFIG OUTPUT_JSON\n";return 2;}
  std::ofstream out(argv[3]);if(!out)return 2;
  char error[2048]={};auto* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));if(!m){std::cerr<<error;return 1;}
  try{
    auto cfg=YAML::LoadFile(argv[2]);StaticMotion sim(m);ConveyorTransport conveyor(m);
    sim.step(2.0);int handle=mj_name2id(m,mjOBJ_SITE,"handle_grasp_site");
    std::array<double,3> pre{};for(int i=0;i<3;++i)pre[i]=sim.data->site_xpos[3*handle+i];pre[2]+=0.1629;
    sim.move(sim.solve(pre),2.0,0.5);
    Eigen::Matrix3d desired;desired<<0,1,0,1,0,0,0,0,-1;
    ArmVector commanded=Eigen::Map<ArmVector>(sim.data->ctrl);
    double speed=cfg["belt_speed_m_s"].as<double>(),dt=0.01;
    double maximum_slip=0,maximum_world_offset_change=0,maximum_joint_speed=0,maximum_condition=0;
    double grip_z=0,grip_hand_z=0,grip_x=0;
    V3 grip_relative=V3::Zero(),grip_world_relative=V3::Zero();
    Eigen::Quaterniond grip_tray_rotation=Eigen::Quaterniond::Identity();
    bool gripped=false;std::string phase="track";
    auto relative_position=[](const ContactState& s)->V3{
      Eigen::Quaterniond rotation(s.hand_quat[0],s.hand_quat[1],s.hand_quat[2],s.hand_quat[3]);
      return rotation.normalized().toRotationMatrix().transpose()*
             (Eigen::Map<const V3>(s.tray.data())-Eigen::Map<const V3>(s.hand.data()));
    };
    auto monitor=[&](){
      auto s=sim.state();
      check(s.forbidden==0,"Forbidden physical contact: "+state_json(s));
      check(s.left_N<cfg["maximum_contact_force_N"].as<double>() && s.right_N<cfg["maximum_contact_force_N"].as<double>(),"Finger force bound");
      for(int i=0;i<7;++i){
        maximum_joint_speed=std::max(maximum_joint_speed,std::fabs(sim.data->qvel[i]));
        check(std::fabs(sim.data->qvel[i])<cfg["maximum_joint_speed_rad_s"].as<double>(),"Measured joint speed bound in "+phase);
      }
      if(gripped){
        check(s.left_N>cfg["minimum_contact_force_N"].as<double>() && s.right_N>cfg["minimum_contact_force_N"].as<double>(),"Handle contact lost in "+phase);
        double slip=(relative_position(s)-grip_relative).norm();
        maximum_slip=std::max(maximum_slip,slip);
        double world_change=(Eigen::Map<const V3>(s.tray.data())-Eigen::Map<const V3>(s.hand.data())-grip_world_relative).norm();
        maximum_world_offset_change=std::max(maximum_world_offset_change,world_change);
        check(slip<cfg["maximum_slip_m"].as<double>(),"Hand-frame relative slip bound in "+phase);
        Eigen::Quaterniond rotation(s.tray_quat[0],s.tray_quat[1],s.tray_quat[2],s.tray_quat[3]);
        check(std::fabs(rotation.normalized().dot(grip_tray_rotation))>std::cos(cfg["maximum_tilt_rad"].as<double>()/2),"Tray tilt bound in "+phase);
        double rise=s.tray[2]-grip_z;
        check(rise>=-0.003 && rise<cfg["maximum_lift_rise_m"].as<double>(),"Pickup height bound in "+phase);
        if(rise>0.005){check(s.support_N<0.1,"Lifted support force");check(std::fabs(s.external_force_x_N)<1e-8,"Transport still active after lift");}
        if(phase=="final_hold")check(rise>0.03,"Final HOLD retained height bound");
      }
    };
    auto control=[&](V3 target,V3 ff){
      mj_forward(m,sim.data);mjtNum hand_v[6];mj_objectVelocity(m,sim.data,mjOBJ_BODY,sim.ids.hand,hand_v,0);
      V3 p=Eigen::Map<V3>(sim.data->xpos+3*sim.ids.hand);
      Eigen::Matrix3d current=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(sim.data->xmat+9*sim.ids.hand);
      V3 angular=V3::Zero();for(int i=0;i<3;++i)angular+=0.5*current.col(i).cross(desired.col(i));angular*=cfg["orientation_gain_per_s"].as<double>();
      V3 linear=ff+cfg["position_gain_per_s"].as<double>()*(target-p);
      double linear_limit=cfg["maximum_servo_linear_speed_m_s"].as<double>(),angular_limit=cfg["maximum_servo_angular_speed_rad_s"].as<double>();
      if(linear.norm()>linear_limit){linear*=linear_limit/linear.norm();}
      if(angular.norm()>angular_limit){angular*=angular_limit/angular.norm();}
      mj_copyData(sim.kinematics,m,sim.data);for(int i=0;i<7;++i)sim.kinematics->qpos[i]=commanded[i];mj_forward(m,sim.kinematics);
      std::vector<mjtNum> jp(3*m->nv),jr(3*m->nv);mj_jacBody(m,sim.kinematics,jp.data(),jr.data(),sim.ids.hand);
      Eigen::Matrix<double,6,7> j;for(int r=0;r<3;++r)for(int c=0;c<7;++c){j(r,c)=jp[r*m->nv+c];j(r+3,c)=jr[r*m->nv+c];}
      auto singular=j.jacobiSvd().singularValues();double condition=singular[0]/singular[5];maximum_condition=std::max(maximum_condition,condition);
      check(condition<30,"Native corridor reaches Servo hard singularity threshold in "+phase);
      double scale=condition>17 ? (30-condition)/13.0 : 1.0;
      Eigen::Matrix<double,6,1> twist;twist<<scale*linear,scale*angular;
      ArmVector velocity=j.transpose()*(j*j.transpose()+0.0001*Eigen::Matrix<double,6,6>::Identity()).ldlt().solve(twist);
      for(int i=0;i<7;++i){check(std::fabs(velocity[i])<cfg["maximum_joint_speed_rad_s"].as<double>(),"Native joint speed outside 0.5 rad/s bound in "+phase+" joint "+std::to_string(i)+" condition "+std::to_string(condition));commanded[i]+=velocity[i]*dt;sim.data->ctrl[i]=commanded[i];}
      for(int step=0;step<5;++step){conveyor.apply(sim.data,speed,true);sim.step(m->opt.timestep);monitor();}
    };
    double begin=sim.data->time;bool close_ready=false;double stable_start=-1,closure_relative_speed=0,closure_error=0;
    for(int tick=0;tick<800 && !close_ready;++tick){
      double elapsed=sim.data->time-begin;double descend=cfg["descend_duration_s"].as<double>();double u=std::clamp(elapsed/descend,0.0,1.0),blend=u*u*(3-2*u);
      V3 target=Eigen::Map<V3>(sim.data->site_xpos+3*handle);target.z()+=0.1029+0.06*(1-blend);
      V3 ff(speed,0,-0.06*6*u*(1-u)/descend);control(target,ff);
      auto s=sim.state();mjtNum v[6];mj_objectVelocity(m,sim.data,mjOBJ_BODY,sim.ids.hand,v,0);
      V3 reference=Eigen::Map<V3>(sim.data->site_xpos+3*handle);reference.z()+=0.1029;
      V3 e=Eigen::Map<V3>(s.hand.data())-reference;closure_relative_speed=(Eigen::Map<V3>(v+3)-Eigen::Map<V3>(s.tray_linear_velocity.data())).norm();closure_error=e.norm();
      bool good=elapsed>=descend && std::hypot(e.x(),e.y())<cfg["close_position_xy_m"].as<double>() && std::fabs(e.z())<cfg["close_position_z_m"].as<double>() && closure_relative_speed<cfg["close_relative_speed_m_s"].as<double>();
      if(good){if(stable_start<0)stable_start=s.time;close_ready=s.time-stable_start>=cfg["contact_stable_span_s"].as<double>();}else stable_start=-1;
    }
    check(close_ready,"Closure readiness did not stabilize");std::cout<<"close ready x="<<sim.state().tray[0]<<" relv="<<closure_relative_speed<<" error="<<closure_error<<'\n';
    phase="grip";sim.data->ctrl[7]=0;begin=sim.data->time;stable_start=-1;
    for(int tick=0;tick<250 && !gripped;++tick){V3 target=Eigen::Map<V3>(sim.data->site_xpos+3*handle);target.z()+=0.1029;control(target,V3(speed,0,0));
      auto s=sim.state();if(s.left_N>0.2 && s.right_N>0.2){if(stable_start<0)stable_start=s.time;if(s.time-stable_start>=cfg["contact_stable_span_s"].as<double>() && s.time-begin>=0.2)gripped=true;}else stable_start=-1;
    }
    check(gripped,"Both fingers did not establish contact");auto grasp=sim.state();grip_z=grasp.tray[2];grip_hand_z=grasp.hand[2];grip_x=grasp.hand[0];
    grip_relative=relative_position(grasp);
    grip_world_relative=Eigen::Map<const V3>(grasp.tray.data())-Eigen::Map<const V3>(grasp.hand.data());
    grip_tray_rotation=Eigen::Quaterniond(grasp.tray_quat[0],grasp.tray_quat[1],grasp.tray_quat[2],grasp.tray_quat[3]).normalized();
    std::cout<<"grip x="<<grasp.tray[0]<<" forces="<<grasp.left_N<<','<<grasp.right_N<<'\n';
    phase="lift";begin=sim.data->time;double brake_start=-1,brake_x=0;
    for(int tick=0;tick<360;++tick){double t=sim.data->time-begin,u=std::clamp(t/3.0,0.0,1.0),blend=u*u*(3-2*u);V3 target(grip_x+speed*t,grasp.hand[1],grip_hand_z+0.05*blend),ff(speed,0,0.05*6*u*(1-u)/3.0);
      if(sim.state().tray[2]-grip_z>0.005 && brake_start<0){brake_start=t;brake_x=target.x();}
      if(brake_start>=0){double b=std::clamp((t-brake_start)/0.5,0.0,1.0);target.x()=brake_x+speed*0.5*(b-b*b+b*b*b/3);ff.x()=speed*(1-b)*(1-b);}
      control(target,ff);
    }
    auto held=sim.state();check(held.tray[2]-grip_z>cfg["minimum_lift_rise_m"].as<double>() && held.support_N<0.1,"Tray was not lifted 45 mm");
    phase="hold";V3 fixed=Eigen::Map<V3>(held.hand.data());for(int tick=0;tick<60;++tick)control(fixed,V3::Zero());
    // End this pickup trial after lift/hold. A paused episode reset follows
    // in the ROS runner; this test makes no moving placement claim.
    phase="final_hold";
    // This direct native hold is a feasibility test. It does not emulate the
    // live controller's trajectory interpolation or the gate's service calls.
    for(int i=0;i<7;++i)sim.data->ctrl[i]=sim.data->qpos[i];
    for(int tick=0;tick<300;++tick){conveyor.apply(sim.data,speed,true);sim.step(m->opt.timestep);monitor();}
    auto released=sim.state();check(released.left_N>0.2 && released.right_N>0.2 && released.support_N<0.1,
                                  "Payload was not retained in final measured-state hold");
    out<<"{\"status\":\"PASS\",\"mode\":\"native physical feasibility; live Servo NOT_RUN\",\"closure_relative_speed_m_s\":"<<closure_relative_speed
       <<",\"closure_position_error_m\":"<<closure_error<<",\"maximum_relative_slip_m\":"<<maximum_slip<<",\"maximum_joint_speed_rad_s\":"<<maximum_joint_speed
       <<",\"slip_frame\":\"panda_hand\",\"slip_metric_version\":2,\"maximum_world_offset_change_m\":"<<maximum_world_offset_change
       <<",\"final_hold_monitored\":true,\"final_held_rise_m\":"<<released.tray[2]-grip_z
       <<",\"maximum_jacobian_condition\":"<<maximum_condition<<",\"held_rise_m\":"<<held.tray[2]-grip_z<<",\"held_transport_force_N\":"<<held.external_force_x_N<<",\"final_held_tray_x_m\":"<<released.tray[0]<<"}\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';out<<"{\"status\":\"FAIL\"}\n";mj_deleteModel(m);return 1;}
  mj_deleteModel(m);return 0;
}
