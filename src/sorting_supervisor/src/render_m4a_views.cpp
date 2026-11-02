// Node-free physical pose sweep. RGB class changes are evaluator counterfactuals only.
#include "sorting_simulation/native_camera_render.hpp"

int main(int argc,char** argv) {
  if(argc!=5) {std::cerr<<"Usage: render_m4a_views SCENE BASELINE CONFIG OUTPUT\n";return 2;}
  try {
    char error[2048]={};
    std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
    check(bool(m),error);
    std::unique_ptr<mjModel,decltype(&mj_deleteModel)> baseline(mj_loadXML(argv[2],nullptr,error,sizeof(error)),mj_deleteModel);
    check(bool(baseline),error);auto* b=baseline.get();
    check(m->nq==b->nq&&m->nv==b->nv&&m->nu==b->nu&&m->nbody==b->nbody&&m->ngeom==b->ngeom,"Physics dimensions changed");
    for(int i=0;i<m->nbody;++i) {
      check(m->body_mass[i]==b->body_mass[i],"Body mass changed");
      for(int j=0;j<3;++j)check(m->body_inertia[3*i+j]==b->body_inertia[3*i+j],"Body inertia changed");
    }
    for(int i=0;i<m->ngeom;++i)if(m->geom_contype[i]||m->geom_conaffinity[i]) {
      check(m->geom_contype[i]==b->geom_contype[i]&&m->geom_conaffinity[i]==b->geom_conaffinity[i],"Contact masks changed");
      for(int j=0;j<3;++j)check(m->geom_pos[3*i+j]==b->geom_pos[3*i+j]&&m->geom_size[3*i+j]==b->geom_size[3*i+j],"Physical geometry changed");
    }
    auto cfg=YAML::LoadFile(argv[3]);std::filesystem::path out(argv[4]);
    check(!std::filesystem::exists(out/"views.yaml"),"Use a new output directory");std::filesystem::create_directories(out);
    StaticMotion sim(m.get());sim.step(2.);auto tray0=sim.state().tray;
    YAML::Node report;report["physics_preserved"]=true;report["moving_conveyor"]=false;double last_sample=-1.;
    sim.guard=[&](const ContactState& s) {
      check(s.forbidden==0,"Forbidden physical contact during inspection");
      check(s.left_N+s.right_N<.1&&std::min(s.fingers[0],s.fingers[1])>.039,"Inspection must keep fingers open");
      check(s.support_N>.5&&distance3(s.tray,tray0)<.001,"Tray moved or lost support");
      for(int j=0;j<7;++j)check(std::abs(sim.data->qvel[j])<cfg["maximum_joint_speed_rad_s"].as<double>(),"Inspection joint speed exceeded bound");
      if(s.time-last_sample>=.05) {
        YAML::Node row;for(int j=0;j<7;++j)row.push_back(sim.data->qpos[j]);
        report["path_joint_positions"].push_back(row);last_sample=s.time;
      }
    };
    Offscreen render;render.init(m.get());int camera=mj_name2id(m.get(),mjOBJ_CAMERA,"wrist");
    int skin=mj_name2id(m.get(),mjOBJ_GEOM,"fruit_class_skin");check(camera>=0&&skin>=0,"Missing inspection asset");
    for(const char* name:{"assess","inspect_left","assess","inspect_right","assess"}) {
      auto values=cfg["poses"][name].as<std::vector<double>>();check(values.size()==3,"Invalid hand pose");
      std::array<double,3> target{{values[0],values[1],values[2]}};
      sim.solve(target);
      // MoveIt supplies kinematic joint positions. Do not silently add the
      // native helper's gravity bias: the live driver did not receive it.
      ArmVector command=Eigen::Map<ArmVector>(sim.kinematics->qpos);
      sim.move(command,cfg["native_motion_duration_s"].as<double>(),cfg["native_settle_s"].as<double>());
      YAML::Node correction;correction["pose"]=name;
      correction["planned_position_error_m"]=distance3(sim.state().hand,target);
      auto c=cfg["refinement"];double begin=sim.data->time,stable=-1.;
      Eigen::Matrix3d desired;desired<<0,1,0,1,0,0,0,0,-1;
      Eigen::Matrix3d entry_rotation=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(sim.data->xmat+9*sim.ids.hand);
      correction["planned_orientation_error_rad"]=Eigen::AngleAxisd(desired*entry_rotation.transpose()).angle();
      // ServoNode::pauseServo(false) resets its command seed to encoder state.
      // Include this transition; carrying over the biased command hides sag.
      command=Eigen::Map<ArmVector>(sim.data->qpos);
      while(true) {
        auto* d=sim.data;auto* k=sim.kinematics;
        Eigen::Matrix3d current=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(d->xmat+9*sim.ids.hand);
        Eigen::Vector3d error=Eigen::Map<const Eigen::Vector3d>(target.data())-Eigen::Map<Eigen::Vector3d>(d->xpos+3*sim.ids.hand);
        double angle=Eigen::AngleAxisd(desired*current.transpose()).angle();
        check(error.norm()<c["maximum_entry_position_error_m"].as<double>()&&angle<c["maximum_entry_orientation_error_rad"].as<double>(),"Native refinement left its correction region");
        if(error.norm()<c["position_tolerance_m"].as<double>()&&angle<c["orientation_tolerance_rad"].as<double>()) {
          if(stable<0)stable=d->time;
          if(d->time-stable>=c["stable_sim_s"].as<double>())break;
        }else stable=-1.;
        check(d->time-begin<c["maximum_sim_s"].as<double>(),"Native feedback refinement timed out");
        Eigen::Vector3d linear=c["position_gain_per_s"].as<double>()*error,angular=Eigen::Vector3d::Zero();
        for(int j=0;j<3;++j)angular+=.5*current.col(j).cross(desired.col(j));
        angular*=c["orientation_gain_per_s"].as<double>();
        double linear_limit=c["maximum_linear_speed_m_s"].as<double>(),angular_limit=c["maximum_angular_speed_rad_s"].as<double>();
        if(linear.norm()>linear_limit)linear*=linear_limit/linear.norm();
        if(angular.norm()>angular_limit)angular*=angular_limit/angular.norm();
        // Like the native M3D plant, integrate the last command while closing
        // the Cartesian loop with measured FK. This is not MoveIt Servo itself.
        mj_copyData(k,m.get(),d);for(int j=0;j<7;++j)k->qpos[j]=command[j];mj_forward(m.get(),k);
        std::vector<mjtNum> jp(3*m->nv),jr(3*m->nv);mj_jacBody(m.get(),k,jp.data(),jr.data(),sim.ids.hand);
        Eigen::Matrix<double,6,7> jac;
        for(int r=0;r<3;++r)for(int j=0;j<7;++j){jac(r,j)=jp[r*m->nv+j];jac(r+3,j)=jr[r*m->nv+j];}
        auto singular=jac.jacobiSvd().singularValues();double condition=singular[0]/singular[5];
        check(condition<30.,"Native refinement singularity bound");
        double scale=condition>17.?(30.-condition)/13.:1.;
        Eigen::Matrix<double,6,1> twist;twist<<scale*linear,scale*angular;
        ArmVector velocity=jac.transpose()*(jac*jac.transpose()+.0001*Eigen::Matrix<double,6,6>::Identity()).ldlt().solve(twist);
        if(velocity.cwiseAbs().maxCoeff()>.1)velocity*=.1/velocity.cwiseAbs().maxCoeff();
        command+=.01*velocity;
        for(int j=0;j<7;++j){check(command[j]>=m->actuator_ctrlrange[2*j]&&command[j]<=m->actuator_ctrlrange[2*j+1],"Native refinement command limit");d->ctrl[j]=command[j];}
        sim.step(.01);
      }
      correction["refinement_sim_s"]=sim.data->time-begin;
      sim.step(cfg["native_settle_s"].as<double>());
      correction["held_position_error_m"]=distance3(sim.state().hand,target);
      Eigen::Matrix3d held_rotation=Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(sim.data->xmat+9*sim.ids.hand);
      double held_angle=Eigen::AngleAxisd(desired*held_rotation.transpose()).angle();
      correction["held_orientation_error_rad"]=held_angle;
      report["refinements"].push_back(correction);
      check(distance3(sim.state().hand,target)<cfg["maximum_position_error_m"].as<double>(),"Physical hand endpoint error");
      check(held_angle<cfg["maximum_orientation_error_rad"].as<double>(),"Physical hand orientation error");
      if(report["views"][name])continue;
      YAML::Node row;for(int j=0;j<7;++j)row["joints"].push_back(sim.data->qpos[j]);
      for(int j=0;j<3;++j)row["camera_xyz"].push_back(sim.data->cam_xpos[3*camera+j]);
      for(int j=0;j<9;++j)row["camera_optical_rotation"].push_back(sim.data->cam_xmat[9*camera+j]*(j%3==0?1:-1));
      int handle=mj_name2id(m.get(),mjOBJ_SITE,"handle_grasp_site");
      for(int j=0;j<3;++j)row["evaluator_handle_xyz"].push_back(sim.data->site_xpos[3*handle+j]);
      row["hand_error_m"]=distance3(sim.state().hand,target);row["fovy_deg"]=m->cam_fovy[camera];
      row["width"]=m->cam_resolution[2*camera];row["height"]=m->cam_resolution[2*camera+1];
      for(int kind=0;kind<2;++kind) {
        for(int c=0;c<3;++c)m->geom_rgba[4*skin+c]=c==kind?.8:.08;
        render.save(m.get(),sim.data,camera,(out/(std::string(name)+(kind==0?"-red.ppm":"-green.ppm"))).string());
      }
      report["views"][name]=row;
    }
    report["benchmark"]=render.benchmark(m.get(),sim.data);
    report["status"]="PASS";std::ofstream file(out/"views.yaml");file<<report;
    std::cout<<"PASS: physical inspection pose sweep and return\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
