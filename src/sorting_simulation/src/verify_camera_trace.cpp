/**
 * Copyright (c) 2026, United States Government, as represented by the
 * Administrator of the National Aeronautics and Space Administration.
 *
 * All rights reserved.
 *
 * This software is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

// Modified for project camera timing verification.
// Exercise the actual render/format helper, snapshot isolation and plugin factory. No ROS node.
#include "traced_camera_plugin.hpp"
#include "sorting_simulation/native_camera_render.hpp"
#include <pluginlib/class_loader.hpp>
#include <thread>
#include <iomanip>
using namespace sorting_simulation;
static CameraData camera_buffer(const mjModel* m,const char* name){
  CameraData c;mjv_defaultCamera(&c.mjv_cam);int id=mj_name2id(m,mjOBJ_CAMERA,name);check(id>=0,"Missing camera");
  c.name=name;c.width=m->cam_resolution[2*id];c.height=m->cam_resolution[2*id+1];
  c.mjv_cam.type=mjCAMERA_FIXED;c.mjv_cam.fixedcamid=id;c.viewport={0,0,static_cast<int>(c.width),static_cast<int>(c.height)};
  c.image_buffer.resize(c.width*c.height*3);c.image.data.resize(c.image_buffer.size());
  c.depth_buffer.resize(c.width*c.height);c.depth_image.data.resize(c.width*c.height*sizeof(float));return c;
}
static void baseline(const mjModel* m,mjData* d,Offscreen& gl,CameraData& c,float near,float scale){
  mjv_updateScene(m,d,&gl.options,nullptr,&c.mjv_cam,mjCAT_ALL,&gl.scene);mjr_render(c.viewport,&gl.scene,&gl.render);
  mjr_readPixels(c.image_buffer.data(),c.depth_buffer.data(),c.viewport,&gl.render);
  auto* dst=reinterpret_cast<float*>(c.depth_image.data.data());
  for(uint32_t h=0;h<c.height;++h){
    const float* src=c.depth_buffer.data()+static_cast<size_t>(h)*c.width;
    float* row=dst+static_cast<size_t>(c.height-1-h)*c.width;
    for(uint32_t w=0;w<c.width;++w)row[w]=near/(1.0f-src[w]*scale);
  }
  const auto row_size=c.width*3;
  for(uint32_t h=0;h<c.height;++h)
    std::memcpy(&c.image.data[(c.height-1-h)*row_size],&c.image_buffer[h*row_size],row_size);
}
int main(int argc,char** argv){try{
  check(argc==3,"Usage: verify_camera_trace scene.xml output.json");
  char error[1024]{};
  auto m=std::unique_ptr<mjModel,decltype(&mj_deleteModel)>(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
  check(bool(m),error);auto pending=std::unique_ptr<mjData,decltype(&mj_deleteData)>(mj_makeData(m.get()),mj_deleteData);
  auto private_data=std::unique_ptr<mjData,decltype(&mj_deleteData)>(mj_makeData(m.get()),mj_deleteData);
  mj_resetDataKeyframe(m.get(),pending.get(),0);mj_forward(m.get(),pending.get());
  copy_camera_snapshot(private_data.get(),m.get(),pending.get());
  std::string worker_error;
  std::thread context_worker([&]{try{
    Offscreen worker_gl;worker_gl.init(m.get());
    auto buffer=camera_buffer(m.get(),"overhead");
    mjr_resizeOffscreen(buffer.width,buffer.height,&worker_gl.render);mjr_setBuffer(mjFB_OFFSCREEN,&worker_gl.render);
    float near=float(m->vis.map.znear*m->stat.extent),far=float(m->vis.map.zfar*m->stat.extent);
    CameraFrameTrace trace;
    render_camera_frame(m.get(),private_data.get(),worker_gl.options,worker_gl.scene,worker_gl.render,buffer,near,1.f-near/far,trace);
  }catch(const std::exception& error){worker_error=error.what();}});
  context_worker.join();check(worker_error.empty(),"Worker EGL context failed: "+worker_error);
  Offscreen gl;gl.init(m.get());const auto backend=render_backend_info();int maxw=1,maxh=1;
  for(const char* name:{"overhead","wrist"}){auto c=camera_buffer(m.get(),name);maxw=std::max(maxw,int(c.width));maxh=std::max(maxh,int(c.height));}
  mjr_resizeOffscreen(maxw,maxh,&gl.render);mjr_setBuffer(mjFB_OFFSCREEN,&gl.render);
  float near=float(m->vis.map.znear*m->stat.extent),far=float(m->vis.map.zfar*m->stat.extent),scale=1.f-near/far;
  int parity=0;CameraFrameTrace last;
  for(int pose=0;pose<4;++pose){
    pending->time=2.+pose*.04;pending->qpos[0]+=.01; mj_forward(m.get(),pending.get());
    copy_camera_snapshot(private_data.get(),m.get(),pending.get());double frozen_time=private_data->time;
    for(const char* name:{"overhead","wrist"}){
      auto expected=camera_buffer(m.get(),name),traced=camera_buffer(m.get(),name);
      baseline(m.get(),private_data.get(),gl,expected,near,scale);
      render_camera_frame(m.get(),private_data.get(),gl.options,gl.scene,gl.render,traced,near,scale,last);
      check(expected.image.data==traced.image.data,"RGB payload changed");
      check(expected.depth_image.data==traced.depth_image.data,"Depth payload changed");++parity;
      check(private_data->time==frozen_time,"Snapshot timestamp changed during rendering");
      check(last.camera_render_begin_wall_ns<=last.scene_ready_wall_ns && last.scene_ready_wall_ns<=last.draw_return_wall_ns &&
            last.draw_return_wall_ns<=last.pixels_ready_wall_ns && last.pixels_ready_wall_ns<=last.format_ready_wall_ns,"Render trace order is invalid");
    }
  }
  // A later control snapshot must not alter the data that the worker is rendering.
  double frozen=private_data->time;std::vector<mjtNum> positions(private_data->geom_xpos,private_data->geom_xpos+3*m->ngeom);
  std::thread writer([&]{for(int i=0;i<100;++i){pending->time+=.002;pending->qpos[0]+=.0001;mj_forward(m.get(),pending.get());}});
  auto c=camera_buffer(m.get(),"overhead");render_camera_frame(m.get(),private_data.get(),gl.options,gl.scene,gl.render,c,near,scale,last);writer.join();
  check(private_data->time==frozen && std::equal(positions.begin(),positions.end(),private_data->geom_xpos),"Worker snapshot changed after pending overwrite");
  pluginlib::ClassLoader<mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase> loader("mujoco_ros2_control_plugins","mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase");
  auto instance=loader.createSharedInstance("sorting_simulation/TracedCameraPlugin");instance->on_reset(nullptr);instance->cleanup();instance.reset();
  last.capture_ns=rclcpp::Duration::from_seconds(frozen).nanoseconds();
  std::ofstream out(argv[2]);check(bool(out),"Cannot write camera trace check");
  out<<"{\"status\":\"PASS\",\"live_ROS\":\"NOT_RUN\",\"parity_frames\":"<<parity
     <<",\"snapshot_isolation\":true,\"plugin_factory\":true,\"worker_context\":true,\"gl_vendor\":"<<std::quoted(backend.vendor)
     <<",\"gl_renderer\":"<<std::quoted(backend.renderer)<<",\"render_backend_request\":"<<std::quoted(render_backend_request())
     <<",\"steady_clock_sample_ns\":"<<camera_wall_ns()
     <<",\"system_clock_sample_ns\":"<<camera_system_ns()<<",\"render_trace\":"<<last.json()<<"}\n";
  std::cout<<"PASS: camera RGB/depth parity, snapshot isolation and plugin factory\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
