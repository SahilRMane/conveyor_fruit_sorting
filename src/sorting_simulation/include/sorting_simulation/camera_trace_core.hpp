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
#pragma once
// Timing and RGB-D operations for the project-owned 0.1.2 camera trace.
#include <mujoco/mujoco.h>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdint>
#include <sstream>
#include <string>
namespace sorting_simulation {
inline int64_t camera_wall_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
inline int64_t camera_system_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
inline int64_t camera_thread_cpu_ns(){timespec t{};return clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0?t.tv_sec*1000000000LL+t.tv_nsec:0;}
struct CameraFrameTrace {
  int64_t update_enter_wall_ns{};
  int64_t rate_clock_sim_ns{};
  int64_t source_data_ns{};
  int64_t snapshot_lock_acquired_wall_ns{};
  int64_t snapshot_accept_wall_ns{};
  int64_t snapshot_accept_system_ns{};
  int64_t snapshot_copy_end_wall_ns{};
  int64_t worker_take_wall_ns{};
  int64_t worker_copy_begin_wall_ns{};
  int64_t worker_copy_end_wall_ns{};
  int64_t render_pass_begin_wall_ns{};
  int64_t camera_render_begin_wall_ns{};
  int64_t scene_ready_wall_ns{};
  int64_t draw_return_wall_ns{};
  int64_t pixels_ready_wall_ns{};
  int64_t format_ready_wall_ns{};
  int64_t image_publish_enter_wall_ns{};
  int64_t image_publish_enter_system_ns{};
  int64_t image_publish_ros_ns{};
  int64_t image_publish_return_wall_ns{};
  int64_t metadata_publish_return_wall_ns{};
  int64_t render_data_begin_ns{};
  int64_t render_data_end_ns{};
  int64_t render_thread_cpu_begin_ns{};
  int64_t render_thread_cpu_end_ns{};
  int64_t capture_ns{};
  int64_t episode_id{};
  int64_t snapshot_serial{};
  int64_t pending_replacements{};
  int64_t selected_count{};
  int64_t camera_order{};
  bool use_egl{};
  std::string json() const {
    std::ostringstream out;
    out<<"{\"stage\":\"camera\",\"version\":1,\"view_id\":\"overhead\",\"snapshot_policy\":\"worker_private_copy\"";
    out<<",\"update_enter_wall_ns\":"<<update_enter_wall_ns;
    out<<",\"rate_clock_sim_ns\":"<<rate_clock_sim_ns;
    out<<",\"source_data_ns\":"<<source_data_ns;
    out<<",\"snapshot_lock_acquired_wall_ns\":"<<snapshot_lock_acquired_wall_ns;
    out<<",\"snapshot_accept_wall_ns\":"<<snapshot_accept_wall_ns;
    out<<",\"snapshot_accept_system_ns\":"<<snapshot_accept_system_ns;
    out<<",\"snapshot_copy_end_wall_ns\":"<<snapshot_copy_end_wall_ns;
    out<<",\"worker_take_wall_ns\":"<<worker_take_wall_ns;
    out<<",\"worker_copy_begin_wall_ns\":"<<worker_copy_begin_wall_ns;
    out<<",\"worker_copy_end_wall_ns\":"<<worker_copy_end_wall_ns;
    out<<",\"render_pass_begin_wall_ns\":"<<render_pass_begin_wall_ns;
    out<<",\"camera_render_begin_wall_ns\":"<<camera_render_begin_wall_ns;
    out<<",\"scene_ready_wall_ns\":"<<scene_ready_wall_ns;
    out<<",\"draw_return_wall_ns\":"<<draw_return_wall_ns;
    out<<",\"pixels_ready_wall_ns\":"<<pixels_ready_wall_ns;
    out<<",\"format_ready_wall_ns\":"<<format_ready_wall_ns;
    out<<",\"image_publish_enter_wall_ns\":"<<image_publish_enter_wall_ns;
    out<<",\"image_publish_enter_system_ns\":"<<image_publish_enter_system_ns;
    out<<",\"image_publish_ros_ns\":"<<image_publish_ros_ns;
    out<<",\"image_publish_return_wall_ns\":"<<image_publish_return_wall_ns;
    out<<",\"metadata_publish_return_wall_ns\":"<<metadata_publish_return_wall_ns;
    out<<",\"render_data_begin_ns\":"<<render_data_begin_ns;
    out<<",\"render_data_end_ns\":"<<render_data_end_ns;
    out<<",\"render_thread_cpu_begin_ns\":"<<render_thread_cpu_begin_ns;
    out<<",\"render_thread_cpu_end_ns\":"<<render_thread_cpu_end_ns;
    out<<",\"capture_ns\":"<<capture_ns;
    out<<",\"episode_id\":"<<episode_id;
    out<<",\"snapshot_serial\":"<<snapshot_serial;
    out<<",\"pending_replacements\":"<<pending_replacements;
    out<<",\"selected_count\":"<<selected_count;
    out<<",\"camera_order\":"<<camera_order;
    out<<",\"use_egl\":"<<(use_egl?"true":"false")<<"}";
    return out.str();
  }
};
inline void copy_camera_snapshot(mjData* destination,const mjModel* model,const mjData* source){mjv_copyData(destination,model,source);}

// The render/read/format operations below retain the upstream 0.1.2 algorithm.
template<class Camera>
void render_camera_frame(const mjModel* model,mjData* data,mjvOption& option,mjvScene& scene,mjrContext& context,
                         Camera& camera,float near_distance,float depth_scale,CameraFrameTrace& trace){
  trace.camera_render_begin_wall_ns=camera_wall_ns();trace.render_thread_cpu_begin_ns=camera_thread_cpu_ns();
  mjv_updateScene(model,data,&option,nullptr,&camera.mjv_cam,mjCAT_ALL,&scene);
  trace.scene_ready_wall_ns=camera_wall_ns();
  mjr_render(camera.viewport,&scene,&context);trace.draw_return_wall_ns=camera_wall_ns();
  mjr_readPixels(camera.image_buffer.data(),camera.depth_buffer.data(),camera.viewport,&context);
  trace.pixels_ready_wall_ns=camera_wall_ns();
  auto* depth_out=reinterpret_cast<float*>(camera.depth_image.data.data());
  for(uint32_t h=0;h<camera.height;++h){
    const float* src=camera.depth_buffer.data()+static_cast<size_t>(h)*camera.width;
    float* dst=depth_out+static_cast<size_t>(camera.height-1-h)*camera.width;
    for(uint32_t w=0;w<camera.width;++w)dst[w]=near_distance/(1.0f-src[w]*depth_scale);
  }
  const auto row_size=camera.width*3;
  for(uint32_t h=0;h<camera.height;++h)
    std::memcpy(&camera.image.data[(camera.height-1-h)*row_size],&camera.image_buffer[h*row_size],row_size);
  trace.format_ready_wall_ns=camera_wall_ns();trace.render_thread_cpu_end_ns=camera_thread_cpu_ns();
}
} // namespace sorting_simulation
