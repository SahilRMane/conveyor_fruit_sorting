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

#include "traced_camera_plugin.hpp"

namespace sorting_simulation
{

bool TracedCameraPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data)
{
  node_ = node;
  mj_model_ = model;
  mj_data_ = data;
  trace_pub_=node_->create_publisher<std_msgs::msg::String>("/sorting/m4c/camera_side_timing",rclcpp::QoS(64).best_effort());

  // Ensure the logger has a name
  logger_ = node_->get_logger().get_child(node->get_sub_namespace());
  RCLCPP_INFO(node_->get_logger(), "TracedCameraPlugin initializing cameras...");

  // Read the mj_model_, identify the number of cameras, and populate containers for them.
  if (!register_cameras())
  {
    RCLCPP_ERROR(node_->get_logger(), "Failed to register cameras.");
    return false;
  }
  if (cameras_.empty())
  {
    return true;
  }

  // The camera worker owns an EGL context and makes no GLFW/X11 calls.
  rendering_thread_ = std::thread(&TracedCameraPlugin::update_loop, this);
  return true;
}

void TracedCameraPlugin::update(const mjModel* model_arg, mjData* data)
{
  if (!publish_images_)
  {
    return;
  }

  if(reset_pending_.exchange(false))last_publish_time_=rclcpp::Time(0,0,RCL_ROS_TIME);
  const auto update_enter=camera_wall_ns();
  // Streaming cameras render on a fixed-rate clock; polled cameras render as soon as a
  // trigger has been received. This control-thread hook receives a recent physics snapshot.
  // Its data timestamp can precede the current ROS clock.
  // TODO: Support per-camera publish rates?
  const auto now = node_->get_clock()->now();
  const bool stream_due =
      has_streaming_cameras_ && (now - last_publish_time_).seconds() >= (1.0 / camera_publish_rate_);
  const bool poll_due = poll_pending_.exchange(false);

  // Nothing to do this step: avoid taking the lock or copying data.
  if (!stream_due && !poll_due)
  {
    return;
  }

  bool any_selected = false;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    const auto lock_acquired=camera_wall_ns();

    // Flag streaming cameras when their interval is due and any polled cameras that have a
    // pending trigger (consuming the one-shot request so they render exactly once).
    for (auto& camera : cameras_)
    {
      if (camera.policy == CameraPolicy::STREAMING && stream_due)
      {
        camera.render_pending = true;
        any_selected = true;
      }
      else if (camera.policy == CameraPolicy::POLLED && camera.poll_requested)
      {
        camera.poll_requested = false;
        camera.render_pending = true;
        any_selected = true;
      }
    }

    if (stream_due)
    {
      last_publish_time_ = now;
    }

    // Only snapshot the simulation data when there is actually a camera to render.
    if (any_selected)
    {
      if(new_data_)++pending_replacements_;
      pending_trace_=CameraFrameTrace{};
      pending_trace_.episode_id=episode_.load();pending_trace_.snapshot_serial=++snapshot_serial_;
      pending_trace_.pending_replacements=pending_replacements_;
      pending_trace_.update_enter_wall_ns=update_enter;
      pending_trace_.rate_clock_sim_ns=now.nanoseconds();
      pending_trace_.source_data_ns=rclcpp::Duration::from_seconds(data->time).nanoseconds();
      pending_trace_.snapshot_lock_acquired_wall_ns=lock_acquired;
      pending_trace_.snapshot_accept_wall_ns=camera_wall_ns();pending_trace_.snapshot_accept_system_ns=camera_system_ns();
      copy_camera_snapshot(mj_camera_data_,model_arg,data);
      pending_trace_.snapshot_copy_end_wall_ns=camera_wall_ns();
      new_data_ = true;
    }
  }

  if (any_selected)
  {
    data_cv_.notify_one();
  }
}

void TracedCameraPlugin::cleanup()
{
  close();
}

void TracedCameraPlugin::on_reset(mjData*)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  episode_.fetch_add(1);new_data_=false;reset_pending_=true;
  for(auto& camera:cameras_)camera.render_pending=false;
}

bool TracedCameraPlugin::register_cameras()
{
  const std::string param_prefix = "mujoco_plugins." + node_->get_sub_namespace() + ".";
  const std::string camera_publish_rate_param = param_prefix + "camera_publish_rate";
  if (!node_->has_parameter(camera_publish_rate_param))
  {
    node_->declare_parameter(camera_publish_rate_param, 5.0);
  }

  camera_publish_rate_ = node_->get_parameter(camera_publish_rate_param).as_double();
  RCLCPP_INFO(node_->get_logger(), "Publishing camera data at rate %f per second.", camera_publish_rate_);

  cameras_.resize(0);
  for (auto i = 0; i < mj_model_->ncam; ++i)
  {
    const char* cam_name = mj_model_->names + mj_model_->name_camadr[i];
    const int* cam_resolution = mj_model_->cam_resolution + 2 * i;
    const mjtNum cam_fovy = mj_model_->cam_fovy[i];

    // Construct CameraData wrapper and set defaults
    CameraData camera;
    camera.name = cam_name;
    camera.mjv_cam.type = mjCAMERA_FIXED;
    camera.mjv_cam.fixedcamid = i;
    camera.width = static_cast<uint32_t>(cam_resolution[0]);
    camera.height = static_cast<uint32_t>(cam_resolution[1]);
    camera.viewport = { 0, 0, cam_resolution[0], cam_resolution[1] };

    const std::string param_ns = param_prefix + cam_name + ".";

    const std::string policy_param = param_ns + "policy";
    if (!node_->has_parameter(policy_param))
    {
      node_->declare_parameter(policy_param, "streaming");
    }
    std::string policy_str = node_->get_parameter(policy_param).as_string();
    if (policy_str == "streaming")
    {
      camera.policy = CameraPolicy::STREAMING;
    }
    else if (policy_str == "polled")
    {
      camera.policy = CameraPolicy::POLLED;
    }
    else if (policy_str == "disabled")
    {
      camera.policy = CameraPolicy::DISABLED;
    }
    else
    {
      RCLCPP_ERROR(node_->get_logger(), "Invalid policy for camera '%s'", cam_name);
      return false;
    }

    const std::string frame_param = param_ns + "frame_name";
    if (!node_->has_parameter(frame_param))
    {
      node_->declare_parameter(frame_param, camera.name + "_frame");
    }
    camera.frame_name = node_->get_parameter(frame_param).as_string();

    if (!node_->has_parameter(param_ns + "info_topic"))
    {
      node_->declare_parameter(param_ns + "info_topic", camera.name + "/camera_info");
    }
    camera.info_topic = node_->get_parameter(param_ns + "info_topic").as_string();

    if (!node_->has_parameter(param_ns + "image_topic"))
    {
      node_->declare_parameter(param_ns + "image_topic", camera.name + "/color");
    }
    camera.image_topic = node_->get_parameter(param_ns + "image_topic").as_string();

    if (!node_->has_parameter(param_ns + "depth_topic"))
    {
      node_->declare_parameter(param_ns + "depth_topic", camera.name + "/depth");
    }
    camera.depth_topic = node_->get_parameter(param_ns + "depth_topic").as_string();

    if (!node_->has_parameter(param_ns + "trigger_service_name"))
    {
      node_->declare_parameter(param_ns + "trigger_service_name", camera.name + "/trigger");
    }
    camera.trigger_service_name = node_->get_parameter(param_ns + "trigger_service_name").as_string();

    RCLCPP_INFO(node_->get_logger(), "Adding camera: '%s'", cam_name);
    RCLCPP_INFO(node_->get_logger(), "    policy: '%s'", policy_str.c_str());
    RCLCPP_INFO(node_->get_logger(), "    frame_name: '%s'", camera.frame_name.c_str());
    if (camera.policy == CameraPolicy::DISABLED)
    {
      continue;
    }
    RCLCPP_INFO(node_->get_logger(), "    info_topic: '%s'", camera.info_topic.c_str());
    RCLCPP_INFO(node_->get_logger(), "    image_topic: '%s'", camera.image_topic.c_str());
    RCLCPP_INFO(node_->get_logger(), "    depth_topic: '%s'", camera.depth_topic.c_str());
    if (camera.policy == CameraPolicy::POLLED)
    {
      RCLCPP_INFO(node_->get_logger(), "    trigger_service_name: '%s'", camera.trigger_service_name.c_str());
    }

    // Configure publishers and services
    camera.camera_info_pub = node_->create_publisher<sensor_msgs::msg::CameraInfo>(camera.info_topic, 1);
    camera.image_pub = node_->create_publisher<sensor_msgs::msg::Image>(camera.image_topic, 1);
    camera.depth_image_pub = node_->create_publisher<sensor_msgs::msg::Image>(camera.depth_topic, 1);
    if (camera.policy == CameraPolicy::POLLED)
    {
      camera.trigger_service = node_->create_service<std_srvs::srv::Trigger>(
          camera.trigger_service_name, [this, i](const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            this->handle_trigger(request, response, i);
          });
    }
    // Setup containers for color image data
    camera.image.header.frame_id = camera.frame_name;

    const auto image_size = camera.width * camera.height * 3;
    camera.image_buffer.resize(image_size);
    camera.image.data.resize(image_size);
    camera.image.width = camera.width;
    camera.image.height = camera.height;
    camera.image.step = camera.width * 3;
    camera.image.encoding = sensor_msgs::image_encodings::RGB8;

    // Depth image data
    camera.depth_image.header.frame_id = camera.frame_name;
    camera.depth_buffer.resize(camera.width * camera.height);
    camera.depth_image.data.resize(camera.width * camera.height * sizeof(float));
    camera.depth_image.width = camera.width;
    camera.depth_image.height = camera.height;
    camera.depth_image.step = camera.width * sizeof(float);
    camera.depth_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;

    // Camera info
    camera.camera_info.header.frame_id = camera.frame_name;
    camera.camera_info.width = camera.width;
    camera.camera_info.height = camera.height;
    camera.camera_info.distortion_model = "plumb_bob";
    camera.camera_info.k.fill(0.0);
    camera.camera_info.r.fill(0.0);
    camera.camera_info.p.fill(0.0);
    camera.camera_info.d.resize(5, 0.0);

    double focal_scaling = (1.0 / std::tan((cam_fovy * M_PI / 180.0) / 2.0)) * camera.height / 2.0;
    camera.camera_info.k[0] = camera.camera_info.p[0] = focal_scaling;
    camera.camera_info.k[2] = camera.camera_info.p[2] = static_cast<double>(camera.width) / 2.0;
    camera.camera_info.k[4] = camera.camera_info.p[5] = focal_scaling;
    camera.camera_info.k[5] = camera.camera_info.p[6] = static_cast<double>(camera.height) / 2.0;
    camera.camera_info.k[8] = camera.camera_info.p[10] = 1.0;

    // Add to list of cameras
    cameras_.push_back(camera);
  }

  has_streaming_cameras_ = std::any_of(cameras_.begin(), cameras_.end(),
                                       [](const CameraData& cam) { return cam.policy == CameraPolicy::STREAMING; });

  // Reserve once so the per-pass render bookkeeping in `update_cameras()` never allocates.
  render_indices_.reserve(cameras_.size());

  return true;
}

void TracedCameraPlugin::close()
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    stop_requested_ = true;
    publish_images_ = false;
  }
  data_cv_.notify_one();
  if (rendering_thread_.joinable())
  {
    rendering_thread_.join();
  }
}

bool TracedCameraPlugin::init_egl_context()
{
  EGLint major=0,minor=0;
  try {
    const auto selected=initialize_egl_display();
    egl_display_=selected.display;major=selected.major;minor=selected.minor;
  } catch(const std::exception& error) {
    RCLCPP_ERROR(node_->get_logger(), "EGL: %s",error.what());
    return false;
  }
  const char* client_apis = eglQueryString(egl_display_, EGL_CLIENT_APIS);
  const char* vendor = eglQueryString(egl_display_, EGL_VENDOR);
  RCLCPP_INFO(node_->get_logger(), "EGL: Initialized version %d.%d", major, minor);
  RCLCPP_INFO(node_->get_logger(), "EGL: Vendor: %s, APIs: %s", vendor, client_apis);
  // Choose EGL config for offscreen rendering
  const EGLint config_attribs[] = { EGL_SURFACE_TYPE,
                                    EGL_PBUFFER_BIT,
                                    EGL_RED_SIZE,
                                    8,
                                    EGL_GREEN_SIZE,
                                    8,
                                    EGL_BLUE_SIZE,
                                    8,
                                    EGL_ALPHA_SIZE,
                                    8,
                                    EGL_DEPTH_SIZE,
                                    24,
                                    EGL_RENDERABLE_TYPE,
                                    EGL_OPENGL_BIT,
                                    EGL_NONE };

  EGLConfig egl_config;
  EGLint num_configs;
  if (!eglChooseConfig(egl_display_, config_attribs, &egl_config, 1, &num_configs) || num_configs == 0)
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to choose config (error: 0x%x)", eglGetError());
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Bind OpenGL API
  if (!eglBindAPI(EGL_OPENGL_API))
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to bind OpenGL API (error: 0x%x)", eglGetError());
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Create EGL context
  egl_context_ = eglCreateContext(egl_display_, egl_config, EGL_NO_CONTEXT, nullptr);
  if (egl_context_ == EGL_NO_CONTEXT)
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to create context (error: 0x%x)", eglGetError());
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Create PBuffer surface for offscreen rendering
  const EGLint pbuffer_attribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
  egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config, pbuffer_attribs);
  if (egl_surface_ == EGL_NO_SURFACE)
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to create PBuffer surface (error: 0x%x)", eglGetError());
    eglDestroyContext(egl_display_, egl_context_);
    eglTerminate(egl_display_);
    egl_context_ = EGL_NO_CONTEXT;
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Make the context current
  if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_))
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to make context current (error: 0x%x)", eglGetError());
    cleanup_egl_context();
    return false;
  }

  try {
    const auto backend=render_backend_info();
    RCLCPP_INFO(node_->get_logger(), "OpenGL renderer: %s / %s",backend.vendor.c_str(),backend.renderer.c_str());
  } catch(const std::exception& error) {
    RCLCPP_ERROR(node_->get_logger(), "OpenGL: %s",error.what());
    cleanup_egl_context();return false;
  }
  RCLCPP_INFO(node_->get_logger(), "EGL: Successfully initialized headless OpenGL context");
  return true;
}

void TracedCameraPlugin::cleanup_egl_context()
{
  if (egl_display_ != EGL_NO_DISPLAY)
  {
    eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (egl_surface_ != EGL_NO_SURFACE)
    {
      eglDestroySurface(egl_display_, egl_surface_);
      egl_surface_ = EGL_NO_SURFACE;
    }
    if (egl_context_ != EGL_NO_CONTEXT)
    {
      eglDestroyContext(egl_display_, egl_context_);
      egl_context_ = EGL_NO_CONTEXT;
    }
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
  }
}

void TracedCameraPlugin::update_loop()
{
  if (!init_egl_context()) {
    publish_images_=false;
    return;
  }
  RCLCPP_INFO(node_->get_logger(), "Initializing rendering for cameras (using EGL)");

  // Initialization of the context and data structures has to happen in the rendering thread
  mjv_defaultOption(&mjv_opt_);
  mjv_defaultScene(&mjv_scn_);
  mjr_defaultContext(&mjr_con_);

  // Turn rangefinder rendering off so we don't get rays in camera images
  mjv_opt_.flags[mjtVisFlag::mjVIS_RANGEFINDER] = 0;
  // Turn off site rendering so that visualization is more realistic in cameras for testing perception.
  for (int i = 0; i < mjNGROUP; i++)
  {
    mjv_opt_.sitegroup[i] = 0;
  }

  // Initialize data for camera rendering
  mj_camera_data_ = mj_makeData(mj_model_);
  mj_render_data_ = mj_makeData(mj_model_);
  RCLCPP_INFO(node_->get_logger(), "Starting the camera rendering loop, publishing at %f Hz", camera_publish_rate_);

  // create scene and context
  mjv_makeScene(mj_model_, &mjv_scn_, 2000);

  mjr_makeContext(mj_model_, &mjr_con_, mjFONTSCALE_150);

  // Ensure the context will support the largest cameras
  int max_width = 1, max_height = 1;
  for (const auto& cam : cameras_)
  {
    max_width = std::max(max_width, static_cast<int>(cam.width));
    max_height = std::max(max_height, static_cast<int>(cam.height));
  }
  mjr_resizeOffscreen(max_width, max_height, &mjr_con_);
  RCLCPP_INFO(node_->get_logger(), "Resized offscreen buffer to %d x %d", max_width, max_height);

  // Only process images once all data has been initialized, and do it until told to stop.
  // Publishing is enabled under the lock so that a shutdown requested during the (possibly
  // slow) initialization above is observed here rather than being clobbered.
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    publish_images_ = !stop_requested_;
  }
  while (rclcpp::ok() && !stop_requested_)
  {
    std::unique_lock<std::mutex> lock(data_mutex_);

    // Wait for the main thread to copy the data and trigger this to process and publish
    // images. Note that condition_variables can be awoken spuriously, so the additional
    // checks are necessary to avoid doing work excepting when updated rendering data has
    // been made available.
    data_cv_.wait(lock, [this] { return new_data_ || stop_requested_; });

    // Shutdown triggered, kill the loop and clean up.
    if (stop_requested_)
    {
      break;
    }
    new_data_ = false;
    prepare_render_snapshot();
    lock.unlock();

    update_cameras();
  }
  publish_images_ = false;

  mjv_freeScene(&mjv_scn_);
  mjr_freeContext(&mjr_con_);
  mj_deleteData(mj_camera_data_);mj_camera_data_=nullptr;
  mj_deleteData(mj_render_data_);mj_render_data_=nullptr;

  cleanup_egl_context();
}

void TracedCameraPlugin::prepare_render_snapshot()
{
  // Caller owns data_mutex_. Rendering uses a separate buffer after releasing it.
  pass_trace_=pending_trace_;pass_trace_.worker_take_wall_ns=camera_wall_ns();
  pass_trace_.worker_copy_begin_wall_ns=camera_wall_ns();
  copy_camera_snapshot(mj_render_data_,mj_model_,mj_camera_data_);
  pass_trace_.worker_copy_end_wall_ns=camera_wall_ns();
  // Gather the cameras flagged for rendering this pass, clearing the flag as we go.
  render_indices_.clear();
  for (size_t i = 0; i < cameras_.size(); ++i)
  {
    auto& camera = cameras_[i];
    if (camera.render_pending)
    {
      camera.render_pending = false;
      render_indices_.push_back(i);
    }
  }
  if (render_indices_.empty())
  {
    return;
  }

  pass_trace_.selected_count=static_cast<int64_t>(render_indices_.size());
}
void TracedCameraPlugin::update_cameras()
{
  if(render_indices_.empty())return;
  pass_trace_.render_pass_begin_wall_ns=camera_wall_ns();
  // Rendering uses the worker-private copy of the accepted control snapshot.
  mjr_setBuffer(mjFB_OFFSCREEN, &mjr_con_);

  camera_near_distance_ = static_cast<float>(mj_model_->vis.map.znear * mj_model_->stat.extent);
  const float far = static_cast<float>(mj_model_->vis.map.zfar * mj_model_->stat.extent);
  camera_depth_scale_ = 1.0f - camera_near_distance_ / far;

  // Use the snapshotted data's timestamp in the ROS header, as that is when the data was actually pulled.
  const rclcpp::Duration duration = rclcpp::Duration::from_seconds(mj_render_data_->time);
  rclcpp::Time stamp(duration.nanoseconds(), RCL_ROS_TIME);

  pass_trace_.camera_order=0;
  for (const auto idx : render_indices_)
  {
    render_and_publish_camera(cameras_[idx], stamp);
    ++pass_trace_.camera_order;
  }
}

void TracedCameraPlugin::render_and_publish_camera(CameraData& camera, const rclcpp::Time& stamp)
{
  auto trace=pass_trace_;trace.capture_ns=stamp.nanoseconds();trace.use_egl=use_egl_;
  trace.render_data_begin_ns=rclcpp::Duration::from_seconds(mj_render_data_->time).nanoseconds();
  render_camera_frame(mj_model_,mj_render_data_,mjv_opt_,mjv_scn_,mjr_con_,camera,
                      camera_near_distance_,camera_depth_scale_,trace);
  trace.render_data_end_ns=rclcpp::Duration::from_seconds(mj_render_data_->time).nanoseconds();
  // Step 3: Publish the images and camera info.
  camera.image.header.stamp = stamp;
  camera.depth_image.header.stamp = stamp;
  camera.camera_info.header.stamp = stamp;

  trace.image_publish_ros_ns=node_->get_clock()->now().nanoseconds();
  trace.image_publish_enter_wall_ns=camera_wall_ns();trace.image_publish_enter_system_ns=camera_system_ns();
  camera.image_pub->publish(camera.image);
  trace.image_publish_return_wall_ns=camera_wall_ns();
  camera.depth_image_pub->publish(camera.depth_image);
  camera.camera_info_pub->publish(camera.camera_info);
  trace.metadata_publish_return_wall_ns=camera_wall_ns();
  if(camera.name=="overhead"){
    std_msgs::msg::String record;record.data=trace.json();trace_pub_->publish(record);
  }
}

void TracedCameraPlugin::handle_trigger(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
                                  std::shared_ptr<std_srvs::srv::Trigger::Response> response, const int camera_idx)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    cameras_.at(camera_idx).poll_requested = true;
  }
  // Wake the fast path in `update()` so the request is picked up on the next sim step.
  poll_pending_.store(true);
  response->success = true;
}

}  // namespace sorting_simulation

// Export the plugin
PLUGINLIB_EXPORT_CLASS(sorting_simulation::TracedCameraPlugin,
                       mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
