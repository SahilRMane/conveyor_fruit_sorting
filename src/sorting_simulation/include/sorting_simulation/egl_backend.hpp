#pragma once
// Select a headless EGL display. Explicit NVIDIA requests never fall back to software.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sorting_simulation {
inline std::string render_backend_request(){
  const char* value=std::getenv("SORTING_RENDER_BACKEND");
  std::string mode=value?value:"auto";
  if(mode!="auto" && mode!="nvidia")throw std::runtime_error("SORTING_RENDER_BACKEND must be auto or nvidia");
  return mode;
}
inline bool software_rendering_requested(){
  const char* value=std::getenv("LIBGL_ALWAYS_SOFTWARE");
  std::string flag=value?value:"";
  return flag=="1" || flag=="true" || flag=="TRUE" || flag=="yes";
}
struct InitializedEglDisplay {EGLDisplay display=EGL_NO_DISPLAY;EGLint major=0,minor=0;};
inline InitializedEglDisplay initialize_egl_display(){
  const auto mode=render_backend_request();
  const bool software=software_rendering_requested();
  if(mode=="nvidia" && software)throw std::runtime_error("Unset LIBGL_ALWAYS_SOFTWARE before selecting NVIDIA rendering");
  if(!software){
    auto query=reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
    auto platform=reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    EGLDeviceEXT devices[16]{};EGLint count=0;
    if(query && platform && query(16,devices,&count)){
      for(EGLint i=0;i<std::min(count,EGLint(16));++i){
        InitializedEglDisplay selected;
        selected.display=platform(EGL_PLATFORM_DEVICE_EXT,devices[i],nullptr);
        if(selected.display==EGL_NO_DISPLAY || !eglInitialize(selected.display,&selected.major,&selected.minor))continue;
        const char* vendor=eglQueryString(selected.display,EGL_VENDOR);
        if(vendor && std::string(vendor).find("NVIDIA")!=std::string::npos)return selected;
        eglTerminate(selected.display);
      }
    }
  }
  if(mode=="nvidia")throw std::runtime_error("No initialized NVIDIA EGL device is available");
  InitializedEglDisplay selected;
  selected.display=eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,nullptr);
  if(selected.display==EGL_NO_DISPLAY || !eglInitialize(selected.display,&selected.major,&selected.minor))
    throw std::runtime_error("Headless EGL initialization failed");
  return selected;
}
struct RenderBackendInfo {std::string vendor,renderer,version;};
inline RenderBackendInfo render_backend_info(){
  const auto read=[](GLenum key){const auto* text=glGetString(key);return text?std::string(reinterpret_cast<const char*>(text)):std::string();};
  RenderBackendInfo info{read(GL_VENDOR),read(GL_RENDERER),read(GL_VERSION)};
  if(info.vendor.empty() || info.renderer.empty())throw std::runtime_error("No current OpenGL renderer");
  if(render_backend_request()=="nvidia" && info.vendor.find("NVIDIA")==std::string::npos)
    throw std::runtime_error("The current OpenGL renderer is not NVIDIA");
  if(software_rendering_requested() && info.renderer.find("llvmpipe")==std::string::npos
       && info.renderer.find("softpipe")==std::string::npos)
    throw std::runtime_error("Software rendering was requested but the current renderer is not software");
  return info;
}
} // namespace sorting_simulation
