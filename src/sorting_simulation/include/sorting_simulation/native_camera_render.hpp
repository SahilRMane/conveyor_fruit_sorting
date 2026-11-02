#pragma once
// Native offscreen rendering. No ROS participant.
#include "sorting_simulation/static_motion.hpp"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <GL/gl.h>
#include <memory>
using namespace sorting_simulation;
struct Offscreen {
  EGLDisplay display=EGL_NO_DISPLAY;EGLContext context=EGL_NO_CONTEXT;EGLSurface surface=EGL_NO_SURFACE;
  mjvScene scene{};mjrContext render{};mjvOption options{};mjvCamera camera{};
  bool ready=false;
  void init(const mjModel* m){
    display=eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,nullptr);
    EGLint major,minor;check(eglInitialize(display,&major,&minor),"EGL initialization failed");
    check(eglBindAPI(EGL_OPENGL_API),"EGL OpenGL binding failed");
    EGLint attributes[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_BIT,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_DEPTH_SIZE,24,EGL_NONE};
    EGLConfig config;EGLint count;check(eglChooseConfig(display,attributes,&config,1,&count)&&count>0,"EGL configuration failed");
    context=eglCreateContext(display,config,EGL_NO_CONTEXT,nullptr);check(context!=EGL_NO_CONTEXT,"EGL context failed");
    EGLint size[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};surface=eglCreatePbufferSurface(display,config,size);
    check(surface!=EGL_NO_SURFACE && eglMakeCurrent(display,surface,surface,context),"EGL offscreen surface failed");
    mjv_defaultScene(&scene);mjr_defaultContext(&render);mjv_defaultOption(&options);mjv_defaultCamera(&camera);
    for(int i=0;i<mjNGROUP;++i)options.sitegroup[i]=0;
    mjv_makeScene(m,&scene,2000);mjr_makeContext(m,&render,mjFONTSCALE_100);mjr_setBuffer(mjFB_OFFSCREEN,&render);ready=true;
  }
  void save(const mjModel* m,mjData* d,int id,const std::string& filename){
    int w=m->cam_resolution[2*id],h=m->cam_resolution[2*id+1];mjrRect rectangle{0,0,w,h};
    camera.type=mjCAMERA_FIXED;camera.fixedcamid=id;
    mjv_updateScene(m,d,&options,nullptr,&camera,mjCAT_ALL,&scene);mjr_render(rectangle,&scene,&render);
    std::vector<unsigned char> rgb(w*h*3);mjr_readPixels(rgb.data(),nullptr,rectangle,&render);
    std::ofstream image(filename,std::ios::binary);check(bool(image),"Cannot write rendered image");image<<"P6\n"<<w<<' '<<h<<"\n255\n";
    for(int row=h-1;row>=0;--row)image.write(reinterpret_cast<const char*>(rgb.data()+row*w*3),w*3);
  }
  YAML::Node benchmark(const mjModel* m,mjData* d){
    YAML::Node out;out["gl_renderer"]=reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    out["shadow_map_size"]=m->vis.quality.shadowsize;out["offscreen_samples"]=m->vis.quality.offsamples;
    int shadow_lights=0;for(int i=0;i<m->nlight;++i)shadow_lights+=m->light_castshadow[i]!=0;
    out["shadow_casting_lights"]=shadow_lights;
    int repeats=24;
    for(const char* view:{"overhead","wrist"}){
      int id=mj_name2id(m,mjOBJ_CAMERA,view);int w=m->cam_resolution[2*id],h=m->cam_resolution[2*id+1];
      mjrRect rect{0,0,w,h};std::vector<unsigned char> rgb(w*h*3);std::vector<float> depth(w*h);
      camera.type=mjCAMERA_FIXED;camera.fixedcamid=id;
      std::vector<double> times;
      for(int i=0;i<repeats+2;++i){
        auto start=std::chrono::steady_clock::now();
        mjv_updateScene(m,d,&options,nullptr,&camera,mjCAT_ALL,&scene);mjr_render(rect,&scene,&render);
        mjr_readPixels(rgb.data(),depth.data(),rect,&render);
        double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(i>=2)times.push_back(elapsed);
      }
      std::sort(times.begin(),times.end());double sum=0;for(double t:times)sum+=t;
      out["views"][view]["mean_render_read_wall_s"]=sum/times.size();
      out["views"][view]["p95_render_read_wall_s"]=times[static_cast<size_t>(std::ceil(.95*times.size()))-1];
      out["views"][view]["maximum_render_read_wall_s"]=times.back();
      out["views"][view]["frames"]=repeats;
    }
    out["scope"]="Native render/read only; excludes ROS transport and delivery";
    return out;
  }
  ~Offscreen(){
    if(ready){mjv_freeScene(&scene);mjr_freeContext(&render);}
    if(display!=EGL_NO_DISPLAY){eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
      if(surface!=EGL_NO_SURFACE)eglDestroySurface(display,surface);
      if(context!=EGL_NO_CONTEXT)eglDestroyContext(display,context);
      eglTerminate(display);}
  }
};
