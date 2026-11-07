# GPU rendering

## Supported path

The project camera worker uses a headless EGL context.
It creates no GLFW window and makes no X11 window-management calls.
Native image generation and camera checks use the same EGL display selector.

Set `SORTING_RENDER_BACKEND=nvidia` to require NVIDIA rendering.
The selector checks the EGL vendor and the current OpenGL renderer.
An explicit NVIDIA request fails if that renderer is unavailable.

Camera rendering uses OpenGL/EGL.
The existing ROS simulation physics, planning, filtering and command gate use CPU execution.
CUDA training and inference belong to the later neural-network extension.
Measure transfer and inference cost before moving small per-frame operations onto a GPU.

## Reference hardware

The verified device is an NVIDIA GeForce RTX 4070 Laptop GPU with 8188 MiB VRAM.
The driver reports version 595.99.02.

Native render/read benchmark, 320 x 240 images:

| Camera | Software p95 | NVIDIA p95 |
| --- | ---: | ---: |
| Overhead | 16.75 ms | 0.31 ms |
| Wrist | 8.87 ms | 0.29 ms |

The benchmark includes scene update, drawing and pixel readback.
Full execution also includes physics, image formatting, ROS delivery, planning and robot motion.

## Environment

Use this environment for both the M4C preflight and live terminals:

```bash
unset LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE __EGL_VENDOR_LIBRARY_FILENAMES
export SORTING_RENDER_BACKEND=nvidia
```

The M4C preflight binds the rendering selection.
Changing that selection requires a new preflight.
Check the renderer in `camera_side_check` within the preflight report.
The live launch also prints an `OpenGL renderer:` line.

See [Running](RUNNING.md) for launch and validation steps.

## Validation scope

The native preflight and a complete green-covered moving inspection passed on this renderer.
The live timing report recorded a 28 ms maximum delivery age with no delivery violations.
The ownership result remained HOLD with zero overlaps.
During planned teardown, MoveIt required SIGTERM. Cleanup qualification remains separate.
