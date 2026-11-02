# Project scope and roadmap

## Objective

Evaluate whether confidence-based visual inspection improves correct tray sorting under limited interception time.
The robot handles a tray by its handle. Classification determines the destination bin.

## Current design

- Simulator: MuJoCo with ROS 2 Jazzy and ros2_control.
- Robot: Franka Panda with a parallel gripper.
- Perception: overhead and wrist RGB cameras with capture-time transforms.
- Tracking: camera-derived position, velocity and uncertainty.
- Classification: calibrated synthetic red/green class probabilities with abstention.
- Motion: collision-aware MoveIt planning and Cartesian Servo.
- Supervision: inspect, propose pickup or pass according to confidence and remaining time.
- Evaluation: isolated truth measurements, physical contact checks and complete trial accounting.

## Engineering requirements

1. Use physical finger contact for grasping. Do not force the tray onto the gripper.
2. Permit only one arm command owner at a time.
3. Preserve capture timestamps and reject stale or inconsistent observations.
4. Keep evaluator truth separate from camera-based policy inputs.
5. Bound each operation and preserve a controlled hold state.
6. Check calibration and model identity before using class evidence.
7. Generate new preflight evidence after source or environment changes.
8. Report achieved results separately from future targets.

## Roadmap

| Area | Scope |
| --- | --- |
| Scene and control | Robot integration, tray handling, conveyor, bins and reset |
| Interception | Target prediction, moving approach and physical pickup |
| Camera sorting | Localization, filtering, class confidence and bin placement |
| Active inspection | Wrist views, bounded decisions and autonomous inspect/pick/pass integration |
| Perception evaluation | Calibration, fusion, grouped data splits and learned view utility |
| Control comparison | Predictive control, bounded recovery and matched-condition evaluation |
| Release evaluation | Held-out episodes, reproducible setup and demonstration |

Current capabilities and remaining integration work are listed in [STATUS.md](STATUS.md).

## Extensions

**E1: fruit-type recognition.** Use licensed fruit assets and a MobileNetV3 baseline.
Separate object instances and scene groups across training and evaluation data.
Measure unknown-class rejection, confidence calibration, occlusion sensitivity and inference time.

**Optional learned policy.** Evaluate a local vision-language-action model after the modular baseline.
Retain command ownership, collision, timing and physical contact checks.
Compare policies on the same held-out episodes.

Calendar estimates guide effort planning. They do not require implementation delays.

## References

- [MuJoCo ROS 2 integration](https://github.com/ros-controls/mujoco_ros2_control)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [MoveIt Servo](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html)
- [Confidence calibration](https://proceedings.mlr.press/v70/guo17a.html)
