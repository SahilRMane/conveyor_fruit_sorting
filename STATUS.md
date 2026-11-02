# Implementation status

## Available

- ROS 2 Jazzy integration with MuJoCo and the Franka Panda.
- Physical tray grasp, lift, transport, bin placement and repeatable reset.
- Moving-target prediction and camera-driven interception.
- Calibrated classification of synthetic red and green cues.
- Stationary wrist inspection and bounded moving-inspection proposals.
- Exclusive command ownership, collision checks and contact-based placement.
- Automated component tests and native and live simulation validation tools.

## Evaluation scope

Fixed-view camera sorting is validated in the nominal synthetic scene at 0.03 m/s.
Moving active-inspection proposals use a separate 0.001 m/s engineering configuration.
The active-inspection workflow does not yet authorize physical pickup.
Recorded test counts and their limits are listed in [Validation](docs/VALIDATION.md).

## Planned

- Complete timing qualification for moving active inspection.
- Integrate autonomous inspect, physical pick and pass execution.
- Evaluate calibration, fusion, additional views and learned view selection.
- Compare predictive control and recovery behaviour on held-out episodes.
- Add MobileNetV3 fruit-type recognition after the core evaluation.
- Evaluate a local vision-language-action policy as an optional extension.
