# Architecture

## Perception

Overhead and wrist cameras publish RGB images and camera calibration.
Localization joins image metadata and transforms at capture time.
The motion filter estimates tray-handle position, velocity and uncertainty.
The classifier assigns calibrated probabilities to synthetic red and green cues.
Class and motion evidence retain capture, episode and target identity.

## Inspection supervisor

The supervisor combines class confidence, target tracking and measured movement budgets.
It requests one extra view when evidence is uncertain and enough time remains.
A clear observation can produce a pickup proposal.
An uncertain or late target can produce a pass decision.
Moving-inspection proposals do not directly command a grasp.

## Motion and command ownership

MoveIt plans collision-aware transfers.
Cartesian Servo handles approach and precise alignment.
The command gate permits one arm owner at a time.
Hold, planned execution and Servo use explicit handoffs and fresh joint feedback.

## Physical interaction

The Panda grips the tray handle through finger contact.
Contact, slip, orientation and payload support checks constrain handling.
Placement uses physical wrist-wrench feedback.
Reset clears episode state and returns the simulator and controllers to a checked configuration.

## Evaluation boundary

Policy inputs use camera-derived evidence.
Evaluator truth checks localization, transport and physical outcomes separately.
Oracle target input is available only for the explicitly labelled debug baseline.
The camera-driven sorting result does not depend on oracle target commands.

## Main interfaces

- ROS messages and services: `sorting_interfaces`.
- Joint feedback and simulation time: `/joint_states` and `/clock`.
- Command ownership state: `/sorting/arm/owner_state`.
- Planning and Servo: MoveIt 2 services, actions and Cartesian commands.
- Model and fixture configuration: package `config/` directories.
- Validation entry points: installed `validate_*.py`, `test_*.py` and native probe executables.
