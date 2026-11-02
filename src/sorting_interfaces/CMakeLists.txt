cmake_minimum_required(VERSION 3.16)
project(sorting_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetArmOwner.srv" "msg/ArmServoCommand.msg" "action/ExecuteArmTrajectory.action"
  "msg/OracleTarget.msg" "msg/OraclePrediction.msg"
  "srv/SetHandleContact.srv"
  "msg/CameraEpoch.msg" "msg/HandleObservation.msg" "msg/CameraTruth.msg"
  "msg/Decision.msg" "srv/SetCameraPlacement.srv" "msg/ClassObservation.msg" "msg/TargetBelief.msg" "msg/TargetBeliefArray.msg" "srv/StartCameraPickup.srv" "srv/SetEstimatedGrasp.srv"
  DEPENDENCIES builtin_interfaces trajectory_msgs control_msgs geometry_msgs std_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
