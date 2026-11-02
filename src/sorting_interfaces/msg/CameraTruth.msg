# Evaluator-only snapshot. Never subscribe from perception or control.
std_msgs/Header header
uint64 episode_id
string view_id
string camera_frame
geometry_msgs/Pose camera_pose
geometry_msgs/Point marker_position
geometry_msgs/Point handle_position
