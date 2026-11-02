# header.stamp is decision time; observation_stamp remains capture time.
std_msgs/Header header
builtin_interfaces/Time observation_stamp
builtin_interfaces/Time target_stamp
uint64 episode_id
uint64 sequence
bool oracle_only
bool valid
string reason
geometry_msgs/Pose handle_pose
geometry_msgs/Vector3 velocity
float64 observation_age_s
float64 horizon_s
float64 remaining_window_s
