# Filter state is referenced to header.stamp, not publication time.
std_msgs/Header header
uint64 episode_id
uint64 track_id
bool valid
string status
string reason
float64[4] state
float64[16] covariance
builtin_interfaces/Time last_observation_time
builtin_interfaces/Time last_update_time
string view_id
float64 quality
float64 fixed_handle_z
geometry_msgs/Quaternion fixed_orientation
string pose_prior
bool camera_only
string[] class_ids
float64[] class_probabilities

# Diagnostic forecast at header.stamp + configured bounded lead.
builtin_interfaces/Time prediction_time
float64[4] prediction_state
float64[16] prediction_covariance
