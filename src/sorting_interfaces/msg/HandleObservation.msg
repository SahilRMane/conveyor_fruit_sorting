# header.stamp is image capture time; frame_id is world.
std_msgs/Header header
uint64 episode_id
uint64 sequence
string view_id
string camera_frame
bool valid
string reason
geometry_msgs/Point position
float64[4] planar_covariance
float64 pixel_u
float64 pixel_v
float64 area_px
float64 quality
float64 compute_wall_s
string covariance_source
