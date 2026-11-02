# Capture-time RGB classification. No simulator label enters this message.
std_msgs/Header header
uint64 episode_id
string view_id
bool valid
bool accepted
string reason
string[] class_ids
float64[] probabilities
float64 confidence
float64 entropy
float64 temperature
float64 inference_wall_s
int32[4] crop_xyxy
string model_sha256
