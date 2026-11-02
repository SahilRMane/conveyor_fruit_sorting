"""Keep RGB queues short and preserve small capture-stamped calibration packets."""
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy,HistoryPolicy,LivelinessPolicy

def camera_subscription_qos(image_depth=1,metadata_depth=64):
    image=QoSProfile(depth=image_depth,history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE,liveliness=LivelinessPolicy.AUTOMATIC)
    metadata=QoSProfile(depth=metadata_depth,history=HistoryPolicy.KEEP_LAST,
                        reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE,liveliness=LivelinessPolicy.AUTOMATIC)
    return image,metadata
