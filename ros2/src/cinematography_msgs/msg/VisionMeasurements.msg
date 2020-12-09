std_msgs/Header header
float32 hde                 # Range [-pi,pi]. Rotation relative to actor
float32 centerx				# 0 is pinned on left edge, 1 is pinned right
float32 centery				# 0 is pinned to top edge, 1 is pinned to bottom
float32 width
float32 height
float32 fov                   # FOV in radians of the camera (likely to be constant)
geometry_msgs/Pose drone_pose   # Position of the drone's camera in absolute space at time the picture was taken