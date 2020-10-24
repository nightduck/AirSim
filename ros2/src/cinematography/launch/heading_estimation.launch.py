from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cinematography",
            node_executable="heading_estimation",
            node_name="heading_estimation",
            remappings=[
                ("rviz_pose", "/rviz/pose"),
                ("actor_pose", "/auto_cinematography/vision/actor_pose"),
                ("bounding_box", "/auto_cinematography/vision/bounding_box"),
                ("satellite_pose", "/airsim_node/drone_1/global_gps"),
                ("odom_pos", "/airsim_node/drone_1/odom_local_ned"),
            ]
        )
    ])
