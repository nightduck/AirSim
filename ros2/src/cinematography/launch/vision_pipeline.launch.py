from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cinematography",
            node_executable="actor_detection",
            node_name="actor_detection",
            remappings=[
                ("actor_pose", "/heading_estimation/gimbal_angle_quat_cmd"),
                ("camera", "/airsim_node/drone_1/front_center_custom/Scene"),
                ("bounding_box", "/auto_cinematography/vision/bounding_box")
            ],
            parameters=[
                {"airsim_hostname" : "192.168.0.13"}
            ]
        ),
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
        ),
        Node(
            package="cinematography",
            node_executable="motion_forecasting",
            node_name="motion_forecasting",
            remappings=[
                ("pred_path", "/auto_cinematography/planning/actor_traj"),
                ("actor_pose", "/auto_cinematography/vision/actor_pose")
            ],
            parameters=[
                {"airsim_hostname": "192.168.0.13"}
            ]
        )
    ])
