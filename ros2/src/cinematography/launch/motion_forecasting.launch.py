from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cinematography",
            node_executable="motion_forecasting",
            node_name="motion_forecasting",
            remappings=[
                ("pred_path", "/auto_cinematography/planning/actor_traj"),
                ("vision_measurements", "/auto_cinematography/vision/vision_measurements")
            ],
            parameters=[
                {"airsim_hostname": "localhost"}
            ]
        )
    ])
