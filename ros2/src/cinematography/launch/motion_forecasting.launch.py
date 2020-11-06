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
                ("actor_pose", "/auto_cinematography/vision/actor_pose")
            ]
        )
    ])
