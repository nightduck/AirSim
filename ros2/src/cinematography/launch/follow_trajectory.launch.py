from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cinematography",
            node_executable="follow_trajectory",
            node_name="follow_trajectory",
            remappings=[
                ("drone_traj", "/auto_cinematography/planning/drone_traj"),
                ("velocities", "/auto_cinematography/planning/velocities")
            ]
        )
    ])