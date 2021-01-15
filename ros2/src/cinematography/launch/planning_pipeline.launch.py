from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cinematography",
            node_executable="motion_planner",
            node_name="motion_planner",
            remappings=[
                ("/actor_traj", "/auto_cinematography/planning/actor_traj")
            ],
            parameters=[
                {"airsim_hostname" : "localhost"}
            ]
        ),
        Node(
            package="cinematography",
            node_executable="follow_trajectory",
            node_name="follow_trajectory"
        )
    ])