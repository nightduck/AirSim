from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cinematography",
            node_executable="motion_planner",
            node_name="motion_planner",
            remappings=[
                ("actor_traj", "/auto_cinematography/planning/actor_traj"),
                ("drone_traj", "/auto_cinematography/planning/drone_traj"),
                ("tsdf", "/auto_cinematography/mapping/tsdf")
            ],
            parameters=[
                {"airsim_hostname" : "localhost"}
            ]
        )
    ])