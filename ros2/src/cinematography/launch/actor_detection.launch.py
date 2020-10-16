from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cinematography",
            node_executable="actor_detection",
            node_name="actor_detection",
            remappings=[
                ("gimbal", "/airsim_node/gimbal_angle_quat_cmd"),
                ("camera", "/airsim_node/drone_1/front_center_custom/Scene"),
                ("bounding_box", "/actor_detection/bounding_box")
            ]
        )
    ])
