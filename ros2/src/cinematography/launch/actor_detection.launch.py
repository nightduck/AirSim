from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cinematography",
            node_executable="actor_detection",
            node_name="actor_detection",
            remappings=[
                ("actor_pose", "/airsim_ros2_wrapper/pose/drone_1/front_center_custom"),
                ("camera", "/airsim_ros2_wrapper/camera"),
                ("bounding_box", "/auto_cinematography/vision/bounding_box")
            ],
            parameters=[
                {"airsim_hostname" : "localhost"},
                {"tensorrt_engine" : "yolo4_deer_fp32.rt"}
            ]
        )
    ])
