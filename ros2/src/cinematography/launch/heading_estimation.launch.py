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
                ("vision_measurements", "/auto_cinematography/vision/vision_measurements"),
                ("bounding_box", "/auto_cinematography/vision/bounding_box")
            ],
            parameters=[
                {"tensorrt_engine" : "deer_hde_fp32.rt"}
            ]
        )
    ])
