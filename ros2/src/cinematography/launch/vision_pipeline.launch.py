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
                {"airsim_hostname" : "ubuntu-workstation"},
                {"tensorrt_engine" : "yolo4_deer_fp32.rt"}
            ]
        ),
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
                {"tensorrt_engine" : "hde_deer_airsim.rt"}
            ]
        ),
        Node(
            package="cinematography",
            node_executable="motion_forecasting",
            node_name="motion_forecasting",
            remappings=[
                ("pred_path", "/auto_cinematography/planning/actor_traj"),
                ("vision_measurements", "/auto_cinematography/vision/vision_measurements")
            ],
            parameters=[
                {"airsim_hostname": "ubuntu-workstation"}
            ]
        )
    ])
