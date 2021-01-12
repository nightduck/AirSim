from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cinematography",
            node_executable="airsim_ros2_wrapper",
            node_name="airsim_ros2_wrapper",
            output="screen",
            parameters=[
                {"vehicle" : "drone_1"},
                {"camera" : "front_center_custom"},
                {"lidar_name" : "LidarCustom"},
                {"is_vulkan" : False}   # Required parameter
            ]
        ),
        Node(
            package="tf2_ros",
            node_executable="static_transform_publisher",
            node_name="ned_to_enu_pub",
            output="screen",
            arguments=['0', '0', '0', '0.707', '0.707', '0', '0', 'world_ned', 'world_enu']
        )
    ])
