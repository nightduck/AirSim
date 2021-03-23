from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cinematography',
            node_executable='mapping',
            node_name='mapping',
            remappings=[
                ("lidar", "/airsim_ros2_wrapper/lidar"),
                ("tsdf", "/auto_cinematography/mapping/tsdf")
            ],
            parameters=[
                {
                    "voxel_size" : .5,
                    "truncation_distance" : 4.0,
                    "max_weight" : 1000.0,
                    "publish_distance_squared" : 2500.0,
                    "garbage_collect_distance_squared" : 250000.0,
                    "lidar_frame" : "drone_1/LidarCustom"
                }
            ]
        )
    ])