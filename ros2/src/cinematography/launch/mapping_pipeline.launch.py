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
                ("tsdf", "/auto_cinematography/mapping/tsdf"),
                ("tsdf_occupied_voxels", "/auto_cinematography/debug/markers")
            ],
            parameters=[
                {
                    "voxel_size" : .5,
                    "truncation_distance" : 4.0,
                    "max_weight" : 10000.0,
                    "visualize_published_voxels" : False,
                    "publish_distance_squared" : 2500.0,
                    "garbage_collect_distance_squared" : 250000.0,
                    "lidar_frame" : "drone_1/LidarCustom"
                }
            ],
            # prefix=['cuda-memcheck --leak-check full']
        )
    ])