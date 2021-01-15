import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/ros2_wrapper.launch.py'])
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/vision_pipeline.launch.py'])
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/planning_pipeline.launch.py'])
        ),
        Node(
            package='tsdf_package',
            node_executable='tsdf_node',
            node_name='tsdf_node',
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
                    "visualize_published_voxels" : True,
                    "publish_distance_squared" : 2500.0,
                    "garbage_collect_distance_squared" : 250000.0,
                    "lidar_frame" : "drone_1/LidarCustom"
                }
            ]
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/debug_viz.launch.py'])
        )
    ])