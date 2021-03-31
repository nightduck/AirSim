from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo, DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import ThisLaunchFileDir, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz_cfg',
            default_value=[ThisLaunchFileDir(), '/debug.rviz'],
            description="Filepath to rviz setup file"
        ),
        Node(
            package="rviz2",
            node_executable="rviz2",
            node_name="rviz2",
            arguments=['-d', LaunchConfiguration("rviz_cfg")]
        ),
        Node(
            package="rqt_image_view",
            node_executable="rqt_image_view",
            node_name="rqt_image_view",
            parameters=[
                {"image" : "/airsim_ros2_wrapper/camera"}
            ]
        ),
        Node(
            package="cinematography",
            node_executable="debug_viz",
            node_name="debug_viz",
            output="screen",
            remappings=[
                ("pose_out", "/auto_cinematography/debug/pose"),
                ("actor_traj_out", "/auto_cinematography/debug/actor_traj"),
                ("drone_traj_out", "/auto_cinematography/debug/drone_traj"),
                ("ideal_traj_out", "/auto_cinematography/debug/ideal_traj"),
                ("img_out", "/auto_cinematography/debug/camera"),
                ("tsdf_occupied_voxels", "/auto_cinematography/debug/tsdf"),
                ("pose_in", "/airsim_ros2_wrapper/pose/drone_1/front_center_custom"),
                ("actor_traj_in", "/auto_cinematography/planning/actor_traj"),
                ("drone_traj_in", "/auto_cinematography/planning/drone_traj"),
                ("ideal_traj_in", "/auto_cinematography/planning/ideal_traj"),
                ("img_in", "/airsim_ros2_wrapper/camera"),
                ("vm_in", "/auto_cinematography/vision/vision_measurements"),
                ("tsdf", "/auto_cinematography/mapping/tsdf")
            ],
        )
    ])
