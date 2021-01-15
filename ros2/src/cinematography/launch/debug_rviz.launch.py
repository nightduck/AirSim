from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="rviz2",
            node_executable="rviz2",
            node_name="rviz2",
            arguments=['-d $(ros2 pkg prefix cinematography)/share/cinematography/launch/debug.rviz']
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
                ("traj_out", "/auto_cinematography/debug/actor_traj"),
                ("img_out", "/auto_cinematography/debug/camera"),
                ("pose_in", "/airsim_ros2_wrapper/pose/drone_1/front_center_custom"),
                ("traj_in", "/auto_cinematography/planning/actor_traj"),
                ("img_in", "/airsim_ros2_wrapper/camera"),
                ("vm_in", "/auto_cinematography/vision/vision_measurements")
            ],
        )
    ])
