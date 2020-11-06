if [ -z "$ROS_DISTRO" ]
then
    source ../ros/devel/setup.bash
    source ../ros2/install/setup.bash
    colcon build --symlink-install --packages-select ros1_bridge --cmake-force-configure
    source install/setup.bash
    ros2 run ros1_bridge dynamic_bridge --print-pairs 
else
    echo "ROS workspace already sourced. Please run in fresh terminal"
fi