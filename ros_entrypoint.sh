#!/bin/bash

set -e

if [ -z $1 ]; then
    echo "Hostname where airsim is running not set"
    echo "Usage: docker run --gpus all -it --rm -v /path/to/this/repo:/workspace/AirSim nightduck/airsim_cinematography:jp4.5_dashing_devel AIRSIM_HOSTNAME AIRSIM_IP_ADDRESS"
    exit
elif [ -z $2 ]; then
    echo "IP address where airsim is running not set"
    echo "Usage: docker run --gpus all -it --rm -v /path/to/this/repo:/workspace/AirSim nightduck/airsim_cinematography:jp4.5_dashing_devel AIRSIM_HOSTNAME AIRSIM_IP_ADDRESS"
    exit
fi

export AIRSIM_HOSTNAME=$1
echo "$2 $1" >> /etc/hosts

ARCH=$(arch)
if [ $ARCH == "aarch64" ]; then       # If running on Jetson
    source /workspace/AirSim/ros2/install/setup.bash
    ros2 launch cinematography run_application.launch.py
elif [ $ARCH == "x86_64" ]; then      # If running on desktop
    source /opt/ros/dashing/setup.bash
    bash
else
    echo "UNSUPPORTED ARCHITECTURE! (How are you even running this?)"
fi
