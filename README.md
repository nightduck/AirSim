# Welcome to AirSim

AirSim is a simulator for drones, cars and more, built on [Unreal Engine](https://www.unrealengine.com/) (we now also have an experimental [Unity](https://unity3d.com/) release). It is open-source, cross platform, and supports software-in-the-loop simulation with popular flight controllers such as PX4 & ArduPilot and hardware-in-loop with PX4 for physically and visually realistic simulations. It is developed as an Unreal plugin that can simply be dropped into any Unreal environment. Similarly, we have an experimental release for a Unity plugin.

Our goal is to develop AirSim as a platform for AI research to experiment with deep learning, computer vision and reinforcement learning algorithms for autonomous vehicles. For this purpose, AirSim also exposes APIs to retrieve data and control vehicles in a platform independent way.

**Check out the quick 1.5 minute demo**

Drones in AirSim

[![AirSim Drone Demo Video](docs/images/demo_video.png)](https://youtu.be/-WfTr1-OBGQ)

Cars in AirSim

[![AirSim Car Demo Video](docs/images/car_demo_video.png)](https://youtu.be/gnz1X3UNM5Y)


## How to Get It

### Windows
[![Build Status](https://github.com/microsoft/AirSim/actions/workflows/test_windows.yml/badge.svg)](https://github.com/microsoft/AirSim/actions/workflows/test_windows.yml)
* [Download binaries](https://github.com/Microsoft/AirSim/releases)
* [Build it](https://microsoft.github.io/AirSim/build_windows)

### Linux
[![Build Status](https://github.com/microsoft/AirSim/actions/workflows/test_ubuntu.yml/badge.svg)](https://github.com/microsoft/AirSim/actions/workflows/test_ubuntu.yml)
* [Download binaries](https://github.com/Microsoft/AirSim/releases)
* [Build it](https://microsoft.github.io/AirSim/build_linux)

### macOS
[![Build Status](https://github.com/microsoft/AirSim/actions/workflows/test_macos.yml/badge.svg)](https://github.com/microsoft/AirSim/actions/workflows/test_macos.yml)
* [Build it](https://microsoft.github.io/AirSim/build_linux)

For more details, see the [use precompiled binaries](docs/use_precompiled.md) document. 
## Build Instructions

Execute the following to download this repo and build the AirSim component

    git clone --recursive https://github.com/nightduck/AirSim.git   # Clone this repo
    cd AirSim
    ./setup.sh                                                      # Build repo
    ./build.sh

The environment required to run the ros packages is provided, just run

    docker run -e DISPLAY=$DISPLAY -v=/tmp/.X11-unix:/tmp/.X11-unix \
        --gpus all -it --rm -v /path/to/this/repo:/workspace/AirSim \
        nightduck/airsim_cinematography:jp4.5_dashing_devel airsim_hostname 192.168.0.255

In order to run the debug nodes with GUIs, you need to give docker access to your display:

    xhost +local:docker

Set /path/to/repo to this absolute path to this repository, so you can mount it in the docker container. Set airsim_hostname and 192.168.0.255 to the hostname and IP address of the machine running airsim (likely the one you're typing on)

This will open a new shell. In the current directory is this repo, as well as the 2 appropriate tensorrt engines. To build the ros2 application:

    cd ros2
    ./build.sh

There are separately tagged docker images, built for arm64, that run the full application pipeline instead of a bash shell. They are not stable yet, but the in-progress dockerfile for the Jetson AGX is still source controlled here.

## Specifying Airsim Host
The ROS launch file at `ros/launch/ros2_wrapper.launch.py` will look for the environment variable `AIRSIM_HOSTNAME`, and use that to connect to airsim. This is set automatically when a docker container is launched with the hostname and IP arguments. If you are not working in a docker container: a) why? b) you'll have to export this variable yourself, otherwise it defaults to localhost.

All other nodes that need to connect to airsim will use the ros2_wrapper as a parameter server to get this value, so as long as `AIRSIM_HOSTNAME` is exported, no code modification is needed.


## Compiling Vision AI TensorRT Engines
Although TensorRT engines are provided with each docker image, you may want to build them for your own system. First, you need to install Onnx:

    sudo apt-get install libprotobuf-dev protobuf-compiler # protobuf is a prerequisite library
    git clone --recursive https://github.com/onnx/onnx.git # Pull the ONNX repository from GitHub 
    cd onnx
    mkdir build && cd build 
    cmake .. # Compile and install ONNX
    make # Use the ‘-j’ option for parallel jobs, for example, ‘make -j $(nproc)’ 
    make install

Then you can run the provided script to download the models weights and compile them. This assumes you have CUDA, cuDNN, and TensorRT setup on your machine.

    ./build_trt_engines.sh

## Running on Jetson

Cross compilation is not yet available, so this repo has to be built on the Jetson board, which will take several minutes for the ROS1 and ROS2 workspaces, and upwards of half an hour for the ROS1 bridge. ROS relies on hostnames to run on a distributed system. So be sure to entries to the /etc/hosts file for all machines involved. Eg if the desktop you're running airsim on is named "ubuntu-workstation" and has an IP of 192.168.0.14, then add the following line to /etc/hosts on your Jetson board.

    192.168.0.14  ubuntu-workstation

And vice-versa for the Jetson's hostname on the workstation's hosts file. A machine's hostname can be found by running `hostname`.

## What's New
* A ROS wrapper for multirotors is available. See [airsim_ros_pkgs](https://github.com/microsoft/AirSim/blob/master/ros/src/airsim_ros_pkgs) for the ROS API, and [airsim_tutorial_pkgs](https://github.com/microsoft/AirSim/blob/master/ros/src/airsim_tutorial_pkgs) for tutorials.
* [Added sensor APIs for Barometer, IMU, GPS, Magnetometer, Distance Sensor](https://microsoft.github.io/AirSim/sensors)
* Added support for [docker in ubuntu](https://github.com/microsoft/AirSim/blob/master/docs/docker_ubuntu.md)
* Added Weather Effects and [APIs](https://microsoft.github.io/AirSim/apis#weather-apis)
* Added [Time of Day API](https://microsoft.github.io/AirSim/apis#time-of-day-api)
* An experimental integration of [AirSim on Unity](https://github.com/Microsoft/AirSim/tree/master/Unity) is now available. Learn more in [Unity blog post](https://blogs.unity3d.com/2018/11/14/airsim-on-unity-experiment-with-autonomous-vehicle-simulation).
* [New environments](https://github.com/Microsoft/AirSim/releases/tag/v1.2.1): Forest, Plains (windmill farm), TalkingHeads (human head simulation), TrapCam (animal detection via camera)
* Highly efficient [NoDisplay view mode](https://microsoft.github.io/AirSim/settings#viewmode) to turn off main screen rendering so you can capture images at high rate
* [Lidar Sensor](https://microsoft.github.io/AirSim/lidar)
* Case Study: [Formula Student Technion Driverless](https://github.com/Microsoft/AirSim/wiki/technion)
* [Multi-Vehicle Capability](https://microsoft.github.io/AirSim/multi_vehicle)
* [ROS publisher](https://github.com/Microsoft/AirSim/pull/1135)

For complete list of changes, view our [Changelog](CHANGELOG.md)

## How to Use It

### Documentation

View our [detailed documentation](https://microsoft.github.io/AirSim/) on all aspects of AirSim.

### Manual drive

If you have remote control (RC) as shown below, you can manually control the drone in the simulator. For cars, you can use arrow keys to drive manually.

[More details](https://microsoft.github.io/AirSim/remote_control/)

![record screenshot](docs/images/AirSimDroneManual.gif)

![record screenshot](docs/images/AirSimCarManual.gif)


### Programmatic control

AirSim exposes APIs so you can interact with the vehicle in the simulation programmatically. You can use these APIs to retrieve images, get state, control the vehicle and so on. The APIs are exposed through the RPC, and are accessible via a variety of languages, including C++, Python, C# and Java.

These APIs are also available as part of a separate, independent cross-platform library, so you can deploy them on a companion computer on your vehicle. This way you can write and test your code in the simulator, and later execute it on the real vehicles. Transfer learning and related research is one of our focus areas.

Note that you can use [SimMode setting](https://microsoft.github.io/AirSim/settings#simmode) to specify the default vehicle or the new [ComputerVision mode](https://microsoft.github.io/AirSim/image_apis#computer-vision-mode-1) so you don't get prompted each time you start AirSim.

[More details](https://microsoft.github.io/AirSim/apis/)

### Gathering training data

There are two ways you can generate training data from AirSim for deep learning. The easiest way is to simply press the record button in the lower right corner. This will start writing pose and images for each frame. The data logging code is pretty simple and you can modify it to your heart's content.

![record screenshot](docs/images/record_data.png)

A better way to generate training data exactly the way you want is by accessing the APIs. This allows you to be in full control of how, what, where and when you want to log data.

### Computer Vision mode

Yet another way to use AirSim is the so-called "Computer Vision" mode. In this mode, you don't have vehicles or physics. You can use the keyboard to move around the scene, or use APIs to position available cameras in any arbitrary pose, and collect images such as depth, disparity, surface normals or object segmentation.

[More details](https://microsoft.github.io/AirSim/image_apis/)

### Weather Effects

Press F10 to see various options available for weather effects. You can also control the weather using [APIs](https://microsoft.github.io/AirSim/apis#weather-apis). Press F1 to see other options available.

![record screenshot](docs/images/weather_menu.png)

## Tutorials

- [Video - Setting up AirSim with Pixhawk Tutorial](https://youtu.be/1oY8Qu5maQQ) by Chris Lovett
- [Video - Using AirSim with Pixhawk Tutorial](https://youtu.be/HNWdYrtw3f0) by Chris Lovett
- [Video - Using off-the-self environments with AirSim](https://www.youtube.com/watch?v=y09VbdQWvQY) by Jim Piavis
- [Reinforcement Learning with AirSim](https://microsoft.github.io/AirSim/reinforcement_learning) by Ashish Kapoor
- [The Autonomous Driving Cookbook](https://aka.ms/AutonomousDrivingCookbook) by Microsoft Deep Learning and Robotics Garage Chapter
- [Using TensorFlow for simple collision avoidance](https://github.com/simondlevy/AirSimTensorFlow) by Simon Levy and WLU team

## Participate

### Paper

More technical details are available in [AirSim paper (FSR 2017 Conference)](https://arxiv.org/abs/1705.05065). Please cite this as:
```
@inproceedings{airsim2017fsr,
  author = {Shital Shah and Debadeepta Dey and Chris Lovett and Ashish Kapoor},
  title = {AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles},
  year = {2017},
  booktitle = {Field and Service Robotics},
  eprint = {arXiv:1705.05065},
  url = {https://arxiv.org/abs/1705.05065}
}
```

### Contribute

Please take a look at [open issues](https://github.com/microsoft/airsim/issues) if you are looking for areas to contribute to.

* [More on AirSim design](https://microsoft.github.io/AirSim/design)
* [More on code structure](https://microsoft.github.io/AirSim/code_structure)
* [Contribution Guidelines](CONTRIBUTING.md)

### Who is Using AirSim?

We are maintaining a [list](https://microsoft.github.io/AirSim/who_is_using) of a few projects, people and groups that we are aware of. If you would like to be featured in this list please [make a request here](https://github.com/microsoft/airsim/issues).

## Contact

Join our [GitHub Discussions group](https://github.com/microsoft/AirSim/discussions) to stay up to date or ask any questions.

We also have an AirSim group on [Facebook](https://www.facebook.com/groups/1225832467530667/). 


## What's New

- [Python wrapper for Open AI gym interfaces.](https://github.com/microsoft/AirSim/pull/3215)
- [Python wrapper for Event camera simulation](https://github.com/microsoft/AirSim/pull/3202)
- [Voxel grid construction](https://github.com/microsoft/AirSim/pull/3209)
- [Programmable camera distortion](https://github.com/microsoft/AirSim/pull/3039)
- [Wind simulation](https://github.com/microsoft/AirSim/pull/2867)
- [Azure development environment with documentation](https://github.com/microsoft/AirSim/pull/2816)
- ROS wrapper for [multirotor](https://github.com/microsoft/AirSim/blob/master/docs/airsim_ros_pkgs.md) and [car](https://github.com/microsoft/AirSim/pull/2743).

For complete list of changes, view our [Changelog](docs/CHANGELOG.md)

## FAQ

If you run into problems, check the [FAQ](https://microsoft.github.io/AirSim/faq) and feel free to post issues in the  [AirSim](https://github.com/Microsoft/AirSim/issues) repository.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## License

This project is released under the MIT License. Please review the [License file](LICENSE) for more details.


