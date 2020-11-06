cmake_minimum_required(VERSION 3.10.0)
project(cinematography)

# set this to path to AirSim root folder if you want your catkin workspace in a custom directory
set(AIRSIM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../AirSim/)
set (common_utility_src ${AIRSIM_ROOT}/ros/src/common)

add_subdirectory("${AIRSIM_ROOT}/cmake/rpclib_wrapper" rpclib_wrapper)
add_subdirectory("${AIRSIM_ROOT}/cmake/AirLib" AirLib)
add_subdirectory("${AIRSIM_ROOT}/cmake/MavLinkCom" MavLinkCom)

set(CMAKE_CXX_STANDARD 14)
# set(CXX_EXP_LIB "-nostdinc++ -I/usr/include/c++/8 -I/usr/include/x86_64-linux-gnu/c++/8 -nodefaultlibs 
# -l/usr/lib/x86_64-linux-gnu/libc++.so -l/usr/lib/x86_64-linux-gnu/libc++abi.so
# -lm -lc -lgcc_s -lgcc  
# -lstdc++fs -fmax-errors=10 -Wnoexcept -Wstrict-null-sentinel") 

set(RPC_LIB_INCLUDES " ${AIRSIM_ROOT}/external/rpclib/rpclib-2.2.1/include")
set(RPC_LIB rpc) # name of .a file with lib prefix
message(STATUS "found RPC_LIB_INCLUDES=${RPC_LIB_INCLUDES}")

set(FastRTPS_INCLUDE_DIR /opt/ros/dashing/include)
set(FastRTPS_LIBRARIES /opt/ros/dashing/lib/libfastrtps.so)

find_package(ament_cmake REQUIRED)
## uncomment the following section in order to fill in
## further dependencies manually.
## find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(nav_msgs REQUIRED)
##find_package(nodelet REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(cinematography_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(message_filters REQUIRED)

# todo eigen3 in AirLib already
# find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

# package add by feiyang jin
# pcl is for octomap_server
find_package( PCL REQUIRED )
include_directories(${catkin_INCLUDE_DIRS}  ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(CUDA REQUIRED)
#find_package(CUDNN REQUIRED)
#find_package(Darknet REQUIRED)

# set(msgs
# 	"msg/BoundingBox.msg"
# 	"msg/GimbalAngleEulerCmd.msg"
# 	"msg/GimbalAngleQuatCmd.msg"
# 	"msg/GPSYaw.msg"
# 	"msg/VelCmd.msg"
# 	"msg/VelCmdGroup.msg"
# 	"msg/MultiDOF.msg"
# 	"msg/MultiDOFarray.msg"
# 	"msg/BoolPlusHeader.msg"
# 	"msg/DroneState.msg"
# 	"msg/ArtisticSpec.msg"
# )

# set(srvs
# 	"srv/SetGPSPosition.srv"
# 	"srv/Takeoff.srv"
# 	"srv/TakeoffGroup.srv"
# 	"srv/Land.srv"
# 	"srv/LandGroup.srv"
# 	"srv/Reset.srv"
# 	"srv/SetLocalPosition.srv"
# 	"srv/GetTrajectory.srv"
# 	"srv/FollowTrajectoryStatusSrv.srv"
# 	"srv/ProfilingDataSrv.srv"
# 	"srv/StartProfilingSrv.srv"
# )

# rosidl_generate_interfaces(${PROJECT_NAME}
#         ${msgs}
#         ${srvs}
#         DEPENDENCIES std_msgs geometry_msgs trajectory_msgs sensor_msgs
# )

include_directories(
        include
        ${AIRSIM_ROOT}/AirLib/deps/eigen3
        ${AIRSIM_ROOT}/AirLib/include
        ${RPC_LIB_INCLUDES}
        ${AIRSIM_ROOT}/MavLinkCom/include
        ${AIRSIM_ROOT}/MavLinkCom/common_utils
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${common_utility_src}
)

CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
SET(CUDA_NVCC_FLAGS ${ARCH_FLAGS} CACHE STRING "nvcc flags" FORCE)
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC)

# cuda compile GPU_CODE for executable
#cuda_add_library(filterImage src/filterImage.cu)

#set(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)
#set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
#set(LIB_TYPE STATIC)

# add_library(airsim_settings_parser src/airsim_settings_parser.cpp)
# ament_target_dependencies(airsim_settings_parser AirLib)

# add_library(airsim_ros src/airsim_ros_wrapper.cpp)
# ament_target_dependencies(airsim_ros rclcpp ${OpenCV_LIBS} yaml-cpp AirLib airsim_settings_parser cinematography_msgs)
# #rosidl_target_interfaces(airsim_ros ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(actor_detection src/actor_detection.cpp)
ament_target_dependencies(actor_detection rclcpp filterImage cinematography_msgs sensor_msgs geometry_msgs cv_bridge)
target_link_libraries(actor_detection AirLib)
#rosidl_target_interfaces(actor_detection ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(heading_estimation src/heading_estimation.cpp)
ament_target_dependencies(heading_estimation rclcpp filterImage cinematography_msgs sensor_msgs geometry_msgs nav_msgs cv_bridge tf2_geometry_msgs)

add_executable(motion_forecasting src/motion_forecasting.cpp)
ament_target_dependencies(motion_forecasting rclcpp geometry_msgs cinematography_msgs)
target_link_libraries(motion_forecasting AirLib)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
        actor_detection heading_estimation motion_forecasting
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/
        DESTINATION include)
install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}/)

ament_export_include_directories(include)
ament_export_dependencies(std_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()