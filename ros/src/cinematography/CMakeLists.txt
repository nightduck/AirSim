cmake_minimum_required(VERSION 3.10.0)
project(cinematography)

# set this to path to AirSim root folder if you want your catkin workspace in a custom directory
set(AIRSIM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../AirSim/)
set (common_utility_src ${AIRSIM_ROOT}/ros/src/common)

add_subdirectory("${AIRSIM_ROOT}/cmake/rpclib_wrapper" rpclib_wrapper)
add_subdirectory("${AIRSIM_ROOT}/cmake/AirLib" AirLib)
add_subdirectory("${AIRSIM_ROOT}/cmake/MavLinkCom" MavLinkCom)

set(CMAKE_CXX_STANDARD 11)

set(RPC_LIB_INCLUDES " ${AIRSIM_ROOT}/external/rpclib/rpclib-2.2.1/include")
set(RPC_LIB rpc) # name of .a file with lib prefix
message(STATUS "found RPC_LIB_INCLUDES=${RPC_LIB_INCLUDES}")

# todo eigen3 in AirLib already
# find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)


# package add by feiyang jin
# pcl is for octomap_server
find_package( PCL REQUIRED )
include_directories(  ${catkin_INCLUDE_DIRS}  ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(octomap REQUIRED)
find_package(octomap_server REQUIRED)
find_package(ompl REQUIRED)


find_package(catkin REQUIRED COMPONENTS
        cv_bridge
        image_transport
        mavros_msgs
        message_generation
        nav_msgs
        # nodelet
        roscpp
        rospy
        sensor_msgs
        std_msgs
        std_srvs
        tf2
        tf2_ros
        mav_trajectory_generation
        mav_trajectory_generation_ros
        airsim_ros_pkgs
        )

add_message_files(
        FILES
        drone_state.msg
        artistic_spec.msg
)

generate_messages(
        DEPENDENCIES
        std_msgs
        geometry_msgs
        trajectory_msgs
        airsim_ros_pkgs
)

catkin_package(
        # LIBRARIES airsim_ros
        CATKIN_DEPENDS message_runtime roscpp std_msgs # nodelet
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${AIRSIM_ROOT}/AirLib/deps/eigen3
        ${AIRSIM_ROOT}/AirLib/include
        ${RPC_LIB_INCLUDES}
        ${AIRSIM_ROOT}/MavLinkCom/include
        ${AIRSIM_ROOT}/MavLinkCom/common_utils
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${common_utility_src}
)

add_executable(motion_planner src/motion_planner.cpp)
add_dependencies(motion_planner ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(motion_planner ${catkin_LIBRARIES} libompl.so ${OCTOMAP_LIBRARIES})

add_executable(follow_trajectory src/follow_trajectory_node.cpp)
add_dependencies(follow_trajectory ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(follow_trajectory ${catkin_LIBRARIES} AirLib)


install(TARGETS
        #list of shared libraries
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        )

# for shipping binaries. run `$ catin_make install`
install(DIRECTORY
        launch
        rviz
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        # PATTERN ".git" EXCLUDE
        )
