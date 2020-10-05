cmake_minimum_required(VERSION 3.5)
project(cinematography_msgs)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(msgs
	"msg/BoundingBox.msg"
	"msg/GimbalAngleEulerCmd.msg"
	"msg/GimbalAngleQuatCmd.msg"
	"msg/GPSYaw.msg"
	"msg/VelCmd.msg"
	"msg/VelCmdGroup.msg"
	"msg/MultiDOF.msg"
	"msg/MultiDOFarray.msg"
	"msg/BoolPlusHeader.msg"
	"msg/DroneState.msg"
	"msg/ArtisticSpec.msg"
)

set(srvs
	"srv/SetGPSPosition.srv"
	"srv/Takeoff.srv"
	"srv/TakeoffGroup.srv"
	"srv/Land.srv"
	"srv/LandGroup.srv"
	"srv/Reset.srv"
	"srv/SetLocalPosition.srv"
	"srv/GetTrajectory.srv"
	"srv/FollowTrajectoryStatusSrv.srv"
	"srv/ProfilingDataSrv.srv"
  "srv/StartProfilingSrv.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msgs}
  ${srvs}
  DEPENDENCIES sensor_msgs nav_msgs std_msgs geometry_msgs trajectory_msgs
  ADD_LINTER_TESTS
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
)

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

ament_export_dependencies(rosidl_default_runtime)
ament_export_include_directories(include)
ament_package()
