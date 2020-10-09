#include "common/common_utils/StrictMode.hpp"
STRICT_MODE_OFF //todo what does this do?
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif // !RPCLIB_MSGPACK
#include "rpc/rpc_error.h"
STRICT_MODE_ON

#include "airsim_settings_parser.h"
#include "common/AirSimSettings.hpp"
#include "common/common_utils/FileSystem.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensors/imu/ImuBase.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include "yaml-cpp/yaml.h"
#include <cinematography_msgs/msg/gimbal_angle_euler_cmd.hpp>
#include <cinematography_msgs/msg/gimbal_angle_quat_cmd.hpp>
#include <cinematography_msgs/msg/gps_yaw.hpp>
#include <cinematography_msgs/srv/land.hpp>
#include <cinematography_msgs/srv/land_group.hpp>
#include <cinematography_msgs/srv/reset.hpp>
#include <cinematography_msgs/srv/takeoff.hpp>
#include <cinematography_msgs/srv/takeoff_group.hpp>
#include <cinematography_msgs/msg/vel_cmd.hpp>
#include <cinematography_msgs/msg/vel_cmd_group.hpp>
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <image_transport/image_transport.h>
#include <iostream>
#include <math.h>
#include <math_common.h>
//#include <mavros_msgs/state.h>
#include <nav_msgs/msg/odometry.h>
#include <opencv2/opencv.hpp>
#include <ros/callback_queue.h>
#include <ros/console.h>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/distortion_models.h>
#include <sensor_msgs/msg/image.h>
#include <sensor_msgs/msg/image_encodings.h>
#include <sensor_msgs/msg/imu.h>
#include <sensor_msgs/msg/nav_sat_fix.h>
#include <sensor_msgs/msg/point_cloud2.h>
#include <std_srvs/srv/empty.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/msg/tf2_geometry_msgs.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <unordered_map>
// #include "nodelet/nodelet.h"

// todo move airlib typedefs to separate header file?
typedef msr::airlib::ImageCaptureBase::ImageRequest ImageRequest;
typedef msr::airlib::ImageCaptureBase::ImageResponse ImageResponse;
typedef msr::airlib::ImageCaptureBase::ImageType ImageType;
typedef msr::airlib::AirSimSettings::CaptureSetting CaptureSetting;
typedef msr::airlib::AirSimSettings::VehicleSetting VehicleSetting;
typedef msr::airlib::AirSimSettings::CameraSetting CameraSetting;
typedef msr::airlib::AirSimSettings::LidarSetting LidarSetting;

// things in drone.h from Mavbench
    const float FACE_FORWARD = std::numeric_limits<float>::infinity();
    const float FACE_BACKWARD = -std::numeric_limits<float>::infinity();
    const float YAW_UNCHANGED = -1e9;


struct SimpleMatrix
{
    int rows;
    int cols;
    double* data;

    SimpleMatrix(int rows, int cols, double* data)
        : rows(rows), cols(cols), data(data)
    {}
};

struct VelCmd
{
    double x;
    double y;
    double z;
    msr::airlib::DrivetrainType drivetrain;
    msr::airlib::YawMode yaw_mode;
    std::string vehicle_name;

    // VelCmd() : 
    //     x(0), y(0), z(0), 
    //     vehicle_name("") {drivetrain = msr::airlib::DrivetrainType::MaxDegreeOfFreedom;
    //             yaw_mode = msr::airlib::YawMode();};

    // VelCmd(const double& x, const double& y, const double& z, 
    //         msr::airlib::DrivetrainType drivetrain, 
    //         const msr::airlib::YawMode& yaw_mode,
    //         const std::string& vehicle_name) : 
    //     x(x), y(y), z(z), 
    //     drivetrain(drivetrain), 
    //     yaw_mode(yaw_mode), 
    //     vehicle_name(vehicle_name) {};
};

struct GimbalCmd
{
    std::string vehicle_name;
    std::string camera_name;
    msr::airlib::Quaternionr target_quat;

    // GimbalCmd() : vehicle_name(vehicle_name), camera_name(camera_name), target_quat(msr::airlib::Quaternionr(1,0,0,0)) {}

    // GimbalCmd(const std::string& vehicle_name, 
    //         const std::string& camera_name, 
    //         const msr::airlib::Quaternionr& target_quat) : 
    //         vehicle_name(vehicle_name), camera_name(camera_name), target_quat(target_quat) {};
};

class AirsimROSWrapper
{
public:

    AirsimROSWrapper(const std::shared_ptr<rclcpp::Node>& nh, const std::shared_ptr<rclcpp::Node>& nh_private, const std::string & host_ip);
    ~AirsimROSWrapper() {}; 

    void initialize_airsim();
    void initialize_ros();

    
    //function from drone.h in MavBench
        geometry_msgs::msg::Pose pose();

        float get_yaw();
        bool set_yaw(int y);
        bool fly_velocity(double vx, double vy, double vz, float yaw = YAW_UNCHANGED, double duration = 3);
        // *** F:DN Drone parameters functions
        float maxYawRate();
        float maxYawRateDuringFlight();
        bool set_yaw_at_z(int y, double z);
        MultirotorState getMultirotorState();

    // function added by feiyang jin
        void takeoff_jin();
        void moveTo(float x, float y, float z, float velocity);
        void moveOnPath(const std::vector<Vector3r>& path, float velocity);
        void hover();
        bool end();
        Vector3r getPosition();
        

    // std::vector<rclcpp::CallbackQueue> callback_queues_;
    rclcpp::AsyncSpinner img_async_spinner_;
    rclcpp::AsyncSpinner lidar_async_spinner_;
    bool is_used_lidar_timer_cb_queue_;
    bool is_used_img_timer_cb_queue_;

    ros::Time first_imu_ros_ts;
    int64_t first_imu_unreal_ts = -1;

private:
    // from drone.h
        float max_yaw_rate = 90.0;
        float max_yaw_rate_during_flight = 90.0;

    /// ROS timer callbacks
    void img_response_timer_cb(const ros::TimerEvent& event); // update images from airsim_client_ every nth sec
    void drone_state_timer_cb(const ros::TimerEvent& event); // update drone state from airsim_client_ every nth sec
    void lidar_timer_cb(const ros::TimerEvent& event);

    /// ROS subscriber callbacks
    void vel_cmd_world_frame_cb(const cinematography_msgs::msg::VelCmd::ConstPtr& msg, const std::string& vehicle_name);
    void vel_cmd_body_frame_cb(const cinematography_msgs::msg::VelCmd::ConstPtr& msg, const std::string& vehicle_name);

    void vel_cmd_group_body_frame_cb(const cinematography_msgs::msg::VelCmdGroup& msg);
    void vel_cmd_group_world_frame_cb(const cinematography_msgs::msg::VelCmdGroup& msg);

    void vel_cmd_all_world_frame_cb(const cinematography_msgs::msg::VelCmd& msg);
    void vel_cmd_all_body_frame_cb(const cinematography_msgs::msg::VelCmd& msg);

    // void vel_cmd_body_frame_cb(const cinematography_msgs::msg::VelCmd& msg, const std::string& vehicle_name);
    void gimbal_angle_quat_cmd_cb(const cinematography_msgs::msg::GimbalAngleQuatCmd& gimbal_angle_quat_cmd_msg);
    void gimbal_angle_euler_cmd_cb(const cinematography_msgs::msg::GimbalAngleEulerCmd& gimbal_angle_euler_cmd_msg);

    rclcpp::Time make_ts(uint64_t unreal_ts);
    // void set_zero_vel_cmd();

    /// ROS service callbacks
    bool takeoff_srv_cb(cinematography_msgs::srv::Takeoff::Request& request, cinematography_msgs::srv::Takeoff::Response& response, const std::string& vehicle_name);
    bool takeoff_group_srv_cb(cinematography_msgs::srv::TakeoffGroup::Request& request, cinematography_msgs::srv::TakeoffGroup::Response& response);
    bool takeoff_all_srv_cb(cinematography_msgs::srv::Takeoff::Request& request, cinematography_msgs::srv::Takeoff::Response& response);
    bool land_srv_cb(cinematography_msgs::srv::Land::Request& request, cinematography_msgs::srv::Land::Response& response, const std::string& vehicle_name);
    bool land_group_srv_cb(cinematography_msgs::srv::LandGroup::Request& request, cinematography_msgs::srv::LandGroup::Response& response);
    bool land_all_srv_cb(cinematography_msgs::srv::Land::Request& request, cinematography_msgs::srv::Land::Response& response);
    bool reset_srv_cb(cinematography_msgs::srv::Reset::Request& request, cinematography_msgs::srv::Reset::Response& response);

    /// ROS tf broadcasters
    void publish_camera_tf(const ImageResponse& img_response, const ros::Time& ros_time, const std::string& frame_id, const std::string& child_frame_id);
    void publish_odom_tf(const nav_msgs::Odometry& odom_ned_msg);

    /// camera helper methods
    sensor_msgs::CameraInfo generate_cam_info(const std::string& camera_name, const CameraSetting& camera_setting, const CaptureSetting& capture_setting) const;
    cv::Mat manual_decode_depth(const ImageResponse& img_response) const;

    sensor_msgs::ImagePtr get_img_msg_from_response(const ImageResponse& img_response, const rclcpp::Time curr_ros_time, const std::string frame_id);
    sensor_msgs::ImagePtr get_depth_img_msg_from_response(const ImageResponse& img_response, const rclcpp::Time curr_ros_time, const std::string frame_id);
    
    void process_and_publish_img_response(const std::vector<ImageResponse>& img_response_vec, const int img_response_idx, const std::string& vehicle_name);

    // methods which parse setting json ang generate ros pubsubsrv
    void create_ros_pubs_from_settings_json();
    void append_static_camera_tf(const std::string& vehicle_name, const std::string& camera_name, const CameraSetting& camera_setting);
    void append_static_lidar_tf(const std::string& vehicle_name, const std::string& lidar_name, const LidarSetting& lidar_setting);
    void append_static_vehicle_tf(const std::string& vehicle_name, const VehicleSetting& vehicle_setting);
    void set_nans_to_zeros_in_pose(VehicleSetting& vehicle_setting) const;
    void set_nans_to_zeros_in_pose(const VehicleSetting& vehicle_setting, CameraSetting& camera_setting) const;
    void set_nans_to_zeros_in_pose(const VehicleSetting& vehicle_setting, LidarSetting& lidar_setting) const;

    /// utils. todo parse into an Airlib<->ROS conversion class
    tf2::Quaternion get_tf2_quat(const msr::airlib::Quaternionr& airlib_quat) const;
    msr::airlib::Quaternionr get_airlib_quat(const geometry_msgs::Quaternion& geometry_msgs_quat) const;
    msr::airlib::Quaternionr get_airlib_quat(const tf2::Quaternion& tf2_quat) const;

    nav_msgs::Odometry get_odom_msg_from_airsim_state(const msr::airlib::MultirotorState& drone_state) const;
    cinematography_msgs::msg::GPSYaw get_gps_msg_from_airsim_geo_point(const msr::airlib::GeoPoint& geo_point) const;
    sensor_msgs::NavSatFix get_gps_sensor_msg_from_airsim_geo_point(const msr::airlib::GeoPoint& geo_point) const;
    sensor_msgs::Imu get_imu_msg_from_airsim(const msr::airlib::ImuBase::Output& imu_data);
    sensor_msgs::PointCloud2 get_lidar_msg_from_airsim(const msr::airlib::LidarData& lidar_data) const;

    // not used anymore, but can be useful in future with an unreal camera calibration environment
    void read_params_from_yaml_and_fill_cam_info_msg(const std::string& file_name, sensor_msgs::CameraInfo& cam_info) const;
    void convert_yaml_to_simple_mat(const YAML::Node& node, SimpleMatrix& m) const; // todo ugly

private:
    // subscriber / services for ALL robots
    rclcpp::Service<cinematography_msgs::srv::Takeoff> vel_cmd_all_body_frame_sub_;
    rclcpp::Service<cinematography_msgs::srv::Takeoff> vel_cmd_all_world_frame_sub_;
    rclcpp::Service<cinematography_msgs::srv::Takeoff> takeoff_all_srvr_;
    rclcpp::Service<cinematography_msgs::srv::Land> land_all_srvr_;

    // todo - subscriber / services for a GROUP of robots, which is defined by a list of `vehicle_name`s passed in the ros msg / srv request
    rclcpp::Subscription<cinematography_msgs::msg::VelCmdGroup> vel_cmd_group_body_frame_sub_;
    rclcpp::Subscription<cinematography_msgs::msg::VelCmdGroup> vel_cmd_group_world_frame_sub_;
    rclcpp::Service<cinematography_msgs::srv::TakeoffGroup> takeoff_group_srvr_;
    rclcpp::Service<cinematography_msgs::srv::LandGroup> land_group_srvr_;

    // utility struct for a SINGLE robot
    struct MultiRotorROS
    {
        std::string vehicle_name;

        /// All things ROS
        rclcpp::Publisher<nav_msgs::msg::Odometry> odom_local_ned_pub;
        rclcpp::Publisher<sensor_msgs::msg::NavSatFix> global_gps_pub;
        // rclcpp::Publisher home_geo_point_pub_; // geo coord of unreal origin

        rclcpp::Subscription<cinematography_msgs::msg::VelCmd> vel_cmd_body_frame_sub;
        rclcpp::Subscription<cinematography_msgs::msg::VelCmd> vel_cmd_world_frame_sub;

        rclcpp::Service<cinematography_msgs::srv::Takeoff> takeoff_srvr;
        rclcpp::Service<cinematography_msgs::srv::Land> land_srvr;

        /// State
        msr::airlib::MultirotorState curr_drone_state;
        // bool in_air_; // todo change to "status" and keep track of this
        nav_msgs::msg::Odometry curr_odom_ned;
        sensor_msgs::msg::NavSatFix gps_sensor_msg;
        bool has_vel_cmd;
        VelCmd vel_cmd;

        std::string odom_frame_id;
        /// Status
        // bool in_air_; // todo change to "status" and keep track of this
        // bool is_armed_;
        // std::string mode_;
    };

    rclcpp::Service<cinematography_msgs::srv::Reset> reset_srvr_;
    rclcpp::Publisher<cinematography_msgs::msg::GPSYaw> origin_geo_point_pub_; // home geo coord of drones
    msr::airlib::GeoPoint origin_geo_point_;// gps coord of unreal origin 
    cinematography_msgs::msg::GPSYaw origin_geo_point_msg_; // todo duplicate

    std::vector<MultiRotorROS> multirotor_ros_vec_;

    std::vector<string> vehicle_names_;
    std::vector<VehicleSetting> vehicle_setting_vec_;
    AirSimSettingsParser airsim_settings_parser_;
    std::unordered_map<std::string, int> vehicle_name_idx_map_;
    static const std::unordered_map<int, std::string> image_type_int_to_string_map_;
    std::map<std::string, std::string> vehicle_imu_map_;
    std::map<std::string, std::string> vehicle_lidar_map_;
    std::vector<geometry_msgs::msg::TransformStamped> static_tf_msg_vec_;
    bool is_vulkan_; // rosparam obtained from launch file. If vulkan is being used, we BGR encoding instead of RGB

    msr::airlib::MultirotorRpcLibClient airsim_client_;
    msr::airlib::MultirotorRpcLibClient airsim_client_images_;
    msr::airlib::MultirotorRpcLibClient airsim_client_lidar_;

    std::shared_ptr<rclcpp::Node> nh_;
    std::shared_ptr<rclcpp::Node> nh_private_;

    // todo not sure if async spinners shuold be inside this class, or should be instantiated in airsim_node.cpp, and cb queues should be public
    // todo for multiple drones with multiple sensors, this won't scale. make it a part of MultiRotorROS?
    rclcpp::CallbackQueue img_timer_cb_queue_;
    rclcpp::CallbackQueue lidar_timer_cb_queue_;

    // todo race condition
    std::recursive_mutex drone_control_mutex_;
    // std::recursive_mutex img_mutex_;
    // std::recursive_mutex lidar_mutex_;

    // gimbal control
    bool has_gimbal_cmd_;
    GimbalCmd gimbal_cmd_; 

    /// ROS tf
    std::string world_frame_id_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    tf2_ros::StaticTransformBroadcaster static_tf_pub_;
    tf2_ros::Buffer tf_buffer_;

    /// ROS params
    double vel_cmd_duration_;

    /// ROS Timers.
    rclcpp::WallTimer<rclcpp::TimerCallbackType> airsim_img_response_timer_;
    rclcpp::WallTimer<rclcpp::TimerCallbackType> airsim_control_update_timer_;
    rclcpp::WallTimer<rclcpp::TimerCallbackType> airsim_lidar_update_timer_;

    typedef std::pair<std::vector<ImageRequest>, std::string> airsim_img_request_vehicle_name_pair;
    std::vector<airsim_img_request_vehicle_name_pair> airsim_img_request_vehicle_name_pair_vec_;
    std::vector<image_transport::Publisher> image_pub_vec_; 
    std::vector<rclcpp::Publisher<sensor_msgs::msg::CameraInfo>> cam_info_pub_vec_;
    std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> lidar_pub_vec_;
    std::vector<rclcpp::Publisher<sensor_msgs::msg::Imu>> imu_pub_vec_;

    std::vector<sensor_msgs::msg::CameraInfo> camera_info_msg_vec_;

    /// ROS other publishers
    rclcpp::Publisher<rosgraph_msgs::Clock> clock_pub_;

    rclcpp::Subscription<airsim_ros_pkgs::msg::GimbalAngleQuatCmd> gimbal_angle_quat_cmd_sub_;
    rclcpp::Subscription<airsim_ros_pkgs::msg::GimbalAngleEulerCmd> gimbal_angle_euler_cmd_sub_;

    static constexpr char CAM_YML_NAME[]    = "camera_name";
    static constexpr char WIDTH_YML_NAME[]  = "image_width";
    static constexpr char HEIGHT_YML_NAME[] = "image_height";
    static constexpr char K_YML_NAME[]      = "camera_matrix";
    static constexpr char D_YML_NAME[]      = "distortion_coefficients";
    static constexpr char R_YML_NAME[]      = "rectification_matrix";
    static constexpr char P_YML_NAME[]      = "projection_matrix";
    static constexpr char DMODEL_YML_NAME[] = "distortion_model";

};