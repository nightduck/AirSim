#include "rclcpp/rclcpp.hpp"
#include <boost/make_shared.hpp>
#include "builtin_interfaces/msg/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/buffer.h>

using namespace std::chrono_literals;
using std::placeholders::_1;

class AirsimROS2Wrapper : public rclcpp::Node {
private:
    msr::airlib::MultirotorRpcLibClient* airsim_client;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_camera;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr actorPose;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr actorAndCamPose;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr actorAndLidarPose;

    tf2_ros::TransformBroadcaster tf_broadcaster;
    //tf2_ros::StaticTransformBroadcaster static_tf_broadcaster;

    rclcpp::TimerBase::SharedPtr timer_img;
    rclcpp::TimerBase::SharedPtr timer_lidar;
    rclcpp::TimerBase::SharedPtr timer_pose;

    std::recursive_mutex drone_control_mutex_;

    std::string camera_name;
    std::string lidar_name;
    std::string vehicle_name;
    int camera_fps;
    int lidar_fps;
    int pose_update;

    std::string world_frame_id_;
    bool is_vulkan_;

    std::string airsim_hostname;

    rclcpp::Clock clock = rclcpp::Clock(RCL_SYSTEM_TIME);
    rclcpp::Time first_ros_ts;
    uint64_t first_unreal_ts = -1;
    
    rclcpp::Time make_ts(uint64_t unreal_ts) {
        if (first_unreal_ts < 0) {
            first_unreal_ts = unreal_ts;
            first_ros_ts = clock.now();
        }
        return first_ros_ts + rclcpp::Duration(unreal_ts- first_unreal_ts);
    }

    void fetchImage() {
        std::unique_lock<std::recursive_mutex> lck(drone_control_mutex_);
        std::vector<ImageCaptureBase::ImageRequest> request = {
            ImageCaptureBase::ImageRequest(camera_name, ImageCaptureBase::ImageType::Scene, false, false),
            ImageCaptureBase::ImageRequest(camera_name, ImageCaptureBase::ImageType::DepthPlanner, true, false)
        };
        std::vector<ImageCaptureBase::ImageResponse> response = airsim_client->simGetImages(request, vehicle_name);
        rclcpp::Time now = this->now();
        lck.unlock();

        sensor_msgs::msg::Image::SharedPtr img_msg_ptr;
        cv::Mat depth_img(response[1].height, response[1].width, CV_32FC1, cv::Scalar(0));
        int img_width = response[1].width;
        for (int row = 0; row < response[1].height; row++)
            for (int col = 0; col < img_width; col++)
                depth_img.at<float>(row, col) = response[1].image_data_float[row * img_width + col];
        img_msg_ptr = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", depth_img).toImageMsg();
        img_msg_ptr->header.stamp = now;
        img_msg_ptr->header.frame_id = response[0].camera_name + "_depth";

        depth_camera->publish(img_msg_ptr);


        img_msg_ptr->data = response[0].image_data_uint8;
        img_msg_ptr->step = response[0].width * 3; // todo un-hardcode. image_width*num_bytes
        img_msg_ptr->header.stamp = now;
        img_msg_ptr->header.frame_id = response[0].camera_name;
        img_msg_ptr->height = response[0].height;
        img_msg_ptr->width = response[0].width;
        img_msg_ptr->encoding = "bgr8";
        if (is_vulkan_)
            img_msg_ptr->encoding = "rgb8";
        img_msg_ptr->is_bigendian = 0;

        camera->publish(img_msg_ptr);
    }

    void fetchLidarCloud() {
        std::unique_lock<std::recursive_mutex> lck(drone_control_mutex_);
        auto lidar_data = airsim_client->getLidarData(lidar_name, vehicle_name); // airsim api is imu_name, vehicle_name
        lck.unlock();

        sensor_msgs::msg::PointCloud2 lidar_msg;
        lidar_msg.header.stamp = this->now();
        lidar_msg.header.frame_id = world_frame_id_; // todo

        if (lidar_data.point_cloud.size() > 3)
        {
            lidar_msg.height = 1;
            lidar_msg.width = lidar_data.point_cloud.size() / 3;

            lidar_msg.fields.resize(3);
            lidar_msg.fields[0].name = "x"; 
            lidar_msg.fields[1].name = "y"; 
            lidar_msg.fields[2].name = "z";
            int offset = 0;

            for (size_t d = 0; d < lidar_msg.fields.size(); ++d, offset += 4)
            {
                lidar_msg.fields[d].offset = offset;
                lidar_msg.fields[d].datatype = sensor_msgs::msg::PointField::FLOAT32;
                lidar_msg.fields[d].count  = 1;
            }

            lidar_msg.is_bigendian = false;
            lidar_msg.point_step = offset; // 4 * num fields
            lidar_msg.row_step = lidar_msg.point_step * lidar_msg.width;

            lidar_msg.is_dense = true; // todo
            std::vector<float> data_std = lidar_data.point_cloud;

            const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&data_std[0]);
            vector<unsigned char> lidar_msg_data(bytes, bytes + sizeof(float) * data_std.size());
            lidar_msg.data = std::move(lidar_msg_data);
        }

        lidar->publish(lidar_msg);
    }

    void fetchPosition() {
        std::unique_lock<std::recursive_mutex> lck(drone_control_mutex_);
        msr::airlib::Pose p = airsim_client->simGetVehiclePose(vehicle_name);
        msr::airlib::CameraInfo c = airsim_client->simGetCameraInfo(camera_name, vehicle_name);
        msr::airlib::Pose l = airsim_client->getLidarData().pose;
        rclcpp::Time now = this->now();
        lck.unlock();

        // Publish transform for drone
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = now;
        transform.header.frame_id = world_frame_id_;
        transform.child_frame_id = vehicle_name;
        transform.transform.translation.x = p.position.x();
        transform.transform.translation.y = p.position.y();
        transform.transform.translation.z = p.position.z();
        transform.transform.rotation.w = p.orientation.w();
        transform.transform.rotation.x = p.orientation.x();
        transform.transform.rotation.y = p.orientation.y();
        transform.transform.rotation.z = p.orientation.z();
        tf_broadcaster.sendTransform(transform);

        // Publish transform for camera
        transform.header.stamp = now;
        transform.header.frame_id = world_frame_id_;
        transform.child_frame_id = vehicle_name + "/" + camera_name;
        transform.transform.translation.x = c.pose.position.x();
        transform.transform.translation.y = c.pose.position.y();
        transform.transform.translation.z = c.pose.position.z();
        transform.transform.rotation.w = c.pose.orientation.w();
        transform.transform.rotation.x = c.pose.orientation.x();
        transform.transform.rotation.y = c.pose.orientation.y();
        transform.transform.rotation.z = c.pose.orientation.z();
        tf_broadcaster.sendTransform(transform);
        
        // Publish transform for lidar
        transform.header.stamp = now;
        transform.header.frame_id = world_frame_id_;
        transform.child_frame_id = vehicle_name + "/" + lidar_name;
        transform.transform.translation.x = l.position.x();
        transform.transform.translation.y = l.position.y();
        transform.transform.translation.z = l.position.z();
        transform.transform.rotation.w = l.orientation.w();
        transform.transform.rotation.x = l.orientation.x();
        transform.transform.rotation.y = l.orientation.y();
        transform.transform.rotation.z = l.orientation.z();
        tf_broadcaster.sendTransform(transform);
    }

public:
    AirsimROS2Wrapper() : Node("airsim_ros2_wrapper"), tf_broadcaster(this) {
        declare_parameter<std::string>("airsim_hostname", "localhost");
        get_parameter("airsim_hostname", airsim_hostname);
        declare_parameter<std::string>("camera_name", "front_center_custom");
        get_parameter("camera_name", camera_name);
        declare_parameter<std::string>("lidar_name", "LidarCustom");
        get_parameter("lidar_name", lidar_name);
        declare_parameter<std::string>("vehicle_name", "drone_1");
        get_parameter("vehicle_name", vehicle_name);
        declare_parameter<std::string>("world_frame", "world_ned");
        get_parameter("world_frame", world_frame_id_);
        declare_parameter<int>("camera_fps", 4);
        get_parameter("camera_fps", camera_fps);
        declare_parameter<int>("lidar_fps", 4);
        get_parameter("lidar_fps", lidar_fps);
        declare_parameter<int>("pose_update_rate", 8);
        get_parameter("pose_update_rate", pose_update);
        get_parameter("is_vulkan", is_vulkan_);

        airsim_client = new msr::airlib::MultirotorRpcLibClient(airsim_hostname);

        camera = this->create_publisher<sensor_msgs::msg::Image>("camera", 10);
        depth_camera = this->create_publisher<sensor_msgs::msg::Image>("camera/depth", 10);
        lidar = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar", 10);
        timer_img = create_wall_timer(std::chrono::milliseconds(1000)/camera_fps,
                std::bind(&AirsimROS2Wrapper::fetchImage, this));
        timer_lidar = create_wall_timer(std::chrono::milliseconds(1000)/lidar_fps,
                std::bind(&AirsimROS2Wrapper::fetchLidarCloud, this));
        timer_pose = create_wall_timer(std::chrono::milliseconds(1000)/pose_update,
                std::bind(&AirsimROS2Wrapper::fetchPosition, this));

        // TODO: Get this to work. It's inefficient to keep publishing the lidar transform when it never changes
        // // Fetch initial position (to set transforms)
        // fetchPosition();

        // // Publish transform for lidar (fixed)
        // msr::airlib::Pose lp = airsim_client->getLidarData().pose;
        // geometry_msgs::msg::TransformStamped transform;
        // transform.header.stamp = this->now();
        // transform.header.frame_id = vehicle_name;
        // transform.child_frame_id = lidar_name;
        // transform.transform.translation.x = lp.position.x();
        // transform.transform.translation.y = lp.position.y();
        // transform.transform.translation.z = lp.position.z();
        // transform.transform.rotation.w = lp.orientation.w();
        // transform.transform.rotation.x = lp.orientation.x();
        // transform.transform.rotation.y = lp.orientation.y();
        // transform.transform.rotation.z = lp.orientation.z();
        // static_tf_broadcaster.sendTransform(transform);
    }
    
    ~AirsimROS2Wrapper() {
        delete airsim_client;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor exec;
    auto ros2wrapper = std::make_shared<AirsimROS2Wrapper>();
    exec.add_node(ros2wrapper);
    exec.spin();
    rclcpp::shutdown();

    return 0;
}