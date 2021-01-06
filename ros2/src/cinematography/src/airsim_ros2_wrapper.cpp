#include "rclcpp/rclcpp.hpp"
#include <boost/make_shared.hpp>
#include "builtin_interfaces/msg/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

class AirsimROS2Wrapper : public rclcpp::Node {
private:
    msr::airlib::MultirotorRpcLibClient* airsim_client;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr actorPose;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr actorAndCamPose;

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

    std::string world_frame_id_ = "world_ned";
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
            ImageCaptureBase::ImageRequest(camera_name, ImageCaptureBase::ImageType::Scene, false, false)
        };
        ImageCaptureBase::ImageResponse response = airsim_client->simGetImages(request, vehicle_name)[0];
        lck.unlock();

        sensor_msgs::msg::Image img_msg_ptr;
        img_msg_ptr.data = response.image_data_uint8;
        img_msg_ptr.step = response.width * 3; // todo un-hardcode. image_width*num_bytes
        img_msg_ptr.header.stamp = make_ts(response.time_stamp);
        img_msg_ptr.header.frame_id = response.camera_name + "_optical";
        img_msg_ptr.height = response.height;
        img_msg_ptr.width = response.width;
        img_msg_ptr.encoding = "bgr8";
        if (is_vulkan_)
            img_msg_ptr.encoding = "rgb8";
        img_msg_ptr.is_bigendian = 0;

        camera->publish(img_msg_ptr);
    }

    void fetchLidarCloud() {
        std::unique_lock<std::recursive_mutex> lck(drone_control_mutex_);
        auto lidar_data = airsim_client->getLidarData(lidar_name, vehicle_name); // airsim api is imu_name, vehicle_name
        lck.unlock();

        sensor_msgs::msg::PointCloud2 lidar_msg;
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

        lidar_msg.header.frame_id = "world_ned"; // sensor frame name. todo add to doc
        lidar_msg.header.stamp = clock.now();
        lidar->publish(lidar_msg);
    }

    void fetchPosition() {
        std::unique_lock<std::recursive_mutex> lck(drone_control_mutex_);
        msr::airlib::Pose p = airsim_client->simGetVehiclePose(vehicle_name);
        msr::airlib::CameraInfo c = airsim_client->simGetCameraInfo(camera_name, vehicle_name);
        lck.unlock();

        msr::airlib::Pose cp = p + c.pose;

        geometry_msgs::msg::Pose p_msg;
        p_msg.orientation.w = p.orientation.w();
        p_msg.orientation.x = p.orientation.x();
        p_msg.orientation.y = p.orientation.y();
        p_msg.orientation.z = p.orientation.z();
        p_msg.position.x = p.position.x();
        p_msg.position.y = p.position.y();
        p_msg.position.z = p.position.z();

        
        geometry_msgs::msg::Pose cp_msg;
        cp_msg.orientation.w = cp.orientation.w();
        cp_msg.orientation.x = cp.orientation.x();
        cp_msg.orientation.y = cp.orientation.y();
        cp_msg.orientation.z = cp.orientation.z();
        cp_msg.position.x = cp.position.x();
        cp_msg.position.y = cp.position.y();
        cp_msg.position.z = cp.position.z();

        actorPose->publish(p_msg);
        actorAndCamPose->publish(cp_msg);
    }

public:
    AirsimROS2Wrapper() : Node("airsim_ros2_wrapper", "/airsim_ros2_wrapper") {
        declare_parameter<std::string>("airsim_hostname", "localhost");
        get_parameter("airsim_hostname", airsim_hostname);
        declare_parameter<std::string>("camera_name", "front_center_custom");
        get_parameter("camera_name", camera_name);
        declare_parameter<std::string>("lidar_name", "LidarCustom");
        get_parameter("lidar_name", lidar_name);
        declare_parameter<std::string>("vehicle_name", "drone_1");
        get_parameter("vehicle_name", vehicle_name);
        declare_parameter<int>("camera_fps", 4);
        get_parameter("camera_fps", camera_fps);
        declare_parameter<int>("lidar_fps", 4);
        get_parameter("lidar_fps", lidar_fps);
        declare_parameter<int>("pose_update_rate", 8);
        get_parameter("pose_update_rate", pose_update);
        get_parameter("is_vulkan", is_vulkan_);

        airsim_client = new msr::airlib::MultirotorRpcLibClient(airsim_hostname);

        camera = this->create_publisher<sensor_msgs::msg::Image>("camera", 10);
        lidar = this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar", 10);
        actorPose = this->create_publisher<geometry_msgs::msg::Pose>("pose/" + vehicle_name, 20);
        actorAndCamPose = this->create_publisher<geometry_msgs::msg::Pose>("pose/" + vehicle_name + "/" + camera_name, 20);
        timer_img = create_wall_timer(std::chrono::milliseconds(1000)/camera_fps,
                std::bind(&AirsimROS2Wrapper::fetchImage, this));
        timer_lidar = create_wall_timer(std::chrono::milliseconds(1000)/lidar_fps,
                std::bind(&AirsimROS2Wrapper::fetchLidarCloud, this));
        timer_pose = create_wall_timer(std::chrono::milliseconds(1000)/pose_update,
                std::bind(&AirsimROS2Wrapper::fetchPosition, this));
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