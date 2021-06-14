#include "rclcpp/rclcpp.hpp"
#include "cinematography_msgs/msg/gimbal_angle_quat_cmd.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include "cinematography_msgs/msg/bounding_box.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/accel.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include <math.h>

#include <iostream>
#include <vector>
#include "tkDNN/tkdnn.h"
#include "tkDNN/Yolo3Detection.h"

using namespace std::chrono_literals;
using std::placeholders::_1;

#define CV_RES cv::Size(192, 192)
#define ACTOR_CLASS 0
#define PROB_THRESHHOLD 0.8
#define GIMBAL_DAMPING_FACTOR   0.8       // Don't immediately face actor, but ease camera in their direction. That way, false positive don't swivel the camera so real actor's out of frame

FILE *fd;

struct odometry {
    geometry_msgs::msg::Pose pose;
    geometry_msgs::msg::Twist vel;
    geometry_msgs::msg::Accel acc;
    rclcpp::Time timestamp;
};

class ActorDetection : public rclcpp::Node {
private:
    rclcpp::Publisher<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_pub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub;

    msr::airlib::Vector3r camera_offset;

    tf2_ros::Buffer* tf_buffer;
    tf2_ros::TransformListener* tf_listener;

    struct odometry odom;

    cv::Mat last_depth_img;

    rclcpp::TimerBase::SharedPtr timer_infr;
    rclcpp::TimerBase::SharedPtr timer_img;
    msr::airlib::MultirotorRpcLibClient* airsim_client;
    tf2::Quaternion gimbal_setpoint;
    std::string world_frame = "world_ned";
    std::string vehicle_name = "drone_1";
    std::string camera_name = "front_center_custom";
    std::string pose_frame;
    float fov = M_PI/2;
    int gimbal_msg_count = 0;

    tk::dnn::Yolo3Detection *detNN;
    tk::dnn::Network *net;
    std::vector<tk::dnn::Layer*> outputs;

    std::string trt_engine_filename;
    std::string airsim_hostname;

    struct odometry updateOdometry(struct odometry old_state, geometry_msgs::msg::PoseStamped new_point_stamped, rclcpp::Duration t_diff) {
        struct odometry temp;
        geometry_msgs::msg::Pose new_point = new_point_stamped.pose;
        temp.pose = new_point;

        tf2::Quaternion quat;
        geometry_msgs::msg::Vector3 rpy1, rpy2;
        tf2::fromMsg(temp.pose.orientation, quat);
        tf2::Matrix3x3(quat).getRPY(rpy1.x, rpy1.y, rpy1.z);
        tf2::fromMsg(new_point.orientation, quat);
        tf2::Matrix3x3(quat).getRPY(rpy2.x, rpy2.y, rpy2.z);

        temp.vel.linear.x = (new_point.position.x - old_state.pose.position.x) / t_diff.seconds();
        temp.vel.linear.y = (new_point.position.y - old_state.pose.position.y) / t_diff.seconds();
        temp.vel.linear.z = (new_point.position.z - old_state.pose.position.z) / t_diff.seconds();
        temp.vel.angular.x = (rpy2.x - rpy2.x) / t_diff.seconds();
        temp.vel.angular.y = (rpy2.y - rpy2.y) / t_diff.seconds();
        temp.vel.angular.z = (rpy2.z - rpy2.z) / t_diff.seconds();

        temp.acc.linear.x = (temp.vel.linear.x - old_state.vel.linear.x) / t_diff.seconds();
        temp.acc.linear.y = (temp.vel.linear.y - old_state.vel.linear.y) / t_diff.seconds();
        temp.acc.linear.z = (temp.vel.linear.z - old_state.vel.linear.z) / t_diff.seconds();
        temp.acc.angular.x = (temp.vel.angular.x - old_state.vel.angular.x) / t_diff.seconds();
        temp.acc.angular.y = (temp.vel.angular.y - old_state.vel.angular.y) / t_diff.seconds();
        temp.acc.angular.z = (temp.vel.angular.z - old_state.vel.angular.z) / t_diff.seconds();

        temp.timestamp = new_point_stamped.header.stamp;

        return temp;
    }

    void getDepthImage(const sensor_msgs::msg::Image::SharedPtr msg) {
        fprintf( fd, "actor_detection_getDepthImage_entry" );
        fflush( fd );

        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        last_depth_img = cv_ptr->image;

        fprintf( fd, "actor_detection_getDepthImage_exit" );
        fflush( fd );
    }

    void fetchImage(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (last_depth_img.cols == 0 || last_depth_img.rows == 0) {
            return;
        }

        fprintf( fd, "actor_detection_fetchImage_entry" );
        fflush( fd );

        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat img = cv_ptr->image;

        cinematography_msgs::msg::BoundingBox bb;   // Contains subimage, position in frame, and camera pose at the moment the image was taken
        geometry_msgs::msg::TransformStamped transform = tf_buffer->lookupTransform(world_frame, pose_frame, tf2::TimePointZero, std::chrono::milliseconds(100)); // TODO: Make sure inference + this block doesn't exceed period

        geometry_msgs::msg::PoseStamped temp;        
        temp.pose.position.x = transform.transform.translation.x;
        temp.pose.position.y = transform.transform.translation.y;
        temp.pose.position.z = transform.transform.translation.z;
        temp.pose.orientation = transform.transform.rotation;
        temp.header.stamp = transform.header.stamp;

        odom = updateOdometry(odom, temp, rclcpp::Time(transform.header.stamp) - odom.timestamp);

        bb.drone_pose = odom.pose;
        bb.drone_vel = odom.vel;
        bb.drone_acc = odom.acc;

        // Perform inference
        std::vector<cv::Mat> frame_array;
        frame_array.push_back(img.clone());
        detNN->update(frame_array);

        // // Draw on boxes and publish for debugging purposes
        // frame_array.clear();
        // frame_array.push_back(img.clone());
        // detNN->draw(frame_array);
        cv_bridge::CvImage cv_msg;
        cv_msg.encoding = sensor_msgs::image_encodings::BGR8;
        // cv_msg.header.frame_id = "world_ned";
        // cv_msg.header.stamp = this->now();
        // cv_msg.image = frame_array[0];
        // objdet_pub->publish(cv_msg.toImageMsg());

        // Sort through found objects, and pick the one that's most likely our actor
        tk::dnn::box mostLikely;
        mostLikely.cl = -1;
        mostLikely.prob = 0;
        for(auto b: detNN->detected) {
            if (b.cl == ACTOR_CLASS && b.prob > mostLikely.prob) {
                mostLikely = b;
            }
        }

        float centerx, centery, width, height;

        // If an actor wasn't found, just return without passing anything down the pipeline
        // TODO: Combine probability, size of window, and proximity to center as a combined heuristic
        if (mostLikely.cl < 0 || mostLikely.prob < PROB_THRESHHOLD) {
            // TODO: If nothing seen, drift towards center
            // msr::airlib::Quaternionr q = airsim_client->simGetCameraInfo(camera_name, vehicle_name).pose.orientation;
            // tf2::Quaternion tfq = tf2::Quaternion(q.x(), q.y(), q.z(), q.w());
            // tfq.slerp(tf2::Quaternion(0,0,0,1), 0.1);
            // q = msr::airlib::Quaternionr(tfq.w(), tfq.x(), tfq.y(), tfq.z());
            //airsim_client->simSetCameraOrientation(camera_name, q, vehicle_name);

            bb.fov = fov;
            bb.centerx = 0.5;
            bb.centery = 0.5;
            bb.width = 0;
            bb.height = 0;
            cv_msg.image = cv::Mat();
            bb.image = *cv_msg.toImageMsg();
            
            bb_pub->publish(bb);
            fprintf( fd, "heading_estimation_processImage_release" );
            fflush( fd );

            fprintf( fd, "actor_detection_fetchImage_exit no_actor" );
            fflush( fd );

            return;
        }

        centerx = (mostLikely.x + mostLikely.w/2) / img.cols;
        centery = (mostLikely.y + mostLikely.h/2) / img.rows;
        width = mostLikely.w / img.cols;
        height = mostLikely.h / img.rows;

        // Look at depth image and get value of pixel in middle of bounding box
        float distance = last_depth_img.at<float>(centerx * last_depth_img.cols, centery * last_depth_img.rows);

        // If the actor is too small, assume it's incorrect. False positives are often in the distance
        if (width < 0.05 && height < 0.05) {
            bb.fov = fov;
            bb.centerx = 0.5;
            bb.centery = 0.5;
            bb.width = 0;
            bb.height = 0;
            bb.depth = 0;
            cv_msg.image = cv::Mat();
            bb.image = *cv_msg.toImageMsg();

            bb_pub->publish(bb);
            fprintf( fd, "heading_estimation_processImage_release" );
            fflush( fd );

            fprintf( fd, "actor_detection_fetchImage_exit far_actor" );
            fflush( fd );
            return;
        }

        tf2::Quaternion vert_adjustment;
        vert_adjustment.setRPY(0, fov * (0.5 - centery) * GIMBAL_DAMPING_FACTOR, 0);
        gimbal_setpoint = gimbal_setpoint * vert_adjustment;             // Get new gimbal setpoint (so actor is vertically centered)

        msr::airlib::Quaternionr alq = msr::airlib::Quaternionr(gimbal_setpoint.w(),
                        gimbal_setpoint.x(), gimbal_setpoint.y(), gimbal_setpoint.z());
        airsim_client->simSetCameraPose(camera_name, msr::airlib::Pose(camera_offset, alq), vehicle_name);

        // Extract subimage, package with centering coordinates, and publish to /bounding_box
        bb.fov = fov;
        bb.centerx = centerx;
        bb.centery = centery;
        bb.width = width;
        bb.height = height;
        bb.depth = distance;
        // bb.drone_pose was copied in the thread safe portion at the top of this function

        int px_left = (centerx - width/2) * img.cols;
        int px_top = (centery - height/2) * img.rows;
        int px_width = width * img.cols;
        int px_height = height * img.rows;
        cv_msg.image = img;
        try {
            cv::Mat cropped = cv_msg.image(cv::Rect((px_left < 0) ? 0 : px_left, (px_top < 0) ? 0 : px_top,
                    (px_width + px_left > img.cols) ? img.cols - px_left : px_width,
                    (px_height + px_top > img.rows) ? img.rows - px_top : px_height));
            cv_msg.image = cropped;
            bb.image = *cv_msg.toImageMsg();

            bb_pub->publish(bb);
            fprintf( fd, "heading_estimation_processImage_release" );
            fflush( fd );
        } catch (cv::Exception& err) {
            const char* err_msg = err.what();
            std::cout << "exception caught: " << err_msg << std::endl;
        }

        fprintf( fd, "actor_detection_fetchImage_exit" );
        fflush( fd );

        return;
    }

public:
    ActorDetection() : Node("actor_detection") {
        declare_parameter<std::string>("tensorrt_engine", "yolo4_deer_fp32.rt");
        get_parameter("tensorrt_engine", trt_engine_filename);

        auto parameters_client = std::make_shared<rclcpp::SyncParametersClient>(this, "airsim_ros2_wrapper");
        while (!parameters_client->wait_for_service(1s)) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "service not available, waiting again...");
        }
        airsim_hostname = parameters_client->get_parameter<std::string>("airsim_hostname");
        vehicle_name = parameters_client->get_parameter<std::string>("vehicle_name");
        camera_name = parameters_client->get_parameter<std::string>("camera_name");
        world_frame = parameters_client->get_parameter<std::string>("world_frame");
        int fps = parameters_client->get_parameter<int>("camera_fps");

        tf_buffer = new tf2_ros::Buffer(this->get_clock()); 
        tf_listener = new tf2_ros::TransformListener(*tf_buffer);

        while(!tf_buffer->_frameExists("world_ned")) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Waiting for world frame...");
            sleep(1);
        };


        pose_frame = vehicle_name + "/" + camera_name;

        geometry_msgs::msg::TransformStamped transform = tf_buffer->lookupTransform(world_frame, pose_frame, tf2::TimePointZero, std::chrono::milliseconds(100)); // TODO: Make sure inference + this block doesn't exceed period
        geometry_msgs::msg::PoseStamped temp;        
        temp.pose.position.x = transform.transform.translation.x;
        temp.pose.position.y = transform.transform.translation.y;
        temp.pose.position.z = transform.transform.translation.z;
        temp.pose.orientation = transform.transform.rotation;
        temp.header.stamp = transform.header.stamp;
        odom = updateOdometry(odom, temp, rclcpp::Duration(1000000000/fps)); // Populate odom initially, use estimated time diff for 3rd argument
        
        airsim_client = new msr::airlib::MultirotorRpcLibClient(airsim_hostname);
        airsim_client->confirmConnection();
        gimbal_setpoint = tf2::Quaternion(0,0,0,1);
        bb_pub = this->create_publisher<cinematography_msgs::msg::BoundingBox>("bounding_box", 50);
        depth_sub = this->create_subscription<sensor_msgs::msg::Image>("camera/depth", 1, std::bind(&ActorDetection::getDepthImage, this, _1));
        camera_sub = this->create_subscription<sensor_msgs::msg::Image>("camera", 1, std::bind(&ActorDetection::fetchImage, this, _1));

        //netRT = new tk::dnn::NetworkRT(NULL, "hde_deer_airsim.rt");
        detNN = new tk::dnn::Yolo3Detection();
        if (!detNN->init(trt_engine_filename, 2, 1)) {
            RCLCPP_ERROR(this->get_logger(), "Cannot find yolo4_airsim.rt! Please place in present directory");
        };

        msr::airlib::CameraInfo ci = airsim_client->simGetCameraInfo(camera_name, vehicle_name);
        fov = ci.fov * M_PI / 180;

        camera_offset = ci.pose.position;
    }

    ~ActorDetection() {
        delete net;
        delete airsim_client;
    }
};

int main(int argc, char **argv) {
    fd = fopen("/sys/kernel/debug/tracing/trace_marker", "a");
    if (fd == NULL) {
        perror("Could not open trace marker");
        return -1;
    }

    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor exec;
    auto actor_detection = std::make_shared<ActorDetection>();
    exec.add_node(actor_detection);
    exec.spin();
    rclcpp::shutdown();

    return 0;
}
