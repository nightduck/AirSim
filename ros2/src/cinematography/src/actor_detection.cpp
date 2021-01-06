#include "rclcpp/rclcpp.hpp"
#include "cinematography_msgs/msg/gimbal_angle_quat_cmd.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include "cinematography_msgs/msg/bounding_box.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/pose.hpp"
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
#define PERIOD 250ms

class ActorDetection : public rclcpp::Node {
private:
    rclcpp::Publisher<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_pub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub;

    rclcpp::TimerBase::SharedPtr timer_infr;
    rclcpp::TimerBase::SharedPtr timer_img;
    msr::airlib::MultirotorRpcLibClient* airsim_client;
    tf2::Quaternion gimbal_setpoint;
    std::string camera_name = "front_center_custom";  // TODO: Set these 3 as parameters
    std::string vehicle_name = "drone_1";
    float fov = M_PI/2;
    int gimbal_msg_count = 0;

    geometry_msgs::msg::Pose drone_pose;
    std::mutex m;

    tk::dnn::Yolo3Detection *detNN;
    tk::dnn::Network *net;
    std::vector<tk::dnn::Layer*> outputs;

    std::string trt_engine_filename;
    std::string airsim_hostname;

    void fetchPose(const geometry_msgs::msg::Pose::SharedPtr& msg) {
        m.lock();
        drone_pose.orientation = msg->orientation;
        drone_pose.position = msg->position;
        m.unlock();
    }

    void fetchImage(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat img = cv_ptr->image;

        cinematography_msgs::msg::BoundingBox bb;   // Contains subimage, position in frame, and camera pose at the moment the image was taken
        bb.drone_pose = drone_pose;

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
            return;
        }

        centerx = (mostLikely.x + mostLikely.w/2) / img.cols;
        centery = (mostLikely.y + mostLikely.h/2) / img.rows;
        width = mostLikely.w / img.cols;
        height = mostLikely.h / img.rows;

        // If the actor is too small, assume it's incorrect. False positives are often in the distance
        if (width < 0.05 && height < 0.05) {
            bb.fov = fov;
            bb.centerx = 0.5;
            bb.centery = 0.5;
            bb.width = 0;
            bb.height = 0;
            cv_msg.image = cv::Mat();
            bb.image = *cv_msg.toImageMsg();
            bb_pub->publish(bb);
            return;
        }

        tf2::Quaternion vert_adjustment;
        vert_adjustment.setRPY(0, fov * (0.5 - centery) * GIMBAL_DAMPING_FACTOR, 0);
        gimbal_setpoint = gimbal_setpoint * vert_adjustment;             // Get new gimbal setpoint (so actor is vertically centered)

        msr::airlib::Quaternionr alq = msr::airlib::Quaternionr(gimbal_setpoint.w(),
                        gimbal_setpoint.x(), gimbal_setpoint.y(), gimbal_setpoint.z());
        airsim_client->simSetCameraOrientation(camera_name, alq, vehicle_name);

        // Extract subimage, package with centering coordinates, and publish to /bounding_box
        bb.fov = fov;
        bb.centerx = centerx;
        bb.centery = centery;
        bb.width = width;
        bb.height = height;
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
        } catch (cv::Exception& err) {
            const char* err_msg = err.what();
            std::cout << "exception caught: " << err_msg << std::endl;
        }

        return;
    }

public:
    ActorDetection() : Node("actor_detection") {
        declare_parameter<std::string>("tensorrt_engine", "yolo4_deer_fp32.rt");
        get_parameter("tensorrt_engine", trt_engine_filename);
        declare_parameter<std::string>("airsim_hostname", "localhost");
        get_parameter("airsim_hostname", airsim_hostname);
        
        airsim_client = new msr::airlib::MultirotorRpcLibClient(airsim_hostname);
        gimbal_setpoint = tf2::Quaternion(0,0,0,1);
        bb_pub = this->create_publisher<cinematography_msgs::msg::BoundingBox>("bounding_box", 50);
        camera_sub = this->create_subscription<sensor_msgs::msg::Image>("camera", 1, std::bind(&ActorDetection::fetchImage, this, _1));

        //netRT = new tk::dnn::NetworkRT(NULL, "hde_deer_airsim.rt");
        detNN = new tk::dnn::Yolo3Detection();
        if (!detNN->init(trt_engine_filename, 2, 1)) {
            RCLCPP_ERROR(this->get_logger(), "Cannot find yolo4_airsim.rt! Please place in present directory");
        };

        msr::airlib::CameraInfo ci = airsim_client->simGetCameraInfo(camera_name, vehicle_name);
        fov = ci.fov * M_PI / 180;
    }

    ~ActorDetection() {
        delete net;
        delete airsim_client;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor exec;
    auto actor_detection = std::make_shared<ActorDetection>();
    exec.add_node(actor_detection);
    exec.spin();
    //rclcpp::spin(actor_detection);
    rclcpp::shutdown();

    return 0;
}
