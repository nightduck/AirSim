#include "rclcpp/rclcpp.hpp"
#include "cinematography_msgs/msg/gimbal_angle_quat_cmd.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include "cinematography_msgs/msg/bounding_box.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
//#include <darknet/include/darknet.h>
#include <math.h>

#include <iostream>
#include <vector>
#include "tkDNN/tkdnn.h"
#include "tkDNN/Yolo3Detection.h"

using std::placeholders::_1;

#define CV_RES cv::Size(192, 192)
#define ACTOR_CLASS 0
#define PROB_THRESHHOLD 0.8
#define GIMBAL_DAMPING_FACTOR   0.8       // Don't immediately face actor, but ease camera in their direction. That way, false positive don't swivel the camera so real actor's out of frame

class ActorDetection : public rclcpp::Node {
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr objdet_pub;
    rclcpp::Publisher<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_pub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera;
    msr::airlib::MultirotorRpcLibClient* airsim_client;
    tf2::Quaternion gimbal_setpoint;
    std::string camera_name = "front_center_custom";  // TODO: Set these 3 as parameters
    std::string vehicle_name = "drone_1";
    float fov = M_PI/2;
    int gimbal_msg_count = 0;

    tk::dnn::Yolo3Detection *detNN;
    tk::dnn::Network *net;
    std::vector<tk::dnn::Layer*> outputs;

    std::string airsim_hostname;

    int difffilter(const std::vector<uint8_t> &v, std::vector<uint8_t> &w, int m, int n);

    void processImage(const sensor_msgs::msg::Image::SharedPtr msg) {
        // TODO: Add parameter to specify which type our actor is (deer, car, person, etc)
        // TODO: Run image through YOLO model and get bounding box coordiantes of object that is most confidently our actor

        // Perform inference
        cv_bridge::CvImagePtr cv_ptr_in = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        std::vector<cv::Mat> frame_array;
        frame_array.push_back(cv_ptr_in->image);
        detNN->update(frame_array);

        // Draw on boxes and publish for debugging purposes
        frame_array[0] = cv_ptr_in->image;
        detNN->draw(frame_array);
        cv_ptr_in->image = frame_array[0];
        objdet_pub->publish(cv_ptr_in->toImageMsg());

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
            return;
        }

        centerx = (mostLikely.x + mostLikely.w/2) / msg->width;
        centery = (mostLikely.y + mostLikely.h/2) / msg->height;
        width = mostLikely.w / msg->width;
        height = mostLikely.h / msg->height;

        // If the actor is too small, assume it's incorrect. False positives are often in the distance
        if (width < 0.05 && height < 0.05) {
            return;
        }


        // Use centering coordinates to move gimbal so actor is in center frame (OPTIONAL: Add parameter to specify
        // where in frame the actor should be)
        msr::airlib::Quaternionr cq = airsim_client->simGetCameraInfo(camera_name, vehicle_name).pose.orientation;
        tf2::Quaternion horiz_adjustment = tf2::Quaternion(0, 0, sin(fov * (centerx - 0.5) / 2), cos(fov * (centerx - 0.5) / 2));
        tf2::Quaternion vert_adjustment = tf2::Quaternion(0, sin(fov * (centery + height/2 - 0.5) / -2), 0, cos(fov * (centery + height/2 - 0.5) / -2));
        tf2::Quaternion actor_direction = horiz_adjustment * tf2::Quaternion(cq.x(), cq.y(), cq.z(), cq.w()) * vert_adjustment;   // Get the orientation in the direction of the actor (centered on bottom center)

        horiz_adjustment = tf2::Quaternion(0, 0, sin(fov * (centerx - 0.5) / 2 * GIMBAL_DAMPING_FACTOR), cos(fov * (centerx - 0.5) / 2 * GIMBAL_DAMPING_FACTOR));
        vert_adjustment = tf2::Quaternion(0, sin(fov * (centery - 0.5) / -2 * GIMBAL_DAMPING_FACTOR), 0, cos(fov * (centery - 0.5) / -2 * GIMBAL_DAMPING_FACTOR));
        gimbal_setpoint = horiz_adjustment * gimbal_setpoint * vert_adjustment;             // Get new gimbal setpoint (centered on actor)

        msr::airlib::Quaternionr alq = msr::airlib::Quaternionr(gimbal_setpoint.w(),
                        gimbal_setpoint.x(), gimbal_setpoint.y(), gimbal_setpoint.z());
        airsim_client->simSetCameraOrientation(camera_name, alq, vehicle_name);

        // Extract subimage, package with centering coordinates, and publish to /bounding_box
        cinematography_msgs::msg::BoundingBox bb;
        bb.actor_direction = tf2::toMsg(actor_direction);

        int px_left = (centerx - width/2) * msg->width;
        int px_top = (centery - height/2) * msg->height;
        int px_width = width * msg->width;
        int px_height = height * msg->height;
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat cropped = cv_ptr->image(cv::Rect(px_left, px_top, px_width, px_height));
        cv_ptr->image = cropped;
        bb.image = *cv_ptr->toImageMsg();

        bb_pub->publish(bb);
    }

public:
    // TODO: Pass IP address of airsim as parameter
    ActorDetection() : Node("actor_detection") {
        declare_parameter<std::string>("airsim_hostname", "localhost");
        get_parameter("airsim_hostname", airsim_hostname);
        
        airsim_client = new msr::airlib::MultirotorRpcLibClient(airsim_hostname);
        gimbal_setpoint = tf2::Quaternion(0,0,0,1);
        //airsim_client->simGetCameraInfo(camera_name, vehicle_name).pose.orientation;
        bb_pub = this->create_publisher<cinematography_msgs::msg::BoundingBox>("bounding_box", 50);
        objdet_pub = this->create_publisher<sensor_msgs::msg::Image>("obj_detect_output", 50);
        camera = this->create_subscription<sensor_msgs::msg::Image>("camera", 1, std::bind(&ActorDetection::processImage, this, _1));

        // tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, net->getNetworkRTName("yolo4/yolo-deer.rt"));
        detNN = new tk::dnn::Yolo3Detection();
        detNN->init("yolo4_airsim.rt", 2, 1);
    }

    ~ActorDetection() {
        delete net;
        delete airsim_client;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ActorDetection>());
    rclcpp::shutdown();

    return 0;
}
