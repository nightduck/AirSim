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
#include <math.h>

using std::placeholders::_1;

class ActorDetection : public rclcpp::Node {
private:
    rclcpp::Publisher<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_pub;
    rclcpp::Publisher<cinematography_msgs::msg::GimbalAngleQuatCmd>::SharedPtr gimbal_control;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera;
    tf2::Quaternion gimbal_setpoint;
    std::string camera_name = "front_center_custom";
    std::string vehicle_name = "drone_1";
    float fov = M_PI/2;
    int gimbal_msg_count = 0;

    int difffilter(const std::vector<uint8_t> &v, std::vector<uint8_t> &w, int m, int n);

    void processImage(const sensor_msgs::msg::Image::SharedPtr msg) {
        // TODO: Add parameter to specify which type our actor is (deer, car, person, etc)
        // TODO: Run image through YOLO model and get bounding box coordiantes of object that is most confidently our actor

        std::vector<uint8_t> array;
        array.reserve(msg->data.size());
        //difffilter(msg->data, array, msg->width, msg->height);

        //=================DUMMY LOAD==========================
        uint8_t prev = msg->data[0];
        int len = msg->data.size();
        array[0] = prev;
        for (int i = 1; i < 1000000; i++) {
            array[i%len] = msg->data[i%len] * prev;
            prev = array[i%len];
        }
        //==================DUMMY LOAD=========================
        float centerx = array[0] / 512.0 + 0.25;
        float centery = array[1] / 512.0 + 0.25;
        float width = array[2] / 1024.0;
        float height = array[3] / 1024.0;

        //DEBUGGING
        centerx = 0.4;
        centery = 0.501;
        width=0.15;
        height=0.2;

        // Use centering coordinates to move gimbal so actor is in center frame (OPTIONAL: Add parameter to specify
        // where in frame the actor should be)
        tf2::Quaternion horiz_adjustment = tf2::Quaternion(0, 0, sin(fov * (centerx - 0.5) / 2), cos(fov * (centerx - 0.5) / 2));
        tf2::Quaternion vert_adjustment = tf2::Quaternion(sin(fov * centery - height / 2), 0, 0, cos(fov * centery - height / 2));
        tf2::Quaternion actor_direction = horiz_adjustment * gimbal_setpoint * vert_adjustment;   // Get the orientation in the direction of the actor (centered on bottom center)

        vert_adjustment = tf2::Quaternion(sin(fov * (centery - 0.5) / 2), 0, 0, cos(fov * (centery - 0.5) / 2));
        gimbal_setpoint = horiz_adjustment * gimbal_setpoint * vert_adjustment;             // Get new gimbal setpoint (centered on actor)

        cinematography_msgs::msg::GimbalAngleQuatCmd gb_msg;
        gb_msg.header.frame_id = "world_ned";
        gb_msg.header.stamp = this->now();
        gb_msg.camera_name = camera_name;
        gb_msg.vehicle_name = vehicle_name;
        gb_msg.orientation = tf2::toMsg(actor_direction);

        gimbal_control->publish(gb_msg);

        // Extract subimage, package with centering coordinates, and publish to /bounding_box
        cinematography_msgs::msg::BoundingBox bb;
        bb.actor_direction = tf2::toMsg(gimbal_setpoint);

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
    ActorDetection() : Node("actor_detection") {
        bb_pub = this->create_publisher<cinematography_msgs::msg::BoundingBox>("bounding_box", 50);
        gimbal_control = this->create_publisher<cinematography_msgs::msg::GimbalAngleQuatCmd>("gimbal", 1);
        camera = this->create_subscription<sensor_msgs::msg::Image>("camera", 50, std::bind(&ActorDetection::processImage, this, _1));
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ActorDetection>());
    rclcpp::shutdown();

    return 0;
}
