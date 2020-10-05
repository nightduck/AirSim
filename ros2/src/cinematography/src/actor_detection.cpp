#include "rclcpp/rclcpp.hpp"
#include "cinematography_msgs/msg/gimbal_angle_quat_cmd.hpp"
#include "cinematography_msgs/msg/bounding_box.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"

using std::placeholders::_1;

class ActorDetection : public rclcpp::Node {
private:
    rclcpp::Publisher<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_pub;
    rclcpp::Publisher<cinematography_msgs::msg::GimbalAngleQuatCmd>::SharedPtr gimbal_control;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera;

    int difffilter(const std::vector<uint8_t> &v, std::vector<uint8_t> &w, int m, int n);

    void processImage(const sensor_msgs::msg::Image::SharedPtr msg) {
        // TODO: Add parameter to specify which type our actor is (deer, car, person, etc)
        // TODO: Run image through YOLO model and get bounding box coordiantes of object that is most confidently our actor

        std::vector<uint8_t> array;
        array.reserve(msg->data.size());
        //difffilter(msg->data, array, msg->width, msg->height);

        uint8_t prev = msg->data[0];
        int len = msg->data.size();
        array[0] = prev;
        for (int i = 1; i < 1000000; i++) {
            array[i%len] = msg->data[i%len] * prev;
            prev = array[i%len];
        }
        float centerx = array[0] / 512.0 + 0.25;
        float centery = array[1] / 512.0 + 0.25;
        float width = array[2] / 1024.0;
        float height = array[3] / 1024.0;

        // TODO: Use centering coordinates to move gimbal so actor is in center frame (OPTIONAL: Add parameter to specify
        //       where in frame the actor should be)

        // TODO: Extract subimage, package with centering coordinates, and publish to /bounding_box
        cinematography_msgs::msg::BoundingBox bb;
        bb.left = centerx - width/2;
        bb.bottom = centery + height/2;

        int px_left = bb.left * msg->width;
        int px_top = (centery - height/2) * msg->height;
        int px_width = width * msg->width;
        int px_height = height * msg->height;
        bb.image = *msg;

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
