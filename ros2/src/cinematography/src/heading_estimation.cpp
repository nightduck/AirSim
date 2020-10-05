#include "rclcpp/rclcpp.hpp"
#include "cinematography_msgs/msg/bounding_box.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <math.h>

using std::placeholders::_1;

class HeadingEstimation : public rclcpp::Node {
private:
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sat_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_sub;

    // NOTE: This variable makes this node not safe for multithreaded spinning
    geometry_msgs::msg::Pose drone_pose;

    void processImage(const cinematography_msgs::msg::BoundingBox::SharedPtr msg) {
        //sensor_msgs::Image img = msg->data.image();

        // TODO: Pad image to 192x192 square with black bars
        // TODO: Pass through Model to get heading estimation
        float hde0 = 1;   // Output #1
        float hde1 = 1;   // Output #2
        geometry_msgs::msg::Quaternion actor_orientation;
        actor_orientation.w = acos(cos(hde0)/2);
        actor_orientation.z = asin(sin(hde1)/2);

        // TODO: (Optionally in parallel) Using actor position, camera FOV, and actor's position within frame, project the
        //       actor onto the ground plane and get their coordinates
        geometry_msgs::msg::Point actor_position;

        // Combine orientation and position into a single pose message and publish it
        geometry_msgs::msg::Pose actor_pose;
        actor_pose.orientation = actor_orientation;
        actor_pose.position = actor_position;
        pose_pub->publish(actor_pose);
    }

    void getCoordinates(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        drone_pose.position.x = msg->latitude;
        drone_pose.position.y = msg->longitude;
        drone_pose.position.z = -1 * msg->longitude;
    }

    void getOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
        drone_pose.orientation = msg->pose.pose.orientation;
        // drone_pose = msg->pose.pose;     # For use if the GPS proves unideal
    }

public:
    HeadingEstimation() : Node("heading_estimation") {
        pose_pub = this->create_publisher<geometry_msgs::msg::Pose>("actor_pose", 50);
        sat_sub = this->create_subscription<sensor_msgs::msg::NavSatFix>("satellite_pos", 1, std::bind(&HeadingEstimation::getCoordinates, this, _1));
        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("odom_pos", 1, std::bind(&HeadingEstimation::getOdometry, this, _1));
        bb_sub = this->create_subscription<cinematography_msgs::msg::BoundingBox>("bounding_box", 50, std::bind(&HeadingEstimation::processImage, this, _1));
    }
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HeadingEstimation>());
    rclcpp::shutdown();

    return 0;
}