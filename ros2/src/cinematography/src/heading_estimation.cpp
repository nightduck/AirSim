#include "rclcpp/rclcpp.hpp"
#include "rclcpp/logging.hpp"
#include "cinematography_msgs/msg/bounding_box.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <math.h>

using std::placeholders::_1;

class HeadingEstimation : public rclcpp::Node {
private:
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sat_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_sub;

    // NOTE: These variables makes this node not safe for multithreaded spinning
    geometry_msgs::msg::PoseStamped drone_pose;
    geometry_msgs::msg::TransformStamped t;         // Dummy variable for math-ing
    geometry_msgs::msg::Vector3Stamped unit_vect;
    geometry_msgs::msg::QuaternionStamped unit_quat;

    void processImage(const cinematography_msgs::msg::BoundingBox::SharedPtr msg) {

        // Pad image to 192x192 square with black bars
        sensor_msgs::msg::Image img = msg->image;
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::BGR8);
        cv::Mat new_img;
        if (cv_ptr->image.rows != 192 && cv_ptr->image.cols != 192) {
            RCLCPP_ERROR(this->get_logger(), "Image for HDE is not 192x192");
            return;  
        } else if (cv_ptr->image.rows != cv_ptr->image.cols) {
            int vert_padding = (192-cv_ptr->image.cols)/2;
            int horz_padding = (192-cv_ptr->image.rows)/2;
            int right_px = (cv_ptr->image.cols % 2 != 0) ? 1 : 0;
            int bottom_px = (cv_ptr->image.rows % 2 != 0) ? 1 : 0;
            cv::copyMakeBorder(cv_ptr->image, new_img, horz_padding, horz_padding + bottom_px,
                    vert_padding, vert_padding + right_px, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
            cv_ptr->image = new_img;
        }


        // TODO: Pass through Model to get heading estimation
        //==================DUMMY LOAD===========================
        std::vector<uint8_t> array;
        array.reserve(msg->image.data.size());
        uint8_t prev = msg->image.data[0];
        int len = msg->image.data.size();
        array[0] = prev;
        for (int i = 1; i < 1000000; i++) {
            array[i%len] = msg->image.data[i%len] * prev;
            prev = array[i%len];
        }
        //================DUMMY LOAD============================
        float hde0 = array[0] / 256.0;   // Output #1
        float hde1 = array[1] / 256.0;   // Output #2

        geometry_msgs::msg::Quaternion actor_orientation;
        int angle = atan2(hde1, hde0);
        actor_orientation.w = acos(cos(angle)/2);
        actor_orientation.z = asin(sin(angle)/2);


        t.transform.rotation = msg->actor_direction;
        geometry_msgs::msg::Vector3Stamped actor_ray;
        tf2::doTransform(unit_vect, actor_ray, t);
        float scale = drone_pose.pose.position.z / actor_ray.vector.z;
        actor_ray.vector.x *= scale;
        actor_ray.vector.y *= scale;
        actor_ray.vector.z *= scale;

        t.transform.translation = actor_ray.vector;
        t.transform.rotation = actor_orientation;

        // // TODO: (Optionally in parallel) Using actor position, camera FOV, and actor's position within frame, project the
        // //       actor onto the ground plane and get their coordinates
        // geometry_msgs::msg::Point actor_position;
        // tf2::Quaternion actor_quat;
        // tf2::fromMsg(msg->actor_direction, actor_quat);                 // Get rotation needed to face actor from msg
        // const tf2::Quaternion unit = tf2::Quaternion(1,0,0,0);
        // actor_quat = actor_quat * unit * actor_quat.inverse();          // Rotate a unit vector in that direction. Actor_quat is now treated as 3D vector
        // float scale = drone_pose.position.z / actor_quat.z();
        // actor_quat *= scale;
        // tf2::Vector3 thing = tf2::Vector3(actor_quat.x(), actor_quat.y(), actor_quat.z());
        // actor_position = tf2::toMsg(thing);

        // Combine orientation and position into a single pose message and publish it
        geometry_msgs::msg::PoseStamped actor_pose;
        actor_pose.header.frame_id = "world_ned";
        actor_pose.header.stamp = this->now();
        tf2::doTransform(drone_pose, actor_pose, t);
        pose_pub->publish(actor_pose.pose);
    }

    void getCoordinates(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        drone_pose.pose.position.x = msg->latitude;
        drone_pose.pose.position.y = msg->longitude;
        drone_pose.pose.position.z = -1 * msg->longitude;
    }

    void getOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
        drone_pose.pose.orientation = msg->pose.pose.orientation;
        // drone_pose = msg->pose.pose;     # For use if the GPS proves unideal
    }

public:
    HeadingEstimation() : Node("heading_estimation") {
        unit_vect.vector.x = unit_quat.quaternion.x = unit_quat.quaternion.w = 1;
        unit_vect.vector.y = unit_vect.vector.z = unit_quat.quaternion.y = unit_quat.quaternion.z = 0;

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