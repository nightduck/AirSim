#include "rclcpp/rclcpp.hpp"
#include "rclcpp/logging.hpp"
#include "cinematography_msgs/msg/bounding_box.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
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
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sat_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_sub;

    // NOTE: These variables makes this node not safe for multithreaded spinning
    geometry_msgs::msg::PoseStamped drone_pose;
    geometry_msgs::msg::TransformStamped t;         // Dummy variable for math-ing
    geometry_msgs::msg::Vector3Stamped unit_vect;
    geometry_msgs::msg::QuaternionStamped unit_quat;

    rclcpp::Clock clock = rclcpp::Clock(RCL_SYSTEM_TIME);

    void flatten(tf2::Quaternion &quat) {
        quat.setX(0);
        quat.setY(0);
        double length = sqrt(quat.w() * quat.w() + quat.z() * quat.z());
        quat.setW(quat.w() / length);
        quat.setZ(quat.z() / length);
    }

    void processImage(const cinematography_msgs::msg::BoundingBox::SharedPtr msg) {
        int res;
        get_parameter("resolution", res);

        // Pad image to 192x192 square with black bars
        sensor_msgs::msg::Image img = msg->image;
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::BGR8);
        cv::Mat new_img;

        // Pad to square
        if (cv_ptr->image.rows != cv_ptr->image.cols) {
            int max_dim = (cv_ptr->image.rows > cv_ptr->image.cols) ? cv_ptr->image.rows : cv_ptr->image.cols;
            int vert_padding = (max_dim-cv_ptr->image.cols)/2;
            int horz_padding = (max_dim-cv_ptr->image.rows)/2;
            int right_px = (cv_ptr->image.cols % 2 != 0) ? 1 : 0;
            int bottom_px = (cv_ptr->image.rows % 2 != 0) ? 1 : 0;
            cv::copyMakeBorder(cv_ptr->image, new_img, horz_padding, horz_padding + bottom_px,
                    vert_padding, vert_padding + right_px, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
            cv_ptr->image = new_img;
        }
        
        // If neither image is 192px, scale input image
        if (cv_ptr->image.rows != 192 && cv_ptr->image.cols != 192) {
            cv::Mat dest;
            double scaling_factor = res / ((cv_ptr->image.rows > cv_ptr->image.cols) ? cv_ptr->image.rows : cv_ptr->image.cols);
            cv::resize(cv_ptr->image, dest, cv::Size(res, res));
            cv_ptr->image = dest;
        }


        // TODO: Pass through Model to get heading estimation
        //==================DUMMY LOAD===========================
        std::vector<uint8_t> array;
        array.reserve(msg->image.data.size());
        uint8_t prev = msg->image.data[0];
        int len = msg->image.data.size();
        array[0] = prev;
        for (int i = 0; i < 1000000; i++) {
            array[i%len] = msg->image.data[i%len] * prev + (i % 256);
            prev = array[i%len];
        }
        //================DUMMY LOAD============================
        double hde0 = array[0] / 128.0 - 1;   // Output #1
        double hde1 = array[1] / 128.0 - 1;   // Output #2
        // TODO: Normalize to [-1,1] range

        // Get the rotation of the actor relative to camera
        double angle = atan2(hde1, hde0);
        tf2::Quaternion quat_actor_rel;
        quat_actor_rel.setRPY(0, 0, angle);

        // Compute a unit vector pointing at the actor
        tf2::Quaternion quat_gimbal, quat_drone, quat_actor_direction;
        tf2::fromMsg(msg->actor_direction, quat_gimbal);
        tf2::fromMsg(drone_pose.pose.orientation, quat_drone);
        t.transform.rotation = tf2::toMsg(quat_drone * quat_gimbal);
        t.transform.translation = geometry_msgs::msg::Vector3();
        geometry_msgs::msg::Vector3Stamped actor_ray;
        tf2::doTransform(unit_vect, actor_ray, t);

        // Project actor ray to ground, resulting vector being position of actor
        double scale = abs(drone_pose.pose.position.z / actor_ray.vector.z);
        actor_ray.vector.x *= scale;
        actor_ray.vector.y *= scale;
        actor_ray.vector.z *= scale;

        // Flatten drone's quaternions (so they only represent yaw), and use that to calculate the
        // actor's absolute rotation
        flatten(quat_gimbal);
        flatten(quat_drone);
        tf2::Quaternion quat_actor_absolute = quat_actor_rel * quat_drone * quat_gimbal;

        // NOTE: RESUME HERE. Verify the actor's absolute rotation is being calcualted correctly from it's relative rotation
        // Combine actor's position and rotation (adjusted from relative to absolute)
        geometry_msgs::msg::PoseStamped actor_pose;
        actor_pose.pose.position.x = actor_ray.vector.x + drone_pose.pose.position.x;
        actor_pose.pose.position.y = actor_ray.vector.y + drone_pose.pose.position.y;
        actor_pose.pose.position.z = actor_ray.vector.z + drone_pose.pose.position.z;
        actor_pose.pose.orientation = tf2::toMsg(quat_actor_absolute);

        // Publish to rviz, as debugging step rviz_pose;
        actor_pose.header.frame_id = "world_ned";
        actor_pose.header.stamp = clock.now();
        pose_pub->publish(actor_pose);
    }

    void getCoordinates(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        // TODO: Maybe implement this as a 2nd source, taking into account starting coordinates aren't (0,0,0)
        // drone_pose.pose.position.x = msg->latitude;
        // drone_pose.pose.position.y = msg->longitude;
        // drone_pose.pose.position.z = -1 * msg->longitude;
    }

    void getOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
        drone_pose.pose = msg->pose.pose;
    }

public:
    HeadingEstimation() : Node("heading_estimation") {
        declare_parameter("resolution", 192);

        unit_vect.vector.x = unit_quat.quaternion.x = unit_quat.quaternion.w = 1;
        unit_vect.vector.y = unit_vect.vector.z = unit_quat.quaternion.y = unit_quat.quaternion.z = 0;

        pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("actor_pose", 50);
        sat_sub = this->create_subscription<sensor_msgs::msg::NavSatFix>("satellite_pose", 1, std::bind(&HeadingEstimation::getCoordinates, this, _1));
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