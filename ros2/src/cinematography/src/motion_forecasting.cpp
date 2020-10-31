#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cinematography_msgs/msg/multi_do_farray.hpp>
#include <cinematography_msgs/msg/multi_dof.hpp>
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

using std::placeholders::_1;

class MotionForecasting : public rclcpp::Node {
private:
    const double FORECAST_WINDOW_SECS = 10; // TODO: Make this a parameter

    rclcpp::Publisher<cinematography_msgs::msg::MultiDOFarray>::SharedPtr predict_pub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub;

    msr::airlib::MultirotorRpcLibClient airsim_client;

    // TODO: Probably replace this with proper type
    msr::airlib::Pose lastPose;
    rclcpp::Time lastPoseTimestamp;

    const std::string DEER_NAME = "DeerBothBP2_19";

    Eigen::Quaternion<float, 2> flatten(Eigen::Quaternion<float, 2> quat) {
        double length = sqrt(quat.w() * quat.w() + quat.z() * quat.z());
        return Eigen::Quaternion<float, 2>(quat.w() / length, 0, 0, quat.z() / length);
    }

    float getYaw(Eigen::Quaternion<float, 2> q) {
        q = flatten(q);
        double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
        return std::atan2(siny_cosp, cosy_cosp);
    }

    void getActorPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        // TODO: Implement an actual EKF

        msr::airlib::Pose pose = airsim_client.simGetObjectPose(DEER_NAME);
        
        rclcpp::Time now = rclcpp::Time(msg->header.stamp, RCL_SYSTEM_TIME);
        rclcpp::Duration duration = now - lastPoseTimestamp;
        lastPoseTimestamp = now;

        cinematography_msgs::msg::MultiDOFarray pred_traj;
        pred_traj.header.stamp = now;

        pred_traj.points.reserve(FORECAST_WINDOW_SECS / duration.seconds() + 1);
        msr::airlib::Pose diff = (pose - lastPose);
        for(int i = 0; i < pred_traj.points.capacity(); i++) {
            cinematography_msgs::msg::MultiDOF point;
            //trajectory_msgs::msg::MultiDOFJointTrajectoryPoint point;
            point.x = pose.position.x() + diff.position.x() * i;
            point.y = pose.position.y() + diff.position.y() * i;
            point.z = pose.position.z() + diff.position.z() * i;
            point.vx = diff.position.x() / duration.seconds();
            point.vy = diff.position.y() / duration.seconds();
            point.vz = diff.position.z() / duration.seconds();
            point.ax = 0;
            point.ay = 0;
            point.az = 0;
            point.duration = duration.seconds();
            point.yaw = getYaw(pose.orientation) + getYaw(diff.orientation) * i;
            pred_traj.points.push_back(point);
        }

        lastPose = pose;

        predict_pub->publish(pred_traj);
    }

public:
    // TODO: Pass IP address of airsim as parameter
    MotionForecasting() : Node("motion_forecasting"), airsim_client("localhost") {
        lastPoseTimestamp = rclcpp::Time(0, 0, RCL_SYSTEM_TIME);
        predict_pub = this->create_publisher<cinematography_msgs::msg::MultiDOFarray>("pred_path", 1);
        pose_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>("actor_pose", 50, std::bind(&MotionForecasting::getActorPose, this, _1));
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotionForecasting>());
    rclcpp::shutdown();

    return 0;
}
