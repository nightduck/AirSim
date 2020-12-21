#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cinematography_msgs/msg/multi_do_farray.hpp>
#include <cinematography_msgs/msg/multi_dof.hpp>
#include "cinematography_msgs/msg/vision_measurements.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include <Eigen/Dense>

using std::placeholders::_1;

class MotionForecasting : public rclcpp::Node {
private:
    double FORECAST_WINDOW_SECS;
    std::string ACTOR_NAME;

    rclcpp::Publisher<cinematography_msgs::msg::MultiDOFarray>::SharedPtr predict_pub;
    rclcpp::Subscription<cinematography_msgs::msg::VisionMeasurements>::SharedPtr pose_sub;

    msr::airlib::MultirotorRpcLibClient* airsim_client;

    // TODO: Probably replace this with proper type
    msr::airlib::Pose lastPose;
    rclcpp::Time lastPoseTimestamp;


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

    void getVisionMeasurements(const cinematography_msgs::msg::VisionMeasurements::SharedPtr msg) {
        // TODO: Implement an actual EKF

        msr::airlib::Pose pose = airsim_client->simGetObjectPose(ACTOR_NAME);
        
        rclcpp::Time now = rclcpp::Time(msg->header.stamp, RCL_SYSTEM_TIME);
        rclcpp::Duration duration = now - lastPoseTimestamp;
        lastPoseTimestamp = now;

        cinematography_msgs::msg::MultiDOFarray pred_traj;
        pred_traj.header.stamp = now;

        int fws;
        get_parameter("forecast_window_secs", fws);
        pred_traj.points.reserve(fws / duration.seconds() + 1);
        msr::airlib::Vector3r posDiff = pose.position - lastPose.position;
        double yawDiff = getYaw(pose.orientation) - getYaw(lastPose.orientation);
        for(int i = 0; i < pred_traj.points.capacity(); i++) {
            cinematography_msgs::msg::MultiDOF point;
            //trajectory_msgs::msg::MultiDOFJointTrajectoryPoint point;
            point.x = pose.position.x() + posDiff.x() * i;
            point.y = pose.position.y() + posDiff.y() * i;
            point.z = pose.position.z() + posDiff.z() * i;
            point.vx = posDiff.x() / duration.seconds();
            point.vy = posDiff.y() / duration.seconds();
            point.vz = posDiff.z() / duration.seconds();
            point.ax = 0;
            point.ay = 0;
            point.az = 0;
            point.duration = duration.seconds();
            point.yaw = getYaw(pose.orientation) + yawDiff * i;
            pred_traj.points.push_back(point);
        }

        lastPose = pose;

        predict_pub->publish(pred_traj);
    }

public:
    MotionForecasting() : Node("motion_forecasting") {
        std::string airsim_hostname;
        declare_parameter<int>("forecast_window_secs", 10);
        get_parameter("forecast_window_secs", FORECAST_WINDOW_SECS);
        declare_parameter<std::string>("actor_name", "DeerBothBP2_19");
        get_parameter("actor_name", ACTOR_NAME);
        declare_parameter<std::string>("airsim_hostname", "localhost");
        get_parameter("airsim_hostname", airsim_hostname);

        airsim_client = new msr::airlib::MultirotorRpcLibClient(airsim_hostname);

        lastPoseTimestamp = rclcpp::Time(0, 0, RCL_SYSTEM_TIME);
        predict_pub = this->create_publisher<cinematography_msgs::msg::MultiDOFarray>("pred_path", 1);
        pose_sub = this->create_subscription<cinematography_msgs::msg::VisionMeasurements>("vision_measurements", 50, std::bind(&MotionForecasting::getVisionMeasurements, this, _1));
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotionForecasting>());
    rclcpp::shutdown();

    return 0;
}
