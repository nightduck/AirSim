#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include "sensor_msgs/msg/image.hpp"
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cinematography_msgs/msg/multi_do_farray.hpp>
#include <cinematography_msgs/msg/multi_dof.hpp>
#include "cinematography_msgs/msg/vision_measurements.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include "filter.h"
#include "unistd.h"

using namespace std::chrono_literals;
using std::placeholders::_1;

class MotionForecasting : public rclcpp::Node {
private:
    std::string ACTOR_NAME;
    int CAMERA_FPS;

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

    geometry_msgs::msg::Quaternion flatten(geometry_msgs::msg::Quaternion quat) {
        double length = sqrt(quat.w * quat.w + quat.z * quat.z);
        quat.w = quat.w / length;
        quat.x = quat.y = 0;
        quat.z = quat.z / length;
        return quat;
    }

    float getYaw(Eigen::Quaternion<float, 2> q) {
        q = flatten(q);
        double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
        return std::atan2(siny_cosp, cosy_cosp);
    }

    float getYaw(geometry_msgs::msg::Quaternion q) {
        q = flatten(q);
        double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
        double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    float getPitch(geometry_msgs::msg::Quaternion q) {
        geometry_msgs::msg::Quaternion f = flatten(q);

        float dot_product = f.w * q.w + f.x * q.x + f.y * q.y + f.z * q.z;
        float angle = acos(2*dot_product*dot_product - 1);
        return angle;
    }

    void getVisionMeasurements(const cinematography_msgs::msg::VisionMeasurements::SharedPtr msg) {
        // TODO: Implement an actual EKF

        msr::airlib::Pose pose = airsim_client->simGetObjectPose(ACTOR_NAME);
        if (pose.position.x() == NAN) {
            RCLCPP_ERROR(this->get_logger(), "Actor name not found!!");
            return;
        }
        
        rclcpp::Time now = rclcpp::Time(msg->header.stamp, RCL_SYSTEM_TIME);
        rclcpp::Duration duration = now - lastPoseTimestamp;
        lastPoseTimestamp = now;

        double fws;                                                     // Calculate length of trajectory
        get_parameter("forecast_window_secs", fws);
        int num_points = fws * CAMERA_FPS;          // Depending on duration of point, the duration
                                                    // of the path might not match the specified
                                                    // fws, but it's important to bound the size of
                                                    // the traj array to guaranteed max computation
                                                    // requirements in motion planning node

        ukf_meas_clear();
        if (msg->width > 0) {
            ukf_set_bb(msg->centerx, msg->centery);
            ukf_set_depth(msg->depth);
            ukf_set_hde(msg->hde);
        }
        ukf_set_position(msg->drone_pose.position.x, msg->drone_pose.position.y, msg->drone_pose.position.z);
        ukf_set_yaw(getYaw(msg->drone_pose.orientation));
        ukf_set_pitch(getPitch(msg->drone_pose.orientation));

        cinematography_msgs::msg::MultiDOFarray pred_traj;
        pred_traj.header.stamp = now;

        // Return a forecasted trajectory, including point for this point in time. Update filter in
        // background
        //pred_traj.points = ukf_iterate(duration, num_points);
        

        // DEBUGGING UNTIL FILTER IS WORKING AND TRAINED
        pred_traj.points.clear();
        msr::airlib::Vector3r posDiff = pose.position - lastPose.position;
        double yawDiff = getYaw(pose.orientation) - getYaw(lastPose.orientation);
        for(int i = 0; i < num_points; i++) {
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
        declare_parameter<double>("forecast_window_secs", 10);
        declare_parameter<std::string>("actor_name", "Deer");
        get_parameter("actor_name", ACTOR_NAME);

        auto parameters_client = std::make_shared<rclcpp::SyncParametersClient>(this, "airsim_ros2_wrapper");
        while (!parameters_client->wait_for_service(1s)) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "service not available, waiting again...");
        }
        ukf_init(0,0,0,0);
        airsim_hostname = parameters_client->get_parameter<std::string>("airsim_hostname");
        CAMERA_FPS = parameters_client->get_parameter<int>("camera_fps");

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
