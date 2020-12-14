// Standard headers
#include "rclcpp/rclcpp.hpp"
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <thread>
#include <functional>
#include <limits>
#include <signal.h>
#include "cmath"
#include <iterator>
#include <boost/range/combine.hpp>

#include <cinematography_msgs/msg/drone_state.hpp>
#include <cinematography_msgs/msg/multi_do_farray.hpp>
#include <cinematography_msgs/msg/multi_dof.hpp>
#include <cinematography_msgs/msg/artistic_spec.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/msg/pose_array.hpp>

#include <ompl/geometric/PathGeometric.h>
#include <ompl/base/State.h>
#include <ompl/base/spaces/TimeStateSpace.h>

#include <Eigen/Dense>


#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

#define DEG_TO_RAD(d)   d*M_PI/180
#define RAD_TO_DEG(r)   r*180/M_PI

using namespace std;

rclcpp::Publisher<cinematography_msgs::msg::MultiDOFarray>::SharedPtr traj_pub;
rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr rviz_actor_pub;
rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr rviz_drone_pub;

bool path_found = false;

// // TODO: change the bound to something meaningful
// double x__low_bound__global = -200, x__high_bound__global = 200;
// double y__low_bound__global = -200 , y__high_bound__global = 200;
// double z__low_bound__global = 0, z__high_bound__global = 40;
// double sampling_interval__global = 0.5;
// double v_max__global = 3, a_max__global = 5;
// float g_planning_budget = 4;
// std::string motion_planning_core_str;

// double drone_height__global = 0.6;
// double drone_radius__global = 2;

// Define default artistic constraints
double viewport_heading = M_PI / 3;
double viewport_pitch = M_PI / 4;
double viewport_distance = 5;

rclcpp::Node::SharedPtr node;
rclcpp::Clock::SharedPtr clock_;

msr::airlib::MultirotorRpcLibClient* airsim_client;
std::string airsim_hostname;
std::string vehicle_name = "drone_1";

void print_rviz_traj(cinematography_msgs::msg::MultiDOFarray& path, std::string name, bool actor) {
    geometry_msgs::msg::PoseArray rviz_path = geometry_msgs::msg::PoseArray();
    rviz_path.header.stamp = clock_->now();
    rviz_path.header.frame_id = "world_ned";
    for(cinematography_msgs::msg::MultiDOF n : path.points) {
        geometry_msgs::msg::Pose p;
        p.orientation = tf2::toMsg(tf2::Quaternion(0, 0, sin(n.yaw / 2), cos(n.yaw / 2)));
        p.position.x = n.x;
        p.position.y = n.y;
        p.position.z = n.z;
        rviz_path.poses.push_back(p);
    }

    if (actor)
        rviz_actor_pub->publish(rviz_path);
    else
        rviz_drone_pub->publish(rviz_path);
}

// Calculate an ideal drone trajectory using a given actor trajectory (both in NED and radians)
cinematography_msgs::msg::MultiDOFarray calc_ideal_drone_traj(const cinematography_msgs::msg::MultiDOFarray &  actor_traj) {
    cinematography_msgs::msg::MultiDOFarray drone_traj;
    drone_traj.points.reserve(actor_traj.points.size());

    float horiz_dist = cos(viewport_pitch) * viewport_distance;
    float height = sin(viewport_pitch) * viewport_distance;

    // For each point in the actor's trajectory...
    for (cinematography_msgs::msg::MultiDOF point : actor_traj.points) {
        cinematography_msgs::msg::MultiDOF n;

        // Center p on the actor to get the drone's ideal coordinates
        n.x = point.x + cos(viewport_heading + point.yaw) * horiz_dist;
        n.y = point.y + sin(viewport_heading + point.yaw) * horiz_dist;
        n.z = point.z - height;
        n.yaw = point.yaw + M_PI + viewport_heading;
        n.duration = point.duration;

        if (n.yaw > M_PI) {
            n.yaw -= 2*M_PI;
        }
        drone_traj.points.push_back(n);
    }
    return drone_traj;
}

// Moves the starting point to the drone's current position, leave the final point in place, and all the intermediate points are stretched in between
void move_traj_start(cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOF& drone_pose) {
    cinematography_msgs::msg::MultiDOF traj_offset;
    traj_offset.x = drone_traj.points[0].x - drone_pose.x;
    traj_offset.x = drone_traj.points[0].y - drone_pose.y;
    traj_offset.x = drone_traj.points[0].z - drone_pose.z;


    for(int i = 0; i < drone_traj.points.size(); i++) {
        float weight = 1 - (i / (drone_traj.points.size() - 1));    // Fraction of the offset to subtract
        drone_traj.points[i].x -= weight * (traj_offset.x);
        drone_traj.points[i].y -= weight * (traj_offset.y);
        drone_traj.points[i].z -= weight * (traj_offset.z);
        drone_traj.points[i].vx = drone_traj.points[i].vy = drone_traj.points[i].vz
         = drone_traj.points[i].ax = drone_traj.points[i].ay = drone_traj.points[i].az;
    }

    // First point has current velocity. Note that if you sum up velocity and acceleration of a point
    // according to actual physics, you will not get the correct next point. It's very approximative
    drone_traj.points[0].vx = drone_pose.vx;
    drone_traj.points[0].vy = drone_pose.vy;
    drone_traj.points[0].vz = drone_pose.vz;
    for(int i = 1; i < drone_traj.points.size() - 1; i++) {
        drone_traj.points[i].vx = (drone_traj.points[i+1].x - drone_traj.points[i].x) / drone_traj.points[i].duration;
        drone_traj.points[i].vy = (drone_traj.points[i+1].y - drone_traj.points[i].y) / drone_traj.points[i].duration;
        drone_traj.points[i].vz = (drone_traj.points[i+1].z - drone_traj.points[i].z) / drone_traj.points[i].duration;
        drone_traj.points[i-1].ax = (drone_traj.points[i+1].vx - drone_traj.points[i].vx) / drone_traj.points[i].duration;
        drone_traj.points[i-1].ay = (drone_traj.points[i+1].vy - drone_traj.points[i].vy) / drone_traj.points[i].duration;
        drone_traj.points[i-1].az = (drone_traj.points[i+1].vz - drone_traj.points[i].vz) / drone_traj.points[i].duration;
    }

    // Add acceleration and velocity to final points. Velocity stays constant, and acceleration goes to zero
    cinematography_msgs::msg::MultiDOF penultimitePoint = drone_traj.points[drone_traj.points.size() - 2];
    penultimitePoint.ax = penultimitePoint.ay = penultimitePoint.az = 0;
    cinematography_msgs::msg::MultiDOF finalPoint = drone_traj.points[drone_traj.points.size() - 1];
    finalPoint.vx = penultimitePoint.vx;
    finalPoint.vy = penultimitePoint.vy;
    finalPoint.vz = penultimitePoint.vz;
    finalPoint.ax = finalPoint.ay = finalPoint.az = 0;

    return;
}

void face_actor(cinematography_msgs::msg::MultiDOFarray& drone_traj, const cinematography_msgs::msg::MultiDOFarray& actor_traj) {
    if (drone_traj.points.size() != actor_traj.points.size()) {
        //RCLCPP_ERROR("Cannot face actor. Two trajectories don't match in number of points");
        return;
    }

    std::vector<cinematography_msgs::msg::MultiDOF>::iterator dit = drone_traj.points.begin();
    std::vector<cinematography_msgs::msg::MultiDOF>::const_iterator ait = actor_traj.points.begin();

    double d_time = dit->duration, a_time = 0;
    for(; dit < drone_traj.points.end(); ait++,dit++) {
        // Get the vector difference (just x and y, since we only care about the yaw of the drone)
        tf2::Vector3 diff = tf2::Vector3(ait->x, ait->y, 0) - tf2::Vector3(dit->x, dit->y, 0);

        // Get the angle
        double angle = atan(diff.y() / diff.x());

        // Double check which cartesian quadrant you're in and add/subtract a 180 offset if necessary
        if (diff.x() < 0) {
            if (diff.y() < 0) {
                angle -= M_PI;
            } else {
                angle += M_PI;
            }
        }

        dit->yaw = angle;
    }

    return;
}

//======================VVV==Cost functions==VVV====================================

double traj_smoothness(const cinematography_msgs::msg::MultiDOFarray& drone_traj, int delta_t) {
    int a0 = 1, a1 = 0.5, a2 = 0;       // TODO: Fix these somehow

    int n = drone_traj.points.size();
    Eigen::MatrixXd q(n-1, 3);
    Eigen::MatrixXd K = Eigen::MatrixXd::Identity(n-1,n-1);

    for(int i = 0; i < n-1; i++) {
        q(i,0) = drone_traj.points[i+1].x;      // Initialize matrix for drone trajectory
        q(i,1) = drone_traj.points[i+1].y;
        q(i,2) = drone_traj.points[i+1].z;
        K(i,i+1) = -1;                            // Initialize K (n-1,n-1) as -I
    }

    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(n-1,3);
    e(0,0) = -1 * drone_traj.points[0].x;
    e(0,1) = -1 * drone_traj.points[0].y;
    e(0,2) = -1 * drone_traj.points[0].z;

    Eigen::MatrixXd K0 = K / delta_t;
    Eigen::MatrixXd K1 = K.array().pow(2) / pow(delta_t,2);
    Eigen::MatrixXd K2 = K.array().pow(3) / pow(delta_t,3);
    Eigen::MatrixXd e0 = e / delta_t;
    Eigen::MatrixXd e1 = e / delta_t;
    Eigen::MatrixXd e2 = e / delta_t;

    Eigen::MatrixXd A = a0 * K0.transpose() * K0 + a1 * K1.transpose() * K1 + a2 * K2.transpose() * K2;  // NOTE: This is always the same
    Eigen::MatrixXd b = a0 * K0.transpose() * e0 + a1 * K1.transpose() * e1 + a2 * K2.transpose() * e2;
    Eigen::MatrixXd c = a0 * e0.transpose() * e0 + a1 * e1.transpose() * e1 + a2 * e2.transpose() * e2;

    double sum = (q.transpose()*A*q + 2*q.transpose()*b + c).trace();

    return sum / (2*(n-1));
}

double shot_quality(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& ideal_traj) {
    int n = drone_traj.points.size();
    Eigen::MatrixXd q(n-1, 3);
    Eigen::MatrixXd shot(n-1, 3);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n-1,n-1);

    for(int i = 0; i < n-1; i++) {
        q(i,0) = drone_traj.points[i+1].x;      // Initialize matrix for drone trajectory
        q(i,1) = drone_traj.points[i+1].y;
        q(i,2) = drone_traj.points[i+1].z;
        shot(i,0) = ideal_traj.points[i+1].x;   // Initialize matrix for ideal shot trajectory
        shot(i,1) = ideal_traj.points[i+1].y;
        shot(i,2) = ideal_traj.points[i+1].z;
        K(i,i) = -1;                            // Initialize K (n-1,n-1) as -I
    }

    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n-1,n-1);
    Eigen::MatrixXd b = K.transpose() * shot;
    Eigen::MatrixXd c = shot.transpose() * shot;

    Eigen::MatrixXd sum = q.transpose() * A * q + 2 * q.transpose() * b + c;
    double trace = sum.trace();
    return trace / (2 * (n-1));
}

double obstacle_avoidance(const cinematography_msgs::msg::MultiDOFarray& drone_traj) {   // TODO: Add 2nd argument for TSDF
    return 1;
}

double occlusion_avoidance(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& actor_traj) {   // TODO: Add 3rd argument for TSDF
    return 1;
}

double traj_cost_function(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& actor_traj, cinematography_msgs::msg::MultiDOFarray& ideal_traj, double t) {      // TODO: Add 4th argument for TSDF
    double LAMBDA_1, LAMBDA_2, LAMBDA_3 = 1;    // TODO: Have these specified as ROS parameters

    return traj_smoothness(drone_traj, t) + LAMBDA_1 * obstacle_avoidance(drone_traj) + LAMBDA_2 * occlusion_avoidance(drone_traj, actor_traj) + LAMBDA_3 * shot_quality(drone_traj, ideal_traj);
}

// TODO: Implement gradient equivalents of all the above, and hessian approximations (A_smooth + delta_1 * A_shot)
Eigen::MatrixXd traj_smoothness_gradient(const cinematography_msgs::msg::MultiDOFarray& drone_traj, double delta_t) {
    int a0 = 1, a1 = 0.5, a2 = 0;       // TODO: Fix these somehow

    int n = drone_traj.points.size();
    Eigen::MatrixXd q(n-1, 3);
    Eigen::MatrixXd K = Eigen::MatrixXd::Identity(n-1,n-1);

    for(int i = 0; i < n-1; i++) {
        q(i,0) = drone_traj.points[i+1].x;      // Initialize matrix for drone trajectory
        q(i,1) = drone_traj.points[i+1].y;
        q(i,2) = drone_traj.points[i+1].z;
        K(i,i+1) = -1;                            // Initialize K (n-1,n-1) as -I
    }

    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(n-1,3);
    e(0,0) = -1 * drone_traj.points[0].x;
    e(0,1) = -1 * drone_traj.points[0].y;
    e(0,2) = -1 * drone_traj.points[0].z;

    Eigen::MatrixXd K0 = K / delta_t;
    Eigen::MatrixXd K1 = K.array().pow(2) / pow(delta_t,2);
    Eigen::MatrixXd K2 = K.array().pow(3) / pow(delta_t,3);
    Eigen::MatrixXd e0 = e / delta_t;
    Eigen::MatrixXd e1 = e / delta_t;
    Eigen::MatrixXd e2 = e / delta_t;

    Eigen::MatrixXd A = a0 * K0.transpose() * K0 + a1 * K1.transpose() * K1 + a2 * K2.transpose() * K2;  // NOTE: This is always the same
    Eigen::MatrixXd b = a0 * K0.transpose() * e0 + a1 * K1.transpose() * e1 + a2 * K2.transpose() * e2;

    Eigen::MatrixXd sum = A*q + b;

    return sum / (n-1);
}

Eigen::MatrixXd shot_quality_gradient(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& ideal_traj) {
    int n = drone_traj.points.size();
    Eigen::MatrixXd q(3, n-1);
    Eigen::MatrixXd shot(3, n-1);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n-1,n-1);

    for(int i = 0; i < n-1; i++) {
        q(i,0) = drone_traj.points[i+1].x;      // Initialize matrix for drone trajectory
        q(i,1) = drone_traj.points[i+1].y;
        q(i,2) = drone_traj.points[i+1].z;
        shot(i,0) = ideal_traj.points[i+1].x;   // Initialize matrix for ideal shot trajectory
        shot(i,1) = ideal_traj.points[i+1].y;
        shot(i,2) = ideal_traj.points[i+1].z;
        K(i,i) = -1;                            // Initialize K (n-1,n-1) as -I
    }

    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(n-1,n-1);
    Eigen::MatrixXd b = K.transpose() * shot;

    Eigen::MatrixXd sum = A * q + b;
    return sum / (n-1);
}

Eigen::MatrixXd obstacle_avoidance_gradient(const cinematography_msgs::msg::MultiDOFarray& drone_traj) {
    return Eigen::MatrixXd::Zero(3,drone_traj.points.size() - 1);
}

Eigen::MatrixXd obstacle_avoidance_gradient(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& actor_traj) {
    return Eigen::MatrixXd::Zero(3,drone_traj.points.size() - 1);
}

void optimize_trajectory(cinematography_msgs::msg::MultiDOFarray& drone_traj, const cinematography_msgs::msg::MultiDOFarray& actor_traj) {
    cinematography_msgs::msg::MultiDOFarray ideal_traj = calc_ideal_drone_traj(actor_traj);     // ξ_shot when calculating shot quality

    double t = 0;
    for(cinematography_msgs::msg::MultiDOF p : actor_traj.points) {
        t += p.duration;
    }
    t /= actor_traj.points.size();

    int MAX_ITERATIONS;     // TODO: Make this a ROS parameter
    for(int i = 0; i < 1000; i++) {
        
    }
}

//======================^^^==Cost functions==^^^====================================

//bool get_trajectory_fun(airsim_ros_pkgs::get_trajectory::Request &req, airsim_ros_pkgs::get_trajectory::Response &res)
// Get actor's predicted trajectory (in NED and radians)
void get_actor_trajectory(cinematography_msgs::msg::MultiDOFarray::SharedPtr actor_traj)
{
    if (actor_traj->points.size() == 0) {
        RCLCPP_ERROR(node->get_logger(), "Received empty actor path"); 
        return;
    }

    print_rviz_traj(*actor_traj, "actor_traj", true);

    // TODO: Get this further up in the vision pipeline and pass it down. Also get velocity and acceleration info
    msr::airlib::Pose currentPose = airsim_client->simGetVehiclePose(vehicle_name);
    cinematography_msgs::msg::MultiDOF currentState;
    currentState.x = currentPose.position.x();
    currentState.y = currentPose.position.y();
    currentState.z = currentPose.position.z();
    currentState.vx = currentState.vy = currentState.vz = currentState.ax = currentState.ay = currentState.az = 0;

    cinematography_msgs::msg::MultiDOFarray drone_path;
    drone_path = calc_ideal_drone_traj(*actor_traj);        // Calculate the ideal observation point for every point in actor trajectory
    move_traj_start(drone_path, currentState);              // Skew ideal path, so it starts at the drone's current position
    
    print_rviz_traj(drone_path, "drone_traj", false);

    optimize_trajectory(drone_path, *actor_traj);

    face_actor(drone_path, *actor_traj);                    // Set all yaws to fact their corresponding point in the actor trajectory

    // Publish the trajectory (for debugging purposes)
    drone_path.header.stamp = clock_->now();
    path_found = true;

    RCLCPP_INFO(node->get_logger(), "Publishing drone trajectory");

    traj_pub->publish(drone_path);
}


int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    node = rclcpp::Node::make_shared("/motion_planner");
    clock_ = node->get_clock();

    node->declare_parameter<std::string>("airsim_hostname", "localhost");
    node->get_parameter("airsim_hostname", airsim_hostname);
    airsim_client = new msr::airlib::MultirotorRpcLibClient(airsim_hostname);

    auto actor_traj_sub = node->create_subscription<cinematography_msgs::msg::MultiDOFarray>("/actor_traj", 1, get_actor_trajectory); 

    traj_pub = node->create_publisher<cinematography_msgs::msg::MultiDOFarray>("/multidoftraj", 1);

    rviz_actor_pub = node->create_publisher<geometry_msgs::msg::PoseArray>("/rviz_actor_traj", 1);
    rviz_drone_pub = node->create_publisher<geometry_msgs::msg::PoseArray>("/rviz_drone_traj", 1);

    // Sleep
    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
}