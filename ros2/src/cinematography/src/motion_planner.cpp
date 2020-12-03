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

void print_rviz_traj(cinematography_msgs::msg::MultiDOFarray path, std::string name, bool actor) {
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

//======================VVV==Cost functions==VVV====================================

double traj_smoothness(const cinematography_msgs::msg::MultiDOFarray& drone_traj) {
    return 1;
}

double shot_quality(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& ideal_traj) {
    return 1;
}

double obstacle_avoidance(const cinematography_msgs::msg::MultiDOFarray& drone_traj) {   // TODO: Add 2nd argument for TSDF
    return 1;
}

double occlusion_avoidance(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& actor_traj) {   // TODO: Add 3rd argument for TSDF
    return 1;
}

double traj_cost_function(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& actor_traj, cinematography_msgs::msg::MultiDOFarray& ideal_traj) {      // TODO: Add 4th argument for TSDF
    double LAMBDA_1, LAMBDA_2, LAMBDA_3 = 1;    // TODO: Have these specified as ROS parameters

    return traj_smoothness(drone_traj) + LAMBDA_1 * obstacle_avoidance(drone_traj) + LAMBDA_2 * occlusion_avoidance(drone_traj, actor_traj) + LAMBDA_3 * shot_quality(drone_traj, ideal_traj);
}

// TODO: Implement gradient equivalents of all the above, and hessian approximations (A_smooth + delta_1 * A_shot)

void optimize_trajectory(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& actor_traj) {
    cinematography_msgs::msg::MultiDOFarray ideal_traj = calc_ideal_drone_traj(actor_traj);     // ξ_shot when calculating shot quality

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

    cinematography_msgs::msg::MultiDOFarray ideal_path;

    ideal_path = calc_ideal_drone_traj(*actor_traj);
    
    print_rviz_traj(ideal_path, "drone_traj", false);

    // TODO: Put in artificial load here to simulate optimizing

    //optimize_trajectory(ideal_path, *actor_traj);

    for(int i = 1; i < ideal_path.points.size(); i++) {
        ideal_path.points[i-1].vx = (ideal_path.points[i].x - ideal_path.points[i-1].x) / ideal_path.points[i-1].duration;
        ideal_path.points[i-1].vy = (ideal_path.points[i].y - ideal_path.points[i-1].y) / ideal_path.points[i-1].duration;
        ideal_path.points[i-1].vz = (ideal_path.points[i].z - ideal_path.points[i-1].z) / ideal_path.points[i-1].duration;
    }

    // Publish the trajectory (for debugging purposes)
    ideal_path.header.stamp = clock_->now();
    path_found = true;

    RCLCPP_INFO(node->get_logger(), "Publishing drone trajectory");

    traj_pub->publish(ideal_path);
}


int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    node = rclcpp::Node::make_shared("/motion_planner");
    clock_ = node->get_clock();

    auto actor_traj_sub = node->create_subscription<cinematography_msgs::msg::MultiDOFarray>("/actor_traj", 1, get_actor_trajectory); 

    traj_pub = node->create_publisher<cinematography_msgs::msg::MultiDOFarray>("/multidoftraj", 1);

    rviz_actor_pub = node->create_publisher<geometry_msgs::msg::PoseArray>("/rviz_actor_traj", 1);
    rviz_drone_pub = node->create_publisher<geometry_msgs::msg::PoseArray>("/rviz_drone_traj", 1);

    // Sleep
    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
}