#include "rclcpp/rclcpp.hpp"
#include <pcl/point_types.h>
#include <signal.h>
#include <boost/foreach.hpp>
#include <math.h>
#include <stdio.h>
#include <cmath>   
#include <iostream>
#include <fstream>
#include <cinematography_msgs/msg/multi_dof.hpp>
#include <cinematography_msgs/msg/multi_do_farray.hpp>
#include "nav_msgs/msg/odometry.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

#define FREQ 5
#define MARGIN 0.01     // Used as a margin of error to prevent the call to follow_trajectory from going over the node's period
#define DAMPING_FACTOR_X 0.05
#define DAMPING_FACTOR_Y 0.05
#define DAMPING_FACTOR_Z 0.15
#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

#define MAX_YAW_RATE    90

using namespace std;

FILE *fd;

// add by feiyang jin
//bool fly_back = false;
//std_msgs::Bool fly_back_msg;
bool global_stop_fly = false;
bool stopped = true;

bool slam_lost = false;
bool created_slam_loss_traj = false;
bool copy_slam_loss_traj = false;
bool copy_panic_traj = false;
bool copy_normal_traj = false;
bool wasEmpty = true;

float yaw_threshold = 5;

struct multiDOFpoint {
    double x, y, z;
    double vx, vy, vz;
    double ax, ay, az;
    double yaw;
    double duration;
};
typedef std::deque<multiDOFpoint> trajectory_t;

trajectory_t normal_traj;
trajectory_t rev_normal_traj;
trajectory_t slam_loss_traj;
trajectory_t panic_traj;
float g_v_max = 6.0;

float g_max_yaw_rate = 90;
float g_max_yaw_rate_during_flight = 90;

bool g_trajectory_done;

bool should_panic = false;
// geometry_msgs::msg::Vector3 panic_velocity;

int rviz_id = 0;

int traj_id = 0;

msr::airlib::MultirotorRpcLibClient* airsim_client;

rclcpp::Node::SharedPtr node;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr rviz_vel_pub;

rclcpp::Clock::SharedPtr clock_;

void callback_trajectory(cinematography_msgs::msg::MultiDOFarray::SharedPtr msg)
{
    fprintf( fd, "follow_trajectory_callback_trajectory_entry" );
    fflush( fd );
	// RCLCPP_INFO(node->get_logger(), "call back trajectory");
    normal_traj.clear();
    rev_normal_traj.clear();
    for (auto point : msg->points){
        multiDOFpoint traj_point;
        traj_point.x = point.x;
        traj_point.y = point.y;
        traj_point.z = point.z;
        traj_point.vx = point.vx;
        traj_point.vy = point.vy;
        traj_point.vz = point.vz;
        traj_point.yaw = point.yaw;
        traj_point.duration = point.duration;
        normal_traj.push_back(traj_point);
    }

    traj_id = msg->traj_id;

    copy_normal_traj = true;
    fprintf( fd, "follow_trajectory_callback_trajectory_exit" );
    fflush( fd );
}

void print_rviz_vel(double x, double y, double z, double vx, double vy, double vz) {
    nav_msgs::msg::Odometry rviz_point = nav_msgs::msg::Odometry();
    rviz_point.header.stamp = clock_->now();
    rviz_point.header.frame_id = "world_enu";
    rviz_point.child_frame_id = rviz_id++;

    rviz_point.pose.pose.position.x = x;
    rviz_point.pose.pose.position.y = y;
    rviz_point.pose.pose.position.z = z;
    rviz_point.twist.twist.linear.x = vx;
    rviz_point.twist.twist.linear.y = vy;
    rviz_point.twist.twist.linear.z = vz;

    rviz_vel_pub->publish(rviz_point);
}

int main(int argc, char **argv)
{
    fd = fopen("/sys/kernel/debug/tracing/trace_marker", "a");
    if (fd == NULL) {
        perror("Could not open trace marker");
        return -1;
    }
    rclcpp::init(argc, argv);

    node = rclcpp::Node::make_shared("follow_trajectory");

    clock_ = node->get_clock();

    auto parameters_client = std::make_shared<rclcpp::SyncParametersClient>(node, "airsim_ros2_wrapper");
    while (!parameters_client->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(node->get_logger(), "Interrupted while waiting for the service. Exiting.");
            return 1;
        }
        RCLCPP_INFO(node->get_logger(), "service not available, waiting again...");
    }
    std::string host_ip = parameters_client->get_parameter<std::string>("airsim_hostname");

    airsim_client = new msr::airlib::MultirotorRpcLibClient(host_ip);
    airsim_client->confirmConnection();
    airsim_client->enableApiControl(true);
    airsim_client->takeoffAsync()->waitOnLastTask();
    // airsim_client->moveToPositionAsync(0,0,-30,3)->waitOnLastTask();
    // airsim_client->moveToPositionAsync(0,0,0,3)->waitOnLastTask();

    //variable     
    rclcpp::Rate loop_rate(FREQ);    // NOTE: Set to frequency of trajectory points
    g_trajectory_done = false;

    //publisher and subscriber

    auto trajectory_follower_sub = node->create_subscription<cinematography_msgs::msg::MultiDOFarray>("drone_traj", 1, callback_trajectory); 

    // DEBUGGING
    rviz_vel_pub = node->create_publisher<nav_msgs::msg::Odometry>("velocities", 1);

    bool following_traj = false;  //decides when the first planning has occured
                               //this allows us to activate all the
                               //functionaliy in follow_trajecotry accordingly
    std::chrono::system_clock::time_point sleep_time; //todo

    // yaw_strategy_t yaw_strategy = ignore_yaw;

    trajectory_t * traj = nullptr;
    while (rclcpp::ok()) {
    	rclcpp::spin_some(node);


        // setup trajectory
        if (unlikely(copy_panic_traj)) {
            traj = &panic_traj;
            copy_panic_traj = false;
        } else if (unlikely(copy_slam_loss_traj)) {
            traj = &slam_loss_traj;
            copy_slam_loss_traj = false;
        }
        else if (copy_normal_traj) {
            traj = &normal_traj;
            copy_normal_traj = false;
        }

        // If there's a trajectory to follow and we haven't been commanded to halt,
        // start the clock and signal that we should follow the trajectory. If
        // commanded to halt, signal to stop following trajectory
        if (normal_traj.size() > 0 && !global_stop_fly) {
            if (!following_traj) {
                sleep_time = std::chrono::system_clock::now();
            }

            following_traj = true;
        } else if (global_stop_fly) {
            following_traj = false;
        }


        // make sure if new traj come, we do not conflict
        const int id_for_this_traj = traj_id;

        // Follow a single point in the trajectory. This will probably mean sleeping until the last one is reached
        if(following_traj) {
            if (likely(!traj->empty())){
                if (wasEmpty) {
                    RCLCPP_ERROR(node->get_logger(), "    forward trajectory now full");
                    wasEmpty = false;
                }

                static double max_speed_so_far = 0;
                static int ctr = 0;

                multiDOFpoint p = traj->front();

                // Calculate the velocities we should be flying at
                double v_x = p.vx;
                double v_y = p.vy;
                double v_z = p.vz;
                float yaw = p.yaw*180/M_PI;

		        //std::cout << "getting position" << std::endl;
                Vector3r pos = airsim_client->getMultirotorState().getPosition();
		        //std::cout << "got position" << std::endl;
                double posx = pos.x();
                double posy = pos.y();
                double posz = pos.z();
                v_x += DAMPING_FACTOR_X*(p.x-pos.x())/p.duration;
                v_y += DAMPING_FACTOR_Y*(p.y-pos.y())/p.duration;
                v_z += DAMPING_FACTOR_Z*(p.z-pos.z())/p.duration;

                // Make sure we're not going over the maximum speed
                double speed = std::sqrt((v_x*v_x + v_y*v_y + v_z*v_z));
                double scale = 1;
               if (speed > g_v_max) {
                   scale = g_v_max / speed;

                   v_x *= scale;
                   v_y *= scale;
                   v_z *= scale;
                   speed = std::sqrt((v_x*v_x + v_y*v_y + v_z*v_z));
               }

                // Calculate the time for which this point's flight commands should run
                auto scaled_flight_time = std::chrono::duration<double>(p.duration / scale);

               fprintf( fd, "follow_trajectory_controlLoop_exit" );
               fflush( fd );

                // Wait until the the last point finishes processing, then tackle this point
                // std::cout << "It is now " << std::chrono::system_clock::now().time_since_epoch().count() << ", sleeping until " << sleep_time.time_since_epoch().count() << std::endl;
                std::this_thread::sleep_until(sleep_time);      // NOTE: Getting stuck here indefinitely?
		        // std::cout << "Ok, I'm up" << std::endl;
                fprintf( fd, "follow_trajectory_controlLoop_entry" );
                fflush( fd );
                                
                auto q = airsim_client->getMultirotorState().getOrientation();
		        // std::cout << "Got orientation" << std::endl;
                float pitch, roll, current_yaw;
                msr::airlib::VectorMath::toEulerianAngle(q, pitch, roll, current_yaw);
                current_yaw = current_yaw*180 / M_PI;

                float yaw_diff = fmod(yaw - current_yaw + 360, 360);
                yaw_diff = yaw_diff <= 180 ? yaw_diff : yaw_diff - 360;

                if(yaw_diff >= yaw_threshold){
                    yaw_diff -= yaw_threshold;
                }
                else if(yaw_diff <= (-1)*yaw_threshold){
                    yaw_diff += yaw_threshold;
                }
                else{
                    yaw_diff = 0;
                }
                
                float yaw_rate = yaw_diff / scaled_flight_time.count();

                if (yaw_rate > MAX_YAW_RATE)
                    yaw_rate = MAX_YAW_RATE;
                else if (yaw_rate < -MAX_YAW_RATE)
                    yaw_rate = -MAX_YAW_RATE;

                auto drivetrain = msr::airlib::DrivetrainType::MaxDegreeOfFreedom;
                auto yawmode = msr::airlib::YawMode(true, yaw_rate);

		        //std::cout << "Flying at (" << v_x << ", " << v_y << ", " << v_z << ") for " << p.duration << " seconds" << std::endl;
                airsim_client->moveByVelocityAsync(v_x, v_y, v_z, scaled_flight_time.count(), drivetrain, yawmode);
		        std::cout << "Flew" << std::endl;
                print_rviz_vel(pos.y(), pos.x(), -1*pos.z(), v_x, v_y, v_z);

                // Get deadline to process next point
		        //std::cout << "Current sleep time: " << sleep_time.time_since_epoch().count() << ", adding" << scaled_flight_time.count() << std::endl;
                sleep_time += std::chrono::duration_cast<std::chrono::system_clock::duration>(scaled_flight_time);
                // std::cout << "Updated sleep time: " << sleep_time.time_since_epoch().count() << std::endl;

                // Update trajectory
                traj->pop_front();
            }
            else {
                if (!wasEmpty) {
                    RCLCPP_ERROR(node->get_logger(), "!!! forward trajectory empty! Doing nothing");
                    wasEmpty = true;
                }
            }
        }
    }
	return 0;
}
