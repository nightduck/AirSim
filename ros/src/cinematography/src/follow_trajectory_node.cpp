#include "ros/ros.h"
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <signal.h>
#include <boost/foreach.hpp>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include "std_msgs/Bool.h"
#include "common_mav.h"
#include <mav_msgs/default_topics.h>    
#include <iostream>
#include <fstream>
#include <std_srvs/SetBool.h>
#include <airsim_ros_pkgs/MultiDOF.h>
#include <airsim_ros_pkgs/MultiDOFarray.h>
#include <airsim_ros_pkgs/follow_trajectory_status_srv.h>
#include <airsim_ros_pkgs/BoolPlusHeader.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <trajectory_msgs/MultiDOFJointTrajectoryPoint.h>
#include <nav_msgs/Odometry.h>
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

trajectory_t normal_traj;
trajectory_t rev_normal_traj;
trajectory_t slam_loss_traj;
trajectory_t panic_traj;
float g_v_max = 2.5;

float g_max_yaw_rate = 90;
float g_max_yaw_rate_during_flight = 90;

bool g_trajectory_done;

bool should_panic = false;
geometry_msgs::Vector3 panic_velocity;

ros::Publisher rviz_vel_pub;
int rviz_id = 0;

int traj_id = 0;

msr::airlib::MultirotorRpcLibClient* airsim_client;


void callback_trajectory(const airsim_ros_pkgs::MultiDOFarray::ConstPtr& msg)
{
	ROS_INFO("call back trajectory");
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
}


bool trajectory_done(const trajectory_msgs::MultiDOFJointTrajectory& trajectory) {
    g_trajectory_done = (trajectory.points.size() == 0);
    return g_trajectory_done;
}

void print_rviz_vel(double x, double y, double z, double vx, double vy, double vz) {
    nav_msgs::Odometry rviz_point = nav_msgs::Odometry();
    rviz_point.header.seq = 0;
    rviz_point.header.stamp = ros::Time::now();
    rviz_point.header.frame_id = "world_enu";
    rviz_point.child_frame_id = rviz_id++;

    rviz_point.pose.pose.position.x = x;
    rviz_point.pose.pose.position.y = y;
    rviz_point.pose.pose.position.z = z;
    rviz_point.twist.twist.linear.x = vx;
    rviz_point.twist.twist.linear.y = vy;
    rviz_point.twist.twist.linear.z = vz;

    rviz_vel_pub.publish(rviz_point);
}


bool follow_trajectory_status_cb(airsim_ros_pkgs::follow_trajectory_status_srv::Request &req,
                                 airsim_ros_pkgs::follow_trajectory_status_srv::Response &res)
{
    //res.success.data = true;
    //ROS_INFO("trajectory done: %i", g_trajectory_done);
    res.success.data = g_trajectory_done;

    const multiDOFpoint& current_point =
        normal_traj.empty() ? rev_normal_traj.front() : normal_traj.front();
    
    geometry_msgs::Twist last_velocity;
    last_velocity.linear.x = current_point.vx;
    last_velocity.linear.y = current_point.vy;
    last_velocity.linear.z = current_point.vz;
    res.twist = last_velocity;
  
    geometry_msgs::Twist last_acceleration;
    last_acceleration.linear.x = current_point.ax;
    last_acceleration.linear.y = current_point.ay;
    last_acceleration.linear.z = current_point.az;
    res.acceleration = last_acceleration;
    return true;
}


airsim_ros_pkgs::MultiDOFarray next_steps_msg(const trajectory_t& traj, const int true_id) {
    airsim_ros_pkgs::MultiDOFarray array_of_point_msg;

    for (const auto& point : traj){
        airsim_ros_pkgs::MultiDOF point_msg;
        point_msg.x = point.x;
        point_msg.y = point.y;
        point_msg.z = point.z;
        point_msg.vx = point.vx;
        point_msg.vy = point.vy;
        point_msg.vz = point.vz;
        point_msg.ax = point.ax;
        point_msg.ay = point.ay;
        point_msg.az = point.az;
        point_msg.yaw = point.yaw;
        point_msg.duration = point.duration;
        array_of_point_msg.points.push_back(point_msg);
    }

    array_of_point_msg.traj_id = true_id;
    array_of_point_msg.header.stamp = ros::Time::now();
    return array_of_point_msg;
}


void stop_fly_callback(const std_msgs::Bool::ConstPtr& msg){
    bool stop_fly_local = msg->data;
    global_stop_fly = stop_fly_local;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "follow_trajectory", ros::init_options::NoSigintHandler);
    ros::NodeHandle n;
    ros::NodeHandle n_private("~");
    std::string host_ip;
    n.param<std::string>("airsim_hostname", host_ip, "localhost");

    airsim_client = new msr::airlib::MultirotorRpcLibClient(host_ip);
    airsim_client->takeoffAsync(5)->waitOnLastTask();

    //variable
	    std::string localization_method; 
	    std::string mav_name;
	    ros::Time cur_t, last_t;
	    float cur_z, last_z = -9999;
	    bool created_future_col_traj = false;
	    trajectory_t future_col_traj;
	    ros::Rate loop_rate(FREQ);     // NOTE: Set to frequency of trajectory points
        g_trajectory_done = false;

    //publisher and subscriber
	    ros::ServiceServer trajectory_done_service = n.advertiseService("follow_trajectory_status", follow_trajectory_status_cb);

	    ros::Publisher next_steps_pub = n.advertise<airsim_ros_pkgs::MultiDOFarray>("/next_steps", 1);
	    ros::Subscriber trajectory_follower_sub = n.subscribe<airsim_ros_pkgs::MultiDOFarray>("multidoftraj", 1, callback_trajectory);

        ros::Subscriber stop_fly_sub = n.subscribe<std_msgs::Bool>("/stop_fly", 1, stop_fly_callback);

        // DEBUGGING
        rviz_vel_pub = n.advertise<nav_msgs::Odometry>("velocities", 1);

    bool following_traj = false;  //decides when the first planning has occured
                               //this allows us to activate all the
                               //functionaliy in follow_trajecotry accordingly
    std::chrono::system_clock::time_point sleep_time;

    yaw_strategy_t yaw_strategy = ignore_yaw;

    trajectory_t * traj = nullptr;
    while (ros::ok()) {
    	ros::spinOnce();


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
                static double max_speed_so_far = 0;
                static int ctr = 0;

                multiDOFpoint p = traj->front();

                // Calculate the velocities we should be flying at
                double v_x = p.vx;
                double v_y = p.vy;
                double v_z = p.vz;
                float yaw = p.yaw*180/M_PI;

                Vector3r pos = airsim_client->getMultirotorState().getPosition();
                double posx = pos.x();
                double posy = pos.y();
                double posz = pos.z();
                v_x += DAMPING_FACTOR_X*(p.x-pos.x())/p.duration;
                v_y += DAMPING_FACTOR_Y*(p.y-pos.y())/p.duration;
                v_z += DAMPING_FACTOR_Z*(p.z-pos.z())/p.duration;

                // Make sure we're not going over the maximum speed
                double speed = std::sqrt((v_x*v_x + v_y*v_y + v_z*v_z));
                double scale = 1;
//                if (speed > g_v_max) {
//                    scale = g_v_max / speed;
//
//                    v_x *= scale;
//                    v_y *= scale;
//                    v_z *= scale;
//                    speed = std::sqrt((v_x*v_x + v_y*v_y + v_z*v_z));
//                }

                // Calculate the time for which this point's flight commands should run
                auto scaled_flight_time = std::chrono::duration<double>(p.duration / scale);

                // Wait until the the last point finishes processing, then tackle this point
                std::this_thread::sleep_until(sleep_time);
                                
                auto q = airsim_client->getMultirotorState().getOrientation();
                float pitch, roll, current_yaw;
                msr::airlib::VectorMath::toEulerianAngle(q, pitch, current_yaw, roll);
                current_yaw = yaw*180 / M_PI;

                float yaw_diff = (int(yaw - current_yaw) + 360) % 360;
                yaw_diff = yaw_diff <= 180 ? yaw_diff : yaw_diff - 360;
                
                float yaw_rate = yaw_diff / scaled_flight_time.count();

                if (yaw_rate > MAX_YAW_RATE)
                    yaw_rate = MAX_YAW_RATE;
                else if (yaw_rate < -MAX_YAW_RATE)
                    yaw_rate = -MAX_YAW_RATE;

                auto drivetrain = msr::airlib::DrivetrainType::MaxDegreeOfFreedom;
                auto yawmode = msr::airlib::YawMode(true, yaw_rate);

                airsim_client->moveByVelocityAsync(v_x, v_y, v_z, scaled_flight_time.count(), drivetrain, yawmode);

                print_rviz_vel(pos.y(), pos.x(), -1*pos.z(), v_x, v_y, v_z);
                ROS_INFO("Flying at (%f, %f, %f) for %f seconds", v_x, v_y, v_z, p.duration);

                // Get deadline to process next point
                sleep_time += std::chrono::duration_cast<std::chrono::system_clock::duration>(scaled_flight_time);

                // Update trajectory
                traj->pop_front();
            }
            else {
                ROS_ERROR("!!! forward trajectory empty! Doing nothing");
            }
        }
    }
	return 0;
}