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
#include <unordered_set>
#include <Eigen/Dense>


#include <cinematography_msgs/msg/drone_state.hpp>
#include <cinematography_msgs/msg/multi_do_farray.hpp>
#include <cinematography_msgs/msg/multi_dof.hpp>
#include <cinematography_msgs/msg/artistic_spec.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include "tsdf_package_msgs/msg/tsdf.hpp"
#include "tsdf_package_msgs/msg/voxel.hpp"
#include "optimize_drone_path.cuh"

#include <Eigen/Dense>

#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

#define DEG_TO_RAD(d)   d*M_PI/180
#define RAD_TO_DEG(r)   r*180/M_PI

using namespace std;

rclcpp::Publisher<cinematography_msgs::msg::MultiDOFarray>::SharedPtr drone_traj_pub;
rclcpp::Publisher<cinematography_msgs::msg::MultiDOFarray>::SharedPtr ideal_traj_pub;

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

tf2_ros::Buffer* tf_buffer;
tf2_ros::TransformListener* tf_listener;

std::string world_frame;
std::string drone_frame;

double truncation_distance = 4;
double voxel_size = .5;
bool received_first_msg = false;

std::vector<Voxel> voxels_set[NUM_BUCKETS];
int voxels_set_size;

std::chrono::time_point<std::chrono::high_resolution_clock> global_start;
int global_iterations = 0;
double average = 0;
bool first_time = true;

int MAX_ITERATIONS;

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

void update_traj_position_derivatives(cinematography_msgs::msg::MultiDOFarray& drone_traj){
    for(int i = 0; i < drone_traj.points.size() - 1; i++) {
        drone_traj.points[i].vx = (drone_traj.points[i+1].x - drone_traj.points[i].x) / drone_traj.points[i].duration;
        drone_traj.points[i].vy = (drone_traj.points[i+1].y - drone_traj.points[i].y) / drone_traj.points[i].duration;
        drone_traj.points[i].vz = (drone_traj.points[i+1].z - drone_traj.points[i].z) / drone_traj.points[i].duration;
    }


    // Note that if you sum up velocity and acceleration of a point according to actual physics,
    // you will not get the correct next point. It's very approximative
    for(int i = 0; i < drone_traj.points.size() - 1; i++) {
        drone_traj.points[i].ax = (drone_traj.points[i+1].vx - drone_traj.points[i].vx) / drone_traj.points[i].duration;
        drone_traj.points[i].ay = (drone_traj.points[i+1].vy - drone_traj.points[i].vy) / drone_traj.points[i].duration;
        drone_traj.points[i].az = (drone_traj.points[i+1].vz - drone_traj.points[i].vz) / drone_traj.points[i].duration;
    }

    // Add acceleration and velocity to final points. Velocity stays constant, and acceleration goes to zero
    cinematography_msgs::msg::MultiDOF penultimitePoint = drone_traj.points[drone_traj.points.size() - 2];
    penultimitePoint.ax = penultimitePoint.ay = penultimitePoint.az = 0;
    cinematography_msgs::msg::MultiDOF finalPoint = drone_traj.points[drone_traj.points.size() - 1];
    finalPoint.vx = penultimitePoint.vx;
    finalPoint.vy = penultimitePoint.vy;
    finalPoint.vz = penultimitePoint.vz;
    finalPoint.ax = finalPoint.ay = finalPoint.az = 0;

    drone_traj.points[drone_traj.points.size() - 2] = penultimitePoint;
    drone_traj.points[drone_traj.points.size() - 1] = finalPoint;

}

// Moves the starting point to the drone's current position, leave the final point in place, and all the intermediate points are stretched in between
void move_traj_start(cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOF& drone_pose) {
    cinematography_msgs::msg::MultiDOF traj_offset;
    traj_offset.x = drone_traj.points[0].x - drone_pose.x;
    traj_offset.y = drone_traj.points[0].y - drone_pose.y;
    traj_offset.z = drone_traj.points[0].z - drone_pose.z;


    for(int i = 0; i < drone_traj.points.size(); i++) {
        float weight = 1.0 - (i / (float)(drone_traj.points.size() - 1));    // Fraction of the offset to subtract
        drone_traj.points[i].x -= weight * (traj_offset.x);
        drone_traj.points[i].y -= weight * (traj_offset.y);
        drone_traj.points[i].z -= weight * (traj_offset.z);
        drone_traj.points[i].ax = drone_traj.points[i].ay = drone_traj.points[i].az = 0;
    }

    update_traj_position_derivatives(drone_traj);

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
    for(; dit <= drone_traj.points.end(); ait++,dit++) {
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

size_t get_bucket(const Eigen::Matrix<double, 3, 1> & position){
    return abs((((int)position(0)*73856093) ^ ((int)position(1)*19349669) ^ ((int)position(2)*83492791)) % NUM_BUCKETS);
}

double floor_fun(const double & x, const double & scale){
    return floor(x*scale) / scale;
}

/*
* Given world point return center of volume given by volume_size
*/
Eigen::Matrix<double, 3, 1> get_volume_center_from_point(Eigen::Matrix<double, 3, 1> point, double volume_size){
    double scale = 1/volume_size;
    double half_volume_size = volume_size / 2;
    Eigen::Matrix<double, 3, 1> volume_center;
    volume_center(0) = floor_fun(point(0), scale) + half_volume_size;
    volume_center(1) = floor_fun(point(1), scale) + half_volume_size;
    volume_center(2) = floor_fun(point(2), scale) + half_volume_size;
    return volume_center;
    }

/*
* Check if two Eigen::Matrix<double, 3, 1> are equal
*/
bool check_floating_point_vectors_equal(Eigen::Matrix<double, 3, 1> A, Eigen::Matrix<double, 3, 1> B, double epsilon){
    Eigen::Matrix<double, 3, 1> diff = A-B;
    //have to use an epsilon value due to floating point precision errors
    if((fabs(diff(0)) < epsilon) && (fabs(diff(1)) < epsilon) && (fabs(diff(2)) < epsilon))
    return true;

    return false;
}

/*
* Get voxels between point start and point end
* For more information on voxel traversal algo: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf
*/
std::vector<Eigen::Matrix<double, 3, 1>> get_voxels(const cinematography_msgs::msg::MultiDOF point_start, const cinematography_msgs::msg::MultiDOF point_end, const double & volume_size){
    
    std::vector<Eigen::Matrix<double, 3, 1>> traversed_voxels;
    const double half_volume_size = volume_size / 2;
    const double epsilon = volume_size / 4;
    const double volume_size_plus_epsilon = volume_size + epsilon;
    const double volume_size_minus_epsilon = volume_size - epsilon;
    Eigen::Matrix<double, 3, 1> start(point_start.x, point_start.y, point_start.z);
    Eigen::Matrix<double, 3, 1> end(point_end.x, point_end.y, point_end.z);
    //   equation of line = u+tv
    Eigen::Matrix<double, 3, 1> u(point_start.x, point_start.y, point_start.z);
    Eigen::Matrix<double, 3, 1> v(point_end.x - point_start.x, point_end.y - point_start.y, point_end.z- point_start.z);
    double stepX = v(0) > 0 ? volume_size : -1 * volume_size;
    double stepY = v(1) > 0 ? volume_size : -1 * volume_size;
    double stepZ = v(2) > 0 ? volume_size : -1 * volume_size;
    Eigen::Matrix<double, 3, 1> start_voxel_center = get_volume_center_from_point(start, volume_size);
    double tMaxX = fabs(v(0) < 0 ? (start_voxel_center(0) - half_volume_size - u(0)) / v(0) : (start_voxel_center(0) + half_volume_size - u(0)) / v(0));
    double tMaxY = fabs(v(1) < 0 ? (start_voxel_center(1) - half_volume_size - u(1)) / v(1) : (start_voxel_center(1) + half_volume_size - u(1)) / v(1));
    double tMaxZ = fabs(v(2) < 0 ? (start_voxel_center(2) - half_volume_size - u(2)) / v(2) : (start_voxel_center(2) + half_volume_size - u(2)) / v(2));
    double tDeltaX = fabs(volume_size / v(0));
    double tDeltaY = fabs(volume_size / v(1));
    double tDeltaZ = fabs(volume_size / v(2));
    Eigen::Matrix<double, 3, 1> current_vol(start(0), start(1), start(2));
    Eigen::Matrix<double, 3, 1> current_vol_center = get_volume_center_from_point(current_vol, volume_size);
    Eigen::Matrix<double, 3, 1> end_voxel_center = get_volume_center_from_point(end, volume_size);

    while(!check_floating_point_vectors_equal(current_vol_center, end_voxel_center, epsilon)){
        //add traversed voxel to list of voxels
        traversed_voxels.push_back(current_vol_center);

        if(traversed_voxels.size() >= 1000){ //change to check if any of the points have been overrun
            // printf("\n!!!VOXEL TRAVERSAL ISSUE!!!\n\n");
            printf("current_vol_center: (%f,%f,%f), end_center: (%f,%f,%f), start_center: (%f,%f,%f), start: (%f,%f,%f), end: (%f,%f,%f)\n", current_vol_center(0), current_vol_center(1), current_vol_center(2), end_voxel_center(0), end_voxel_center(1), end_voxel_center(2), start_voxel_center(0), start_voxel_center(1), start_voxel_center(2), start(0), start(1), start(2), end(0), end(1), end(2));
            return traversed_voxels;
        }
        
        if(tMaxX < tMaxY){
            if(tMaxX < tMaxZ)
            {
                current_vol(0) += stepX;
                tMaxX += tDeltaX;
            }
            else if(tMaxX > tMaxZ){
                current_vol(2) += stepZ;
                tMaxZ += tDeltaZ;
            }
            else{
                current_vol(0) += stepX;
                current_vol(2) += stepZ;
                tMaxX += tDeltaX;
                tMaxZ += tDeltaZ;
            }
        }
        else if(tMaxX > tMaxY){
            if(tMaxY < tMaxZ){
            current_vol(1) += stepY;
            tMaxY += tDeltaY;
            }
            else if(tMaxY > tMaxZ){
                current_vol(2) += stepZ;
                tMaxZ += tDeltaZ;
            }
            else{
                current_vol(1) += stepY;
                current_vol(2) += stepZ;
                tMaxY += tDeltaY;
                tMaxZ += tDeltaZ;
            }
        }
        else{
            if(tMaxZ < tMaxX){
                current_vol(2) += stepZ;
                tMaxZ += tDeltaZ;
            }
            else if(tMaxZ > tMaxX){
                current_vol(0) += stepX;
                current_vol(1) += stepY;
                tMaxX += tDeltaX;
                tMaxY += tDeltaY;
            }
            else{ 
                current_vol(0) += stepX;
                current_vol(1) += stepY;
                current_vol(2) += stepZ;
                tMaxX += tDeltaX;
                tMaxY += tDeltaY;
                tMaxZ += tDeltaZ;
            }
        } 
        //deals with floating point precision errors
        Eigen::Matrix<double, 3, 1> temp_current_vol_center = current_vol_center;
        current_vol_center = get_volume_center_from_point(current_vol, volume_size);
        Eigen::Matrix<double, 3, 1> diff;
        diff(0) = fabs(temp_current_vol_center(0) - current_vol_center(0));
        diff(1) = fabs(temp_current_vol_center(1) - current_vol_center(1));
        diff(2) = fabs(temp_current_vol_center(2) - current_vol_center(2));
        if((diff(0) < volume_size_minus_epsilon && diff(1) < volume_size_minus_epsilon && diff(2) < volume_size_minus_epsilon) 
        || (diff(0) > volume_size_plus_epsilon || diff(1) > volume_size_plus_epsilon || diff(2) > volume_size_plus_epsilon))
        {
            return traversed_voxels;
        }
    }      

    //add traversed voxel to list of voxels
    traversed_voxels.push_back(current_vol_center);
    return traversed_voxels;
}

/*
* Cost of sdf value
*/
inline double get_cost(const double & sdf){
    if(fabs(sdf) >= truncation_distance){
        return 0;
    }
    else if(sdf > 0){
        return pow((sdf - truncation_distance), 2) / (2* truncation_distance);
    }else{
        return sdf * -1 + .5 * truncation_distance;
    }

}

inline double get_voxel_cost(const Eigen::Matrix<double, 3, 1> & voxel_pos){

    size_t bucket = get_bucket(voxel_pos);

    for(Voxel v : voxels_set[bucket]){
        if(check_floating_point_vectors_equal(voxel_pos, v.position, voxel_size)){
            return get_cost(v.sdf);
        }
    }

    return 0; //voxel does not exist so it is in free space(or inside an object) and return 0 cost
}

/*
* Compute cost gradient for a voxel specified by voxel_pos. Check cost values of voxel at voxel_pos and voxels around
*/
Eigen::Matrix<double, 3, 1> get_voxel_cost_gradient(const Eigen::Matrix<double, 3, 1> & voxel_pos){
    double cost = get_voxel_cost(voxel_pos);
    Eigen::Matrix<double, 3, 1> gradient_val;

    Eigen::Matrix<double, 3, 1> xNext(voxel_pos(0) + 5 * voxel_size, voxel_pos(1), voxel_pos(2));
    Eigen::Matrix<double, 3, 1> xPrev(voxel_pos(0) - 5 * voxel_size, voxel_pos(1), voxel_pos(2));
    double xDiffNext = get_voxel_cost(xNext) - cost;
    double xDiffPrev = cost - get_voxel_cost(xPrev);
    gradient_val(0) = (xDiffNext + xDiffPrev) / 2;

    Eigen::Matrix<double, 3, 1> yNext(voxel_pos(0), voxel_pos(1) + 5 * voxel_size, voxel_pos(2));
    Eigen::Matrix<double, 3, 1> yPrev(voxel_pos(0), voxel_pos(1) - 5 * voxel_size, voxel_pos(2));
    double yDiffNext = get_voxel_cost(yNext) - cost;
    double yDiffPrev = cost - get_voxel_cost(yPrev);
    gradient_val(1) = (yDiffNext + yDiffPrev) / 2;

    Eigen::Matrix<double, 3, 1> zNext(voxel_pos(0), voxel_pos(1), voxel_pos(2) + 5 * voxel_size);
    Eigen::Matrix<double, 3, 1> zPrev(voxel_pos(0), voxel_pos(1), voxel_pos(2) - 5 * voxel_size);
    double zDiffNext = get_voxel_cost(zNext) - cost;
    double zDiffPrev = cost - get_voxel_cost(zPrev);
    gradient_val(2) = (zDiffNext + zDiffPrev) / 2;

    return gradient_val;
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
    Eigen::MatrixXd K1 = K * K / pow(delta_t,2);
    Eigen::MatrixXd K2 = K * K * K / pow(delta_t,3);
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

double obstacle_avoidance(const cinematography_msgs::msg::MultiDOFarray& drone_traj) {
    std::vector<cinematography_msgs::msg::MultiDOF> points = drone_traj.points;
    double cost = 0;
    for(size_t i = 0; i<points.size()-1; ++i){
        cinematography_msgs::msg::MultiDOF point_start = points[i];
        cinematography_msgs::msg::MultiDOF point_end = points[i+1];
        std::vector<Eigen::Matrix<double, 3, 1>> traversed_voxels = get_voxels(point_start, point_end, voxel_size); //get traversed voxels between each drone point
        double velocity = sqrt(pow(point_end.vx, 2) + pow(point_end.vy ,2) + pow(point_end.vz , 2));

        for(size_t j = 1; j<traversed_voxels.size(); ++j){ //skip first voxel for each drone point so not double counting voxel cost
            cost += get_voxel_cost(traversed_voxels[j]) * velocity;
        }
    }
    return cost;
}

double occlusion_avoidance(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& actor_traj) { 
    std::vector<cinematography_msgs::msg::MultiDOF> drone_points = drone_traj.points;
    std::vector<cinematography_msgs::msg::MultiDOF> actor_points = actor_traj.points;
    double cost = 0;
    for(size_t i = 1; i<drone_points.size(); ++i){ // skips first point since the drone is already there
      cinematography_msgs::msg::MultiDOF point_start = drone_points[i];
      cinematography_msgs::msg::MultiDOF point_end = actor_points[i];
      std::vector<Eigen::Matrix<double, 3, 1>> traversed_voxels = get_voxels(point_start, point_end, voxel_size); //get voxels on 2d manifold between actor and drone traj
      double drone_traj_velocity = sqrt(pow(point_start.vx, 2) + pow(point_start.vy ,2) + pow(point_start.vz , 2));
      double manifold_cost = 0;
      double manifold_velocity = sqrt(pow(point_end.x - point_start.x, 2) + pow(point_end.y - point_start.y ,2) + pow(point_end.z - point_start.z , 2));
      for(size_t j = 0; j<traversed_voxels.size(); ++j){
        //for each voxel point check in the current tsdf if it exists and if it does then get the cost based of the sdf value or 0 otherwise
        manifold_cost += get_voxel_cost(traversed_voxels[j]) * manifold_velocity;
      }
      cost+=manifold_cost * drone_traj_velocity;

    }
    return cost;
}

double traj_cost_function(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& actor_traj, cinematography_msgs::msg::MultiDOFarray& ideal_traj, double t) {      // TODO: Add 4th argument for TSDF
    double LAMBDA_1, LAMBDA_2, LAMBDA_3 = 1;    // TODO: Have these specified as ROS parameters

    return traj_smoothness(drone_traj, t) + LAMBDA_1 * obstacle_avoidance(drone_traj) + LAMBDA_2 * occlusion_avoidance(drone_traj, actor_traj) + LAMBDA_3 * shot_quality(drone_traj, ideal_traj);
}

//======================^^^==Cost functions==^^^====================================

//======================VVV==Gradient functions==VVV====================================

// TODO: Implement gradient equivalents of all the above, and hessian approximations (A_smooth + delta_1 * A_shot)
Eigen::Matrix<double, Eigen::Dynamic, 3> traj_smoothness_gradient(const cinematography_msgs::msg::MultiDOFarray& drone_traj, double delta_t, const Eigen::MatrixXd  & K, const Eigen::MatrixXd  & K0, const Eigen::MatrixXd  & K1, const Eigen::MatrixXd  & K2, const Eigen::MatrixXd  & A) {
    int a0 = 1, a1 = 0.5, a2 = 0;       // TODO: Fix these somehow

    int n = drone_traj.points.size();
    Eigen::MatrixXd q(n-1, 3);

    for(int i = 0; i < n-1; i++) {
        q(i,0) = drone_traj.points[i+1].x;      // Initialize matrix for drone trajectory
        q(i,1) = drone_traj.points[i+1].y;
        q(i,2) = drone_traj.points[i+1].z;
    }

    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(n-1,3);
    e(0,0) = -1 * drone_traj.points[0].x;
    e(0,1) = -1 * drone_traj.points[0].y;
    e(0,2) = -1 * drone_traj.points[0].z;

    Eigen::MatrixXd e0 = e / delta_t; //these need to be fixed
    Eigen::MatrixXd e1 = K * e / pow(delta_t, 2);
    Eigen::MatrixXd e2 = K * K * e / pow(delta_t, 3);

    Eigen::MatrixXd b = a0 * K0.transpose() * e0 + a1 * K1.transpose() * e1 + a2 * K2.transpose() * e2;

    Eigen::MatrixXd sum = A*q + b;

    return sum / (n-1);
}

Eigen::Matrix<double, Eigen::Dynamic, 3> shot_quality_gradient(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& ideal_traj, const Eigen::MatrixXd & A) {
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

    Eigen::MatrixXd b = K.transpose() * shot;

    Eigen::MatrixXd sum = A * q + b;
    return sum / (n-1);
}

Eigen::Matrix<double, Eigen::Dynamic, 3>  obstacle_avoidance_gradient(const cinematography_msgs::msg::MultiDOFarray& drone_traj){
    std::vector<cinematography_msgs::msg::MultiDOF> points = drone_traj.points;
    int n = points.size();
    Eigen::MatrixXd gradient_vals(n-1,3);
    gradient_vals = Eigen::MatrixXd::Zero(n-1,3);

    for(size_t i = 0; i<n-1; ++i){
        cinematography_msgs::msg::MultiDOF point_start = points[i];
        cinematography_msgs::msg::MultiDOF point_end = points[i+1];

        std::vector<Eigen::Matrix<double, 3, 1>> traversed_voxels = get_voxels(point_start, point_end, voxel_size);

        double velocity_mag = sqrt(pow(point_end.vx, 2) + pow(point_end.vy ,2) + pow(point_end.vz , 2));

        Eigen::Matrix<double, 3, 1> point_acceleration(point_end.ax, point_end.ay, point_end.az);

        Eigen::Matrix<double, 3, 1> p_hat(point_end.vx/velocity_mag, point_end.vy/velocity_mag, point_end.vz/velocity_mag);
        if(isnan(p_hat(0)) || isnan(p_hat(1)) || isnan(p_hat(2))){
          p_hat(0) = 0;
          p_hat(1) = 0;
          p_hat(2) = 0;
        }
        Eigen::Matrix<double, 3, 3> p_hat_multiplied = p_hat * p_hat.transpose();
        Eigen::Matrix<double, 3, 3> identity_minus_p_hat_multiplied = Eigen::Matrix3d::Identity(3,3) - p_hat_multiplied;
        Eigen::Matrix<double, 3, 1> gradient_val(0,0,0);

        for(size_t j = 1; j<traversed_voxels.size(); ++j){ //skip first voxel for each drone point so not double counting voxel cost
            Eigen::Matrix<double, 3, 1> cost_function_gradient = get_voxel_cost_gradient(traversed_voxels[j]);
            // Eigen::Matrix<double, 3, 1> gradient_multiplied_result = identity_minus_p_hat_multiplied * cost_function_gradient;
            // double voxel_cost = get_voxel_cost(traversed_voxels[j]);
            // Eigen::Matrix<double, 3, 1> k = ((1 / pow(velocity_mag, 2)) * identity_minus_p_hat_multiplied) * point_acceleration;
            // Eigen::Matrix<double, 3, 1> inner_term = gradient_multiplied_result - (voxel_cost * k);
            // Eigen::Matrix<double, 3, 1> grad_j_obs = inner_term * velocity_mag;
            // if(!(isnan(grad_j_obs(0)) || isnan(grad_j_obs(1)) || isnan(grad_j_obs(2)))){
            //     gradient_val+=grad_j_obs;
            // }      
            gradient_val += cost_function_gradient;    
        }
        //normalize
        gradient_val/=traversed_voxels.size();

        gradient_vals(i, 0) += gradient_val(0);
        gradient_vals(i, 1) += gradient_val(1);
        gradient_vals(i, 2) += gradient_val(2);

        for(int k = i + 1; k < i + 5; ++k){
            if(k >=  n-1){
                break;
            }

            int iter = k - i;

            gradient_vals(k,0) += (1 - .2 * iter) * gradient_val(0);
            gradient_vals(k,1) += (1 - .2 * iter) * gradient_val(1);
            gradient_vals(k,2) += (1 - .2 * iter) * gradient_val(2);
        }

        for(int k = i - 1; k > i - 5; --k){
            if(k < 0){
                break;
            }

            int iter = i - k;

            gradient_vals(k,0) += (1 - .2 * iter) * gradient_val(0);
            gradient_vals(k,1) += (1 - .2 * iter) * gradient_val(1);
            gradient_vals(k,2) += (1 - .2 * iter) * gradient_val(2);
        }
    }
    return gradient_vals;
}

Eigen::Matrix<double, Eigen::Dynamic, 3>  occlusion_avoidance_gradient(const cinematography_msgs::msg::MultiDOFarray& drone_traj, const cinematography_msgs::msg::MultiDOFarray& actor_traj){
    std::vector<cinematography_msgs::msg::MultiDOF> drone_points = drone_traj.points;
    std::vector<cinematography_msgs::msg::MultiDOF> actor_points = actor_traj.points;
    int n = drone_points.size();
    Eigen::MatrixXd gradient_vals(n-1,3);
    gradient_vals = Eigen::MatrixXd::Zero(n-1,3);
    for(size_t i = 1; i<n; ++i){ // skips first point since drone is already there
        cinematography_msgs::msg::MultiDOF point_start = drone_points[i];
        cinematography_msgs::msg::MultiDOF point_end = actor_points[i];
        std::vector<Eigen::Matrix<double, 3, 1>> traversed_voxels = get_voxels(point_start, point_end, voxel_size);
        double manifold_cost = 0;
        //velocity of the line between the drone and actor traj
        double manifold_velocity = sqrt(pow(point_end.x - point_start.x, 2) + pow(point_end.y - point_start.y ,2) + pow(point_end.z - point_start.z , 2));
        
        Eigen::Matrix<double, 1, 3> gradient_val(0,0,0);

        Eigen::Matrix<double, 3, 1> drone_point_velocity(point_start.vx, point_start.vy, point_start.vz);
        double drone_point_velocity_mag = drone_point_velocity.norm();
        Eigen::Matrix<double, 3, 1> normalized_drone_point_velocity = drone_point_velocity/drone_point_velocity_mag; 
        //if normalized_drone_point velocity is 0
        if(isnan(normalized_drone_point_velocity(0)) || isnan(normalized_drone_point_velocity(1) || isnan(normalized_drone_point_velocity(2)))){
          normalized_drone_point_velocity(0) = 0;
          normalized_drone_point_velocity(1) = 0;
          normalized_drone_point_velocity(2) = 0;
        }
        Eigen::Matrix<double, 1, 3> normalized_drone_point_velocity_transpose = normalized_drone_point_velocity.transpose();
        
        Eigen::Matrix<double, 3, 1> actor_point_velocity(point_end.vx, point_end.vy, point_end.vz);

        Eigen::Matrix<double, 3, 1> L(point_end.x - point_start.x, point_end.y - point_start.y, point_end.z - point_start.z);
        double L_mag = L.norm();
        Eigen::Matrix<double, 3, 1> normalized_L = L/L_mag;
        Eigen::Matrix<double, 1, 3> normalized_L_transpose = normalized_L.transpose();
        Eigen::Matrix<double, 3, 1> L_velocity = actor_point_velocity - drone_point_velocity;

        //used for determining the value of τ
        double increment = 1.0/(traversed_voxels.size()-1);

        for(size_t j = 0; j<traversed_voxels.size(); ++j){
            Eigen::Matrix<double, 3, 1> cost_function_gradient = get_voxel_cost_gradient(traversed_voxels[j]);
            // Eigen::Matrix<double, 3, 1> inner_first_term = actor_point_velocity/drone_point_velocity_mag - normalized_drone_point_velocity;
            // inner_first_term*=j*increment;
            // inner_first_term +=normalized_drone_point_velocity;
            // Eigen::Matrix<double, 3, 3> inner_first_term_matrix = inner_first_term * normalized_drone_point_velocity_transpose;
            // inner_first_term_matrix = Eigen::Matrix3d::Identity(3,3) - inner_first_term_matrix;
            // Eigen::Matrix<double, 1, 3> first_term = cost_function_gradient.transpose() * L_mag * drone_point_velocity_mag * inner_first_term_matrix;

            // Eigen::Matrix<double, 1, 3> inner_second_term = normalized_L_transpose + normalized_L_transpose * L_velocity * normalized_drone_point_velocity_transpose;
            // Eigen::Matrix<double, 1, 3> secondTerm = get_voxel_cost(traversed_voxels[j]) * drone_point_velocity_mag * inner_second_term;
            // Eigen::Matrix<double, 3, 1> grad_j_occ = first_term - secondTerm; 
            // if(!(isnan(grad_j_occ(0)) || isnan(grad_j_occ(1)) || isnan(grad_j_occ(2)))){
            //     gradient_val+=grad_j_occ;
            // }  
            gradient_val += cost_function_gradient;
        }
        // normalize 
        gradient_val/=traversed_voxels.size();
        gradient_vals(i-1, 0) += gradient_val(0);
        gradient_vals(i-1, 1) += gradient_val(1);
        gradient_vals(i-1, 2) += gradient_val(2); 

        for(int k = i + 1; k < i + 5; ++k){
            if(k >=  n-1){
                break;
            }

            int iter = k - i;

            gradient_vals(k,0) += (1 - .2 * iter) * gradient_val(0);
            gradient_vals(k,1) += (1 - .2 * iter) * gradient_val(1);
            gradient_vals(k,2) += (1 - .2 * iter) * gradient_val(2);
        }

        for(int k = i - 1; k > i - 5; --k){
            if(k <= 0){
                break;
            }

            int iter = i - k;

            gradient_vals(k,0) += (1 - .2 * iter) * gradient_val(0);
            gradient_vals(k,1) += (1 - .2 * iter) * gradient_val(1);
            gradient_vals(k,2) += (1 - .2 * iter) * gradient_val(2);
        }
    }
    return gradient_vals;
}

//======================^^^==Gradient functions==^^^====================================

void optimize_trajectory(cinematography_msgs::msg::MultiDOFarray& drone_traj, const cinematography_msgs::msg::MultiDOFarray& actor_traj) {
    cinematography_msgs::msg::MultiDOFarray ideal_traj = calc_ideal_drone_traj(actor_traj);     // ξ_shot when calculating shot quality

    int normalization = 1; // todo: change

    double t = 0;

    double LAMBDA_2, LAMBDA_3 = 1;    // TODO: Have these specified as ROS parameters
    double LAMBDA_1 = 15;


    int n = drone_traj.points.size();

    std::vector<cinematography_msgs::msg::MultiDOF> drone_points = drone_traj.points;
    std::vector<cinematography_msgs::msg::MultiDOF> actor_points = actor_traj.points;
    for(size_t i=0; i<n; ++i){
        t += actor_points[i].duration;
    }
    t /= n;

    //Intializing A_smooth
    int a0 = 1, a1 = 0.5, a2 = 0;  // TODO: Fix these somehow
    double e_1 = .01;

    Eigen::MatrixXd K = Eigen::MatrixXd::Identity(n-1,n-1);
    for(int i = 0; i < n-1; i++) {
        K(i,i) = -1;                            // Initialize K (n-1,n-1) as -I
    }
    Eigen::MatrixXd K0 = K / t;
    Eigen::MatrixXd K1 = K * K / pow(t,2);
    Eigen::MatrixXd K2 = K * K * K / pow(t,3);
    Eigen::MatrixXd A_smooth = a0 * K0.transpose() * K0 + a1 * K1.transpose() * K1 + a2 * K2.transpose() * K2;

    //Intializing A_shot
    Eigen::MatrixXd A_shot = Eigen::MatrixXd::Identity(n-1,n-1);

    //Intializing M_inv
    Eigen::MatrixXd M_inv = (A_smooth + LAMBDA_3 * A_shot).inverse();
    
    // int curr_voxels_set_size = voxels_set_size;
    // printf("voxels_set_size: %d\n", curr_voxels_set_size);

    for(int i = 0; i < MAX_ITERATIONS; i++) {
        // auto start = std::chrono::high_resolution_clock::now();
        // Eigen::Matrix<double, Eigen::Dynamic, 3> smooth_grad = traj_smoothness_gradient(drone_traj, t, K, K0, K1, K2, A_smooth);
        // auto stop = std::chrono::high_resolution_clock::now(); 
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
        // std::cout << "smooth grad duration: ";
        // std::cout << duration.count() << std::endl;

        // auto start1 = std::chrono::high_resolution_clock::now();
        Eigen::Matrix<double, Eigen::Dynamic, 3> shot_grad = shot_quality_gradient(drone_traj, ideal_traj, A_shot);
        // auto stop1 = std::chrono::high_resolution_clock::now(); 
        // auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1); 
        // std::cout << "shot grad duration: ";
        // std::cout << duration1.count() << std::endl;

        // auto start2 = std::chrono::high_resolution_clock::now();
        Eigen::Matrix<double, Eigen::Dynamic, 3> obs_grad = obstacle_avoidance_gradient(drone_traj);
        // auto stop2 = std::chrono::high_resolution_clock::now(); 
        // auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2); 
        // std::cout << "obs grad duration: ";
        // std::cout << duration2.count() << std::endl;

        // auto start3 = std::chrono::high_resolution_clock::now();
        Eigen::Matrix<double, Eigen::Dynamic, 3> occ_grad = occlusion_avoidance_gradient(drone_traj, actor_traj);
        // auto stop3 = std::chrono::high_resolution_clock::now(); 
        // auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(stop3 - start3); 
        // std::cout << "occ grad duration: ";
        // std::cout << duration3.count() << std::endl;
    
        // Eigen::Matrix<double, Eigen::Dynamic, 3> j_grad = smooth_grad + LAMBDA_1 * obs_grad + LAMBDA_2 * occ_grad + LAMBDA_3 * shot_grad;
        Eigen::Matrix<double, Eigen::Dynamic, 3> j_grad = LAMBDA_1 * obs_grad + LAMBDA_2 * occ_grad + LAMBDA_3 * shot_grad;
        // Eigen::Matrix<double, Eigen::Dynamic, 3> j_grad = smooth_grad + LAMBDA_3 * shot_grad;


        Eigen::MatrixXd traj_change = (1 / normalization) * M_inv * j_grad;

        bool converged = true;

        for(size_t j = 0; j < n-1; ++j){
            drone_traj.points[j+1].x -= traj_change(j, 0);
            drone_traj.points[j+1].y -= traj_change(j, 1);
            drone_traj.points[j+1].z -= traj_change(j, 2);

            if(traj_change(i,0) > e_1 || traj_change(i,1) > e_1 || traj_change(i,2) > e_1){
                converged = false;
            }
        }

        update_traj_position_derivatives(drone_traj);

        if(converged){
            // RCLCPP_INFO(node->get_logger(), "Converged: %d\n", i);
            return;
        }
    }
}

//bool get_trajectory_fun(airsim_ros_pkgs::get_trajectory::Request &req, airsim_ros_pkgs::get_trajectory::Response &res)
// Get actor's predicted trajectory (in NED and radians)
void get_actor_trajectory(cinematography_msgs::msg::MultiDOFarray::SharedPtr actor_traj)
{
    if(first_time){
        global_start = std::chrono::high_resolution_clock::now();
        first_time = false;
    }

    auto start = std::chrono::high_resolution_clock::now();
    global_iterations++;

    if (actor_traj->points.size() == 0) {
        RCLCPP_ERROR(node->get_logger(), "Received empty actor path"); 
        return;
    }

    // TODO: Get this further up in the vision pipeline and pass it down. Also get velocity and acceleration info
    geometry_msgs::msg::PointStamped origin;
    origin.header.frame_id = drone_frame;
    origin.point.x = 0;
    origin.point.y = 0;
    origin.point.z = 0;
    geometry_msgs::msg::PointStamped point = tf_buffer->transform<geometry_msgs::msg::PointStamped>(origin, world_frame);
    cinematography_msgs::msg::MultiDOF currentState;
    currentState.x = point.point.x;
    currentState.y = point.point.y;
    currentState.z = point.point.z;
    // currentState.x = 0;
    // currentState.y = 0;
    // currentState.z = 0;
    currentState.vx = currentState.vy = currentState.vz = currentState.ax = currentState.ay = currentState.az = 0;

    cinematography_msgs::msg::MultiDOFarray drone_path;
    drone_path = calc_ideal_drone_traj(*actor_traj);        // Calculate the ideal observation point for every point in actor trajectory
    move_traj_start(drone_path, currentState);              // Skew ideal path, so it starts at the drone's current position

    ideal_traj_pub->publish(drone_path);
    
    optimize_trajectory(drone_path, *actor_traj);

    face_actor(drone_path, *actor_traj);                    // Set all yaws to fact their corresponding point in the actor trajectory

    // Publish the trajectory (for debugging purposes)
    drone_path.header.stamp = clock_->now();

    // RCLCPP_INFO(node->get_logger(), "Publishing drone trajectory\n");

    drone_traj_pub->publish(drone_path);

    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    // std::cout << "Actor Traj Callback Duration: ";
    // std::cout << duration.count() << std::endl;

    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop - global_start); 
    // std::cout << "Global Duration: ";
    // std::cout << duration1.count() << std::endl;

    // printf("num callbacks: %d\n", global_iterations);
    // printf("---------------------------------------------\n");

    // RCLCPP_INFO(node->get_logger(), "average time: %ld\n", duration1.count() / global_iterations);
}

void tsdf_callback(tsdf_package_msgs::msg::Tsdf::SharedPtr tsdf){
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<NUM_BUCKETS;++i){
        voxels_set[i].clear();
    }

    std::vector<tsdf_package_msgs::msg::Voxel> voxels = tsdf->voxels;
    voxels_set_size = voxels.size();
    for (tsdf_package_msgs::msg::Voxel v : voxels){
        Voxel val(v.sdf, v.x, v.y, v.z);
        size_t bucket = get_bucket(val.position);
        voxels_set[bucket].push_back(val);
    }

    if(!received_first_msg){
        truncation_distance = tsdf->truncation_distance;
        voxel_size = tsdf->voxel_size;
        received_first_msg = true;
    }
    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    // std::cout << "tsdf callback duration: ";
    // std::cout << duration.count() << std::endl;
}

int main(int argc, char ** argv)
{
    sleep(3);
    rclcpp::init(argc, argv);
    node = rclcpp::Node::make_shared("motion_planner");
    clock_ = node->get_clock();

    allocate_bucket_indices();

    node->declare_parameter<int>("max_iterations", 1);
    node->get_parameter("max_iterations", MAX_ITERATIONS);
    airsim_client = new msr::airlib::MultirotorRpcLibClient(airsim_hostname);

    tf_buffer = new tf2_ros::Buffer(node->get_clock()); 
    tf_listener = new tf2_ros::TransformListener(*tf_buffer);
    
    auto parameters_client = std::make_shared<rclcpp::SyncParametersClient>(node, "airsim_ros2_wrapper");
    while (!parameters_client->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(node->get_logger(), "Interrupted while waiting for the service. Exiting.");
            rclcpp::shutdown();
        }
        RCLCPP_INFO(node->get_logger(), "service not available, waiting again...");
    }
    airsim_hostname = parameters_client->get_parameter<std::string>("airsim_hostname");
    world_frame = parameters_client->get_parameter<std::string>("world_frame");
    drone_frame = parameters_client->get_parameter<std::string>("vehicle_name");

    while(!tf_buffer->canTransform(world_frame, drone_frame, tf2::TimePointZero)) {
        std::cout << "Waiting for world to drone frame transform..." << std::endl;
        sleep(1);
    };

    auto actor_traj_sub = node->create_subscription<cinematography_msgs::msg::MultiDOFarray>("actor_traj", 1, get_actor_trajectory); 
    auto tsdf_sub = node->create_subscription<tsdf_package_msgs::msg::Tsdf>("tsdf", 1, tsdf_callback);

    drone_traj_pub = node->create_publisher<cinematography_msgs::msg::MultiDOFarray>("drone_traj", 1);
    ideal_traj_pub = node->create_publisher<cinematography_msgs::msg::MultiDOFarray>("ideal_traj", 1);

    // Sleep
    rclcpp::spin(node);

    rclcpp::shutdown();
    
    return 0;
}