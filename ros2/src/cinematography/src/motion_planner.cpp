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

double VOXEL_SIZE = .5; //todo: add this to tsdf msg
double HALF_VOXEL_SIZE = VOXEL_SIZE / 2;

double truncation_distance = 4;
double voxel_size = .5;
bool received_first_msg = false;

std::vector<Voxel> voxels_list;

std::unordered_map<Key, Voxel> voxel_map;

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

    //used to calculate velocity of each point
    cinematography_msgs::msg::MultiDOF last_point;
    //using previous point to calculate velocity so skip first point in for loop
    bool firstPoint = true;

    // For each point in the actor's trajectory...
    for (cinematography_msgs::msg::MultiDOF point : actor_traj.points) {
        cinematography_msgs::msg::MultiDOF n;

        // Center p on the actor to get the drone's ideal coordinates
        n.x = point.x + cos(viewport_heading + point.yaw) * horiz_dist;
        n.y = point.y + sin(viewport_heading + point.yaw) * horiz_dist;
        n.z = point.z - height;
        n.yaw = point.yaw + M_PI + viewport_heading;
        n.duration = point.duration;
        //set velocity
        if(!firstPoint){
            n.vx = (n.x - last_point.x) / n.duration;
            n.vy = (n.y - last_point.y) / n.duration;
            n.vz = (n.z - last_point.z) / n.duration;
        }
        else{
            firstPoint = false;
        }

        if (n.yaw > M_PI) {
            n.yaw -= 2*M_PI;
        }
        drone_traj.points.push_back(n);
        last_point = n;
    }

    //set velocity of first point
    if(drone_traj.points.size() > 1){
        drone_traj.points[0].vx = drone_traj.points[1].vx;
        drone_traj.points[0].vy = drone_traj.points[1].vy;
        drone_traj.points[0].vz = drone_traj.points[1].vz;
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

double floor_fun(const double & x, const double & scale){
    return floor(x*scale) / scale;
}

/*
* Given world point return voxel center
*/
Key get_voxel_key_from_point(const double & x, const double & y, const double & z){
    double scale = 1 / VOXEL_SIZE;
    Key k(floor_fun(x, scale) + HALF_VOXEL_SIZE, floor_fun(y, scale) + HALF_VOXEL_SIZE, floor_fun(z, scale) + HALF_VOXEL_SIZE);
    return k;
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
void get_voxels(const cinematography_msgs::msg::MultiDOF pointStart, const cinematography_msgs::msg::MultiDOF pointEnd, std::vector<Key> & voxelKeys, const double & volume_size){
    const double half_volume_size = volume_size / 2;
    const double epsilon = volume_size / 4;
    const double volume_size_plus_epsilon = volume_size + epsilon;
    const double volume_size_minus_epsilon = volume_size - epsilon;
    Eigen::Matrix<double, 3, 1> start(pointStart.x, pointStart.y, pointStart.z);
    Eigen::Matrix<double, 3, 1> end(pointEnd.x, pointEnd.y, pointEnd.z);
    //   equation of line = u+tv
    Eigen::Matrix<double, 3, 1> u(pointStart.x, pointStart.y, pointStart.z);
    Eigen::Matrix<double, 3, 1> v(pointEnd.x - pointStart.x, pointEnd.y - pointStart.y, pointEnd.z- pointStart.z);
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
        //add traversed voxel key to list of voxel keys
        Key voxel_key = get_voxel_key_from_point(current_vol_center(0), current_vol_center(1), current_vol_center(2));
        voxelKeys.push_back(voxel_key);
        
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
            return;
        }
    }      

    //add traversed voxel key to list of voxel keys
    Key voxel_key = get_voxel_key_from_point(current_vol_center(0), current_vol_center(1), current_vol_center(2));
    voxelKeys.push_back(voxel_key);
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

inline double get_voxel_cost(const Key k){
    std::unordered_map<Key,Voxel>::const_iterator it;
    it = voxel_map.find (k); //look for voxel if it exists, then it lies within truncation distance of a surface
    if ( it != voxel_map.end() )
    {
        Voxel voxel = it->second;
        return get_cost(voxel.sdf);
    }
    return 0; //voxel does not exist so it is in free space(or inside an object) and return 0 cost
}

/*
* Compute cost gradient for a voxel specified by key k. Check cost values of key k and voxels around
*/
Eigen::Matrix<double, 3, 1> get_cost_gradient(const Key k){
    double cost = get_voxel_cost(k);
    Eigen::Matrix<double, 3, 1> gradient_val;

    Key xNext(k.x + VOXEL_SIZE, k.y, k.z);
    Key xPrev(k.x - VOXEL_SIZE, k.y, k.z);
    double xDiffNext = get_voxel_cost(xNext) - cost;
    double xDiffPrev = cost - get_voxel_cost(xPrev);
    gradient_val(0) = (xDiffNext + xDiffPrev) / 2;

    Key yNext(k.x, k.y + VOXEL_SIZE, k.z);
    Key yPrev(k.x, k.y - VOXEL_SIZE, k.z);
    double yDiffNext = get_voxel_cost(yNext) - cost;
    double yDiffPrev = cost - get_voxel_cost(yPrev);
    gradient_val(1) = (yDiffNext + yDiffPrev) / 2;

    Key zNext(k.x, k.y, k.z + VOXEL_SIZE);
    Key zPrev(k.x, k.y, k.z - VOXEL_SIZE);
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
        cinematography_msgs::msg::MultiDOF pointStart = points[i];
        cinematography_msgs::msg::MultiDOF pointEnd = points[i+1];
        std::vector<Key> voxelKeys;
        get_voxels(pointStart, pointEnd, voxelKeys, VOXEL_SIZE); //get traversed voxels between each drone point
        double velocity = sqrt(pow(pointEnd.vx, 2) + pow(pointEnd.vy ,2) + pow(pointEnd.vz , 2));

        for(size_t j = 1; j<voxelKeys.size(); ++j){ //skip first voxel for each drone point so not double counting voxel cost
            cost += get_voxel_cost(voxelKeys[j]) * velocity;
        }
    }
    return cost;
}

double occlusion_avoidance(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& actor_traj) { 
    std::vector<cinematography_msgs::msg::MultiDOF> drone_points = drone_traj.points;
    std::vector<cinematography_msgs::msg::MultiDOF> actor_points = actor_traj.points;
    double cost = 0;
    for(size_t i = 1; i<drone_points.size(); ++i){ // skips first point since the drone is already there
      cinematography_msgs::msg::MultiDOF pointStart = drone_points[i];
      cinematography_msgs::msg::MultiDOF pointEnd = actor_points[i];
      std::vector<Key> voxelKeys;
      get_voxels(pointStart, pointEnd, voxelKeys, VOXEL_SIZE); //get voxels on 2d manifold between actor and drone traj
      double drone_traj_velocity = sqrt(pow(pointStart.vx, 2) + pow(pointStart.vy ,2) + pow(pointStart.vz , 2));
      double manifold_cost = 0;
      double manifold_velocity = sqrt(pow(pointEnd.x - pointStart.x, 2) + pow(pointEnd.y - pointStart.y ,2) + pow(pointEnd.z - pointStart.z , 2));
      for(size_t j = 0; j<voxelKeys.size(); ++j){
        //for each voxel point check in the current tsdf if it exists and if it does then get the cost based of the sdf value or 0 otherwise
        manifold_cost += get_voxel_cost(voxelKeys[j]) * manifold_velocity;
      }
      cost+=manifold_cost * drone_traj_velocity;

    }
    return cost;
}

double traj_cost_function(const cinematography_msgs::msg::MultiDOFarray& drone_traj, cinematography_msgs::msg::MultiDOFarray& actor_traj, cinematography_msgs::msg::MultiDOFarray& ideal_traj, double t) {      // TODO: Add 4th argument for TSDF
    double LAMBDA_1, LAMBDA_2, LAMBDA_3 = 1;    // TODO: Have these specified as ROS parameters

    return traj_smoothness(drone_traj, t) + LAMBDA_1 * obstacle_avoidance(drone_traj) + LAMBDA_2 * occlusion_avoidance(drone_traj, actor_traj) + LAMBDA_3 * shot_quality(drone_traj, ideal_traj);
}

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

void optimize_trajectory(cinematography_msgs::msg::MultiDOFarray& drone_traj, const cinematography_msgs::msg::MultiDOFarray& actor_traj) {
    cinematography_msgs::msg::MultiDOFarray ideal_traj = calc_ideal_drone_traj(actor_traj);     // ξ_shot when calculating shot quality

    int normalization = 1; // change

    std::vector<MultiDOF> drone_traj_cuda;
    std::vector<MultiDOF> actor_traj_cuda;

    double t = 0;

    double LAMBDA_1, LAMBDA_2, LAMBDA_3 = 1;    // TODO: Have these specified as ROS parameters

    int n = drone_traj.points.size();

    std::vector<cinematography_msgs::msg::MultiDOF> drone_points = drone_traj.points;
    std::vector<cinematography_msgs::msg::MultiDOF> actor_points = actor_traj.points;
    for(size_t i=0; i<n; ++i){
        cinematography_msgs::msg::MultiDOF p_d = drone_points[i];
        cinematography_msgs::msg::MultiDOF p_a = actor_points[i];
        MultiDOF drone_cuda_pt(p_d.x,p_d.y,p_d.z,p_d.vx,p_d.vy,p_d.vz,p_d.ax,p_d.ay,p_d.az,p_d.yaw,p_d.duration);
        MultiDOF actor_cuda_pt(p_a.x,p_a.y,p_a.z,p_a.vx,p_a.vy,p_a.vz,p_a.ax,p_a.ay,p_a.az,p_a.yaw,p_a.duration);
        drone_traj_cuda.push_back(drone_cuda_pt);
        actor_traj_cuda.push_back(actor_cuda_pt);
        t += p_a.duration;
    }
    t /= n;

    //Intializing A_smooth
    int a0 = 1, a1 = 0.5, a2 = 0;  // TODO: Fix these somehow

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

    printf("host map size: %lu\n", voxel_map.size());
    init_set_cuda(voxels_list);
    
    int MAX_ITERATIONS;     // TODO: Make this a ROS parameter
    for(int i = 0; i < 1; i++) {
        Eigen::Matrix<double, Eigen::Dynamic, 3> smooth_grad = traj_smoothness_gradient(drone_traj, t, K, K0, K1, K2, A_smooth);
        Eigen::Matrix<double, Eigen::Dynamic, 3> shot_grad = shot_quality_gradient(drone_traj, ideal_traj, A_shot);

        Eigen::Matrix<double, Eigen::Dynamic, 3> obs_grad = obstacle_avoidance_gradient_cuda(drone_traj_cuda, truncation_distance, voxel_size);
        Eigen::Matrix<double, Eigen::Dynamic, 3> occ_grad = occlusion_avoidance_gradient_cuda(drone_traj_cuda, actor_traj_cuda, truncation_distance, voxel_size);

        

        Eigen::Matrix<double, Eigen::Dynamic, 3> j_grad = smooth_grad + LAMBDA_1 * obs_grad + LAMBDA_2 * occ_grad + LAMBDA_3 * shot_grad;
        // Eigen::Matrix<double, Eigen::Dynamic, 3> j_grad = LAMBDA_1 * obs_grad + LAMBDA_2 * occ_grad;
        // Eigen::Matrix<double, Eigen::Dynamic, 3> j_grad = smooth_grad + LAMBDA_3 * shot_grad;

        //Todo:
        // if((j_grad.transpose()*M_inv * j_grad).array().pow(2) / 2 < e0){
          
        // }

        Eigen::MatrixXd traj_change = (1 / normalization) * M_inv * j_grad;
        for(size_t i = 0; i < n-1; ++i){
            drone_traj.points[i+1].x -= traj_change(i, 0);
            drone_traj.points[i+1].y -= traj_change(i, 1);
            drone_traj.points[i+1].z -= traj_change(i, 2);
            drone_traj_cuda[i+1].x -= traj_change(i,0);
            drone_traj_cuda[i+1].y -= traj_change(i,1);
            drone_traj_cuda[i+1].z -= traj_change(i,2);
        }
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
    // msr::airlib::Pose currentPose = airsim_client->simGetVehiclePose(vehicle_name);
    cinematography_msgs::msg::MultiDOF currentState;
    // currentState.x = currentPose.position.x();
    // currentState.y = currentPose.position.y();
    // currentState.z = currentPose.position.z();
    currentState.x = 0;
    currentState.y = 0;
    currentState.z = 0;
    currentState.vx = currentState.vy = currentState.vz = currentState.ax = currentState.ay = currentState.az = 0;

    cinematography_msgs::msg::MultiDOFarray drone_path;
    drone_path = calc_ideal_drone_traj(*actor_traj);        // Calculate the ideal observation point for every point in actor trajectory
    move_traj_start(drone_path, currentState);              // Skew ideal path, so it starts at the drone's current position

    face_actor(drone_path, *actor_traj);                    // Set all yaws to fact their corresponding point in the actor trajectory

    auto start = std::chrono::high_resolution_clock::now();
    optimize_trajectory(drone_path, *actor_traj);
    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout << "Optimization Duration: ";
    std::cout << duration.count() << std::endl;

    print_rviz_traj(drone_path, "drone_traj", false);

    // Publish the trajectory (for debugging purposes)
    drone_path.header.stamp = clock_->now();
    path_found = true;

    RCLCPP_INFO(node->get_logger(), "Publishing drone trajectory");

    traj_pub->publish(drone_path);
}

void tsdf_callback(tsdf_package_msgs::msg::Tsdf::SharedPtr tsdf){
    voxel_map.clear();
    voxels_list.clear();
    std::vector<tsdf_package_msgs::msg::Voxel> voxels = tsdf->voxels;
    for (tsdf_package_msgs::msg::Voxel v : voxels){
        Key k(v.x, v.y, v.z);
        Voxel val(v.sdf, v.x, v.y, v.z);
        voxel_map[k] = val;
        voxels_list.push_back(val);
    }

    if(!received_first_msg){
        truncation_distance = tsdf->truncation_distance;
        voxel_size = tsdf->voxel_size;
        received_first_msg = true;
    }
}

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    node = rclcpp::Node::make_shared("motion_planner");
    clock_ = node->get_clock();

    allocate_set();

    node->declare_parameter<std::string>("airsim_hostname", "localhost");
    node->get_parameter("airsim_hostname", airsim_hostname);
    airsim_client = new msr::airlib::MultirotorRpcLibClient(airsim_hostname);

    auto actor_traj_sub = node->create_subscription<cinematography_msgs::msg::MultiDOFarray>("/auto_cinematography/planning/actor_traj", 1, get_actor_trajectory); 
    auto tsdf_sub = node->create_subscription<tsdf_package_msgs::msg::Tsdf>("tsdf", 1, tsdf_callback);

    traj_pub = node->create_publisher<cinematography_msgs::msg::MultiDOFarray>("/multidoftraj", 1);

    rviz_actor_pub = node->create_publisher<geometry_msgs::msg::PoseArray>("/rviz_actor_traj", 1);
    rviz_drone_pub = node->create_publisher<geometry_msgs::msg::PoseArray>("/rviz_drone_traj", 1);

    // Sleep
    rclcpp::spin(node);

    rclcpp::shutdown();
    
    return 0;
}