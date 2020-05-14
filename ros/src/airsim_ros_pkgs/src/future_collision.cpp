#include "ros/ros.h"

// Standard headers
#include <chrono>
#include <string>
#include <cmath>
#include <signal.h>

// ROS message headers
#include "std_msgs/Bool.h"
#include "std_msgs/Int32.h"
#include "trajectory_msgs/MultiDOFJointTrajectory.h"

// Octomap specific headers
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/conversions.h>

// MAVBench headers
#include <airsim_ros_pkgs/BoolPlusHeader.h>
#include <airsim_ros_pkgs/multiDOF_array.h>

// Octomap server headers
#include <octomap_server/OctomapServer.h>

//airsim
#include "airsim_ros_wrapper.h"
#include <airsim_ros_pkgs/profiling_data_srv.h>
#include <airsim_ros_pkgs/start_profiling_srv.h>

using namespace octomap_server;

#include <visualization_msgs/Marker.h>
// collision visualization
visualization_msgs::Marker collision_point;


// Typedefs
typedef airsim_ros_pkgs::multiDOF_array traj_msg_t;
typedef std::chrono::system_clock sys_clock;
typedef std::chrono::time_point<sys_clock> sys_clock_time_point;
static const sys_clock_time_point never = sys_clock_time_point::min();

// Profiling variables
ros::Time start_hook_chk_col_t, end_hook_chk_col_t;                                          
long long g_checking_collision_kernel_acc = 0;
ros::Time g_checking_collision_t;
long long g_future_collision_main_loop = 0;
int g_check_collision_ctr = 0;
double g_distance_to_collision_first_realized = 0;
bool CLCT_DATA = true;
bool DEBUG = false;
ros::Time g_pt_cloud_header;    //this is used to figure out the octomap msg that 
                                //collision was detected in
long long g_pt_cloud_future_collision_acc = 0;
int g_octomap_rcv_ctr = 0;
ros::Duration g_pt_cloud_to_future_collision_t;


// Global variables
bool g_got_new_traj = false;
bool this_traj_already_has_collision = false;
ros::Time traj_timestamp;
ros::Time nextSteps_timestamp;
int traj_id = 0;
int nextSteps_id = 0;


octomap::OcTree * octree = nullptr;
traj_msg_t traj;
double drone_height__global = 0.6;
double drone_radius__global = 1.5;

AirsimROSWrapper* airsim_ros_wrapper_pointer;

//Profiling
int g_main_loop_ctr = 0;
long long g_accumulate_loop_time = 0; //it is in ms
long long g_pt_cld_to_octomap_commun_olverhead_acc = 0;

long long octomap_integration_acc = 0;
int octomap_ctr = 0;


bool occupied(octomap::OcTree * octree, double x, double y, double z){
  const double OCC_THRESH = 0.5;
  octomap::OcTreeNode * otn = octree->search(x, y, z);

  return otn != nullptr && otn->getOccupancy() >= OCC_THRESH;
}


template <class T>
bool collision(octomap::OcTree * octree, const T& n1, const T& n2)
{
	const double pi = 3.14159265359;

    const double height = drone_height__global; 
    const double radius = drone_radius__global; 

	const double angle_step = pi/4;

    using namespace octomap_server;
	const double radius_step = radius/3;
	const double height_step = height/2;

	double dx = n2.x - n1.x;
	double dy = n2.y - n1.y;
	double dz = n2.z - n1.z;

	double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

	octomap::point3d direction(dx, dy, dz);
	octomap::point3d end;

	for (double h = -height/2; h <= height/2; h += height_step) {
		for (double r = 0; r <= radius; r += radius_step) {
			for (double a = 0; a <= pi*2; a += angle_step) {
				octomap::point3d start(n1.x + r*std::cos(a), n1.y + r*std::sin(a), n1.z + h);

				if (octree->castRay(start, direction, end, true, distance)) {
					return true;
				}
			}
		}
	}

	return false;
}


template <class T>
double dist_to_collision(AirsimROSWrapper& airsim_ros_wrapper, const T& col_pos) {
    auto drone_pos = airsim_ros_wrapper.getPosition();

	double dx = abs(drone_pos.x() - col_pos.y);
	double dy = abs(drone_pos.y() - col_pos.x);
	double dz = abs(drone_pos.z()) - col_pos.z;

	return std::sqrt(dx*dx + dy*dy + dz*dz);
}


void pull_octomap(const octomap_msgs::Octomap& msg)
{
    if (octree != nullptr) {
        delete octree;
    }

	octomap::AbstractOcTree * tree = octomap_msgs::msgToMap(msg);
	octree = dynamic_cast<octomap::OcTree*> (tree);
    
    if (octree == nullptr) {
        ROS_ERROR("Octree could not be pulled.");
    }

    if (CLCT_DATA){ 
        g_pt_cloud_header = msg.header.stamp; 
        
        g_pt_cloud_future_collision_acc += (ros::Time::now() - g_pt_cloud_header).toSec()*1e9;
        g_octomap_rcv_ctr++;
    }

}


void pull_traj(const traj_msg_t::ConstPtr& msg)
{
    auto pos = airsim_ros_wrapper_pointer->getPosition();
    const auto& traj_front = msg->points.front();
    double x_offset = pos.x() - traj_front.y;
    double y_offset = pos.y() - traj_front.x;
    double z_offset = abs(pos.z()) - traj_front.z;

    traj = *msg;
    for (auto& point : traj.points){
        point.x += x_offset;
        point.y += y_offset;
        point.z += z_offset;
    }
    nextSteps_timestamp = msg->header.stamp;
    nextSteps_id = msg->traj_id;
}


bool check_for_collisions(AirsimROSWrapper& airsim_ros_wrapper, sys_clock_time_point& time_to_warn)
{
    start_hook_chk_col_t = ros::Time::now();

    if(traj_id != nextSteps_id){
        //ROS_INFO("traj_id and nextSteps_id doesn't match !");
        return false;
    }

    if(nextSteps_timestamp.sec < traj_timestamp.sec){
        //ROS_INFO("this is old next steps, ignore");
        return false;
    }

    const double min_dist_from_collision = 150.0;
    const std::chrono::milliseconds grace_period(1000);

    if (octree == nullptr || traj.points.size() < 1) {
        return false;
    };

    bool col = false;

    for (int i = 0; i < traj.points.size() - 1; ++i) {
        auto& pos1 = traj.points[i]; // .transforms[0].translation;
        auto& pos2 = traj.points[i+1]; // .transforms[0].translation;

        if (collision(octree, pos1, pos2)) {
            ROS_INFO("collision pos: %f, %f, %f", traj.points[i].x, traj.points[i].y, traj.points[i].z);
            geometry_msgs::Point p;
            p.x = pos1.x;
            p.y = pos1.y;
            p.z = pos1.z;

            collision_point.points.push_back(p);

            col = true;

            // Check whether the drone is very close to the point of collision
            auto now = sys_clock::now();
            if (dist_to_collision(airsim_ros_wrapper, pos1) < min_dist_from_collision){
                time_to_warn = now;
            }
            // Otherwise, give the drone a grace period to continue along its
            // path. Don't update the time_to_warn if it's already been set to
            // some time in the future
            else if (time_to_warn == never) {
                time_to_warn = now + grace_period;
            }

            break;
        }
    }

    if (!col){
        time_to_warn = never;
    }
    else{
        ROS_INFO("collision coming !");
    }
    

    end_hook_chk_col_t = ros::Time::now(); 
    g_checking_collision_t = end_hook_chk_col_t;
    g_checking_collision_kernel_acc += ((end_hook_chk_col_t - start_hook_chk_col_t).toSec()*1e9);
    g_check_collision_ctr++;
    return col;
}


void callback_trajectory(const airsim_ros_pkgs::multiDOF_array::ConstPtr& msg){
    g_got_new_traj = true;
    this_traj_already_has_collision = false;
    traj_timestamp = msg->header.stamp;
    traj_id = msg->traj_id;
}


void setup(){
    ros::param::get("/future_collision/drone_radius", drone_radius__global);
    ros::param::get("/future_collision/drone_height", drone_height__global);
}


void log_data_before_shutting_down(){

    std::string ns = ros::this_node::getName();
    airsim_ros_pkgs::profiling_data_srv profiling_data_srv_inst;
    
    
    profiling_data_srv_inst.request.key = "future_collision_kernel";
    profiling_data_srv_inst.request.value = (((double)g_checking_collision_kernel_acc)/1e9)/g_check_collision_ctr;
    if (ros::service::waitForService("/record_profiling_data", 10)){ 
        if(!ros::service::call("/record_profiling_data",profiling_data_srv_inst)){
            ROS_ERROR_STREAM("could not probe data using stats manager");
            ros::shutdown();
        }
    }

    profiling_data_srv_inst.request.key = "future_collision_main_loop";
    profiling_data_srv_inst.request.value = (((double)g_future_collision_main_loop)/1e9)/g_check_collision_ctr;
    if (ros::service::waitForService("/record_profiling_data", 10)){ 
        if(!ros::service::call("/record_profiling_data",profiling_data_srv_inst)){
            ROS_ERROR_STREAM("could not probe data using stats manager");
            ros::shutdown();
        }
    }

    profiling_data_srv_inst.request.key = "img_to_octomap_commun_t";
    profiling_data_srv_inst.request.value = ((double)g_pt_cld_to_octomap_commun_olverhead_acc/1e9)/octomap_ctr;
    if (ros::service::waitForService("/record_profiling_data", 10)){ 
        if(!ros::service::call("/record_profiling_data",profiling_data_srv_inst)){
            ROS_ERROR_STREAM("could not probe data using stats manager using octomap");
            ros::shutdown();
        }
    }
    
    profiling_data_srv_inst.request.key = "octomap_integration";
    profiling_data_srv_inst.request.value = (((double)octomap_integration_acc)/1e9)/octomap_ctr;
    if (ros::service::waitForService("/record_profiling_data", 10)){ 
        if(!ros::service::call("/record_profiling_data",profiling_data_srv_inst)){
            ROS_ERROR_STREAM("could not probe data using stats manager using octomap");
            ros::shutdown();
        }
    }

    ROS_INFO_STREAM("done with the octomap profiles");
}


void sigIntHandlerPrivate(int signo){
    if (signo == SIGINT) {
        log_data_before_shutting_down(); 
        ros::shutdown();
    }
    exit(0);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "future_collision", ros::init_options::NoSigintHandler);
    ros::NodeHandle n;
    ros::NodeHandle nh("~");
    signal(SIGINT, sigIntHandlerPrivate);

    setup();

    AirsimROSWrapper airsim_ros_wrapperb(n, nh);
    airsim_ros_wrapper_pointer = &airsim_ros_wrapperb;
    
    //----------------------------------------------------------------- 
	// *** F:DN variables	
	//----------------------------------------------------------------- 
    
    enum State {checking_for_collision, waiting_for_response};

    bool collision_coming = false;
    auto time_to_warn = never;

    airsim_ros_pkgs::BoolPlusHeader col_coming_msg;
    std_msgs::Bool col_imminent_msg;

    ros::Subscriber octomap_sub = nh.subscribe("/octomap_binary", 1, pull_octomap);

    ros::Subscriber traj_sub = nh.subscribe<traj_msg_t>("/next_steps", 1, pull_traj);
    ros::Publisher col_coming_pub = nh.advertise<airsim_ros_pkgs::BoolPlusHeader>("/col_coming", 1);

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("collision_visualization_marker", 100);
    ros::Subscriber trajectory_follower_sub = n.subscribe<airsim_ros_pkgs::multiDOF_array>("normal_traj", 1, callback_trajectory);

    State state, next_state;
    next_state = state = checking_for_collision;
    ros::Time main_loop_start_hook_t, main_loop_end_hook_t;
    
    // collision point visulization
        collision_point.header.frame_id = "world_enu";
        collision_point.header.stamp = ros::Time::now();
        collision_point.ns = "collision_position";
        collision_point.action = visualization_msgs::Marker::ADD;
        collision_point.pose.orientation.w= 1.0;

        collision_point.id = 3;

        collision_point.type = visualization_msgs::Marker::POINTS;

        collision_point.scale.x = 5;
        collision_point.scale.y = 5;

        collision_point.color.r = 255;
        collision_point.color.b = 255;
        collision_point.color.g = 255;
        collision_point.color.a = 0.5;


    ros::Rate loop_rate(60);
    while (ros::ok()) {
        main_loop_start_hook_t = ros::Time::now();

        marker_pub.publish(collision_point);

        ros::spinOnce();

        // if (CLCT_DATA){ 
        //     g_pt_cloud_header = server.rcvd_point_cld_time_stamp; 
        //     octomap_ctr = server.octomap_ctr;
        //     octomap_integration_acc = server.octomap_integration_acc; 
        //     g_pt_cld_to_octomap_commun_olverhead_acc = server.pt_cld_octomap_commun_overhead_acc;
        // }

        // State machine 
        if (state == checking_for_collision) {
            collision_coming = check_for_collisions(airsim_ros_wrapperb, time_to_warn);
            if (collision_coming && !this_traj_already_has_collision) {
                airsim_ros_wrapper_pointer->fly_velocity(0, 0, 0);
                next_state = waiting_for_response;

                col_coming_msg.header.stamp = g_pt_cloud_header;
                col_coming_msg.data = collision_coming;
                col_coming_pub.publish(col_coming_msg);
                g_got_new_traj = false; 
                this_traj_already_has_collision = true;
            }
            else if(collision_coming && this_traj_already_has_collision){
                ROS_INFO("this one already has collision, not published");
            }

            // Profiling 
            if(CLCT_DATA){ 
                g_pt_cloud_to_future_collision_t = start_hook_chk_col_t - g_pt_cloud_header;
            } 
            if(DEBUG) {
                ROS_INFO_STREAM("pt cloud to start of checking collision in future collision"<< g_pt_cloud_to_future_collision_t);
            }
        }else if (state == waiting_for_response) {
            if (g_got_new_traj){
                next_state = checking_for_collision;
            }
        }
        
        state = next_state;
        
        main_loop_end_hook_t = ros::Time::now();
        g_future_collision_main_loop += (main_loop_end_hook_t - main_loop_start_hook_t).toSec()*1e9; 

        loop_rate.sleep();
    }
}