#include "ros/ros.h"
#include <ros/spinner.h>
#include <signal.h>
#include "common_mav.h"
#include <visualization_msgs/Marker.h>
#include <airsim_ros_pkgs/profiling_data_srv.h>
#include <airsim_ros_pkgs/start_profiling_srv.h>


// Every coordinate in airsim is NED coordinate, which is used by Airsim
// +X is North, +Y is East and +Z is down

string g_stats_file_addr;
string ns;
std::string g_supervisor_mailbox; //file to write to when completed
std::string g_mission_status = "failed";
long long g_planning_time_acc = 0;
int g_planning_ctr = 0;
long long g_accumulate_loop_time = 0;
int g_main_loop_ctr = 0;

bool g_start_profiling = false;
bool clct_data = true;


double max_x = 0, max_y = 0, max_z = 0, min_x = 0, min_y = 0, min_z = 0;
enum State { setup, waiting, flying, completed, invalid };

// get the scanning width, length and number of lanes
void get_goal(int& width, int& length, int& lanes, int& height) {
    std::cout << "Enter width ,length, number of lanes and height"<<std::endl
        << "associated with the area you like to sweep "<<endl;

    std::cin >> width >> length >> lanes >> height;
}

bool out_of_bound(const graph::node& pos){
    double offset = 0.1;
    return (pos.x < min_x - offset
        || pos.x > max_x + offset
        || pos.y < min_y - offset
        || pos.y > max_y + offset
        || pos.z < min_z - offset
        || pos.z > max_z + offset);
}

std::vector<Vector3r> get_turn_points(int width, int length, int lanes, int height){
	std::vector<Vector3r> turn_points;
	double length_step = length / (double) (lanes);

	double y_coordinate = width;
	double x_coordinate = length_step;
	while(x_coordinate <= length){
        turn_points.push_back(Vector3r(x_coordinate,y_coordinate,height));
		if(y_coordinate == 0){
			y_coordinate = width;
		}
		else{
			y_coordinate = 0;
		}
		x_coordinate += length_step;
	}

    // set y_coordinate to current coordinate
    y_coordinate = width ? y_coordinate == 0 : 0;

    // deal with left-over 
    if(turn_points.size() < lanes){
        y_coordinate = width ? y_coordinate == 0 : 0;
        turn_points.push_back(Vector3r(length,y_coordinate,height));
    }

    // return to the beginning
    if(y_coordinate == 0){
        turn_points.push_back(Vector3r(0,0,height));
    }
    else{
        turn_points.push_back(Vector3r(length,0,height));
        turn_points.push_back(Vector3r(0,0,height));
    }
	return turn_points;
}


mav_trajectory_generation::Trajectory get_scanning_trajectory(std::vector<Vector3r> turn_points){
	//Vector3r.x_val
	
	mav_trajectory_generation::Vertex::Vector vertices;
    const int dimension = 3;
    const int derivative_to_optimize = mav_trajectory_generation::derivative_order::SNAP;

    // begin to create and push vertice
    mav_trajectory_generation::Vertex start(dimension);
    auto start_point = turn_points.at(0);
    start.makeStartOrEnd(Eigen::Vector3d(start_point.x(),start_point.y(),start_point.z()), derivative_to_optimize);
    vertices.push_back(start);

    for(unsigned i=1; i<turn_points.size()-1; i++){
    	auto point = turn_points.at(i);
    	mav_trajectory_generation::Vertex middle(dimension);
    	middle.addConstraint(mav_trajectory_generation::derivative_order::POSITION, Eigen::Vector3d(point.x(),point.y(),point.z()));
        vertices.push_back(middle);
    }

    Vector3r end_point = turn_points.at(turn_points.size()-1);
    mav_trajectory_generation::Vertex end(dimension);
    end.makeStartOrEnd(Eigen::Vector3d(end_point.x(),end_point.y(),end_point.z()), derivative_to_optimize);
    vertices.push_back(end);

    for(auto t=vertices.begin(); t!=vertices.end(); ++t){
        std::cout << *t << endl;
    }

    const int N = 10;
    mav_trajectory_generation::PolynomialOptimization<N> opt(dimension);
    bool col;
    do{
        col = false;
        std::vector<double> segment_times;
        const double v_max = max_velocity;
        const double a_max = max_accelaration;
        segment_times = estimateSegmentTimes(vertices, v_max, a_max);

        // Optimize and create a smooth path from the vertices
        opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
        opt.solveLinear();

        // Return all the smooth segments in the path
        // (Each segment goes from one of the original nodes to the next one in the path)
        mav_trajectory_generation::Segment::Vector segments;
        opt.getSegments(&segments);

        // Loop through the vector of segments looking for collisions
        for(int i=0; !col && i < segments.size(); ++i){
            const double time_step = 0.1;
            double segment_len = segments[i].getTime();

            auto segment_start = *(turn_points.begin() + i);
            auto segment_end = *(turn_points.begin() + i + 1);

            // Step through each individual segment, at increments of "time_step" seconds, looking for a collision
            for(double t=0; t<segment_len - time_step; t += time_step){
                auto pos1 = segments[i].evaluate(t);
                auto pos2 = segments[i].evaluate(t + time_step);

                graph::node n1 = {pos1.x(), pos1.y(), pos1.z()};
                graph::node n2 = {pos2.x(), pos2.y(), pos2.z()};

                // Check for a collision between two near points on the segment
                if (out_of_bound(n1) || out_of_bound(n2)){
                    // Add a new vertex in the middle of the segment we are currently on
                    mav_trajectory_generation::Vertex middle(dimension);

                    double middle_x = (segment_start.x() + segment_end.x()) / 2;
                    double middle_y = (segment_start.y() + segment_end.y()) / 2;
                    double middle_z = (segment_start.z() + segment_end.z()) / 2;

                    middle.addConstraint(mav_trajectory_generation::derivative_order::POSITION, Eigen::Vector3d(middle_x, middle_y, middle_z));

                    vertices.insert(vertices.begin()+i+1, middle);

                    // Add a new node to the turn_points where the vertex is
                    turn_points.insert(turn_points.begin() + i + 1, Vector3r(middle_x,middle_y,middle_z));
                    col = true;
                    break;
                }
            }

        }
    } while(col);


    // Return the collision-free smooth trajectory
    mav_trajectory_generation::Trajectory trajectory;
    opt.getTrajectory(&trajectory);
    return trajectory;
}


void log_data_before_shutting_down(){
    std::string ns = ros::this_node::getName();
    airsim_ros_pkgs::profiling_data_srv profiling_data_srv_inst;

    profiling_data_srv_inst.request.key = "mission_status";
    profiling_data_srv_inst.request.value = (g_mission_status == "completed" ? 1.0: 0.0);
    if (ros::service::waitForService("/record_profiling_data", 10)){ 
        if(!ros::service::call("/record_profiling_data",profiling_data_srv_inst)){
            ROS_ERROR_STREAM("could not probe data using stats manager");
            ros::shutdown();
        }
    }
    
    profiling_data_srv_inst.request.key = "scanning_main_loop average time";
    profiling_data_srv_inst.request.value = (((double)g_accumulate_loop_time)/1e9)/g_main_loop_ctr;
    if (ros::service::waitForService("/record_profiling_data", 10)){ 
        if(!ros::service::call("/record_profiling_data",profiling_data_srv_inst)){
            ROS_ERROR_STREAM("could not probe data using stats manager");
        }
    }

    profiling_data_srv_inst.request.key = "planning average time";
    profiling_data_srv_inst.request.value = ((double)g_planning_time_acc/g_planning_ctr)/1e9;
    if (ros::service::waitForService("/record_profiling_data", 10)){ 
        if(!ros::service::call("/record_profiling_data",profiling_data_srv_inst)){
            ROS_ERROR_STREAM("could not probe data using stats manager");
            ros::shutdown();
        }
    }
}

void sigIntHandlerPrivate(int signo){
    if (signo == SIGINT) {
        log_data_before_shutting_down(); 
        ros::shutdown();
    }
    exit(0);
}

int main(int argc, char ** argv)
{
    

	// initialize ROS
	ros::init(argc, argv, "scanning_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    AirsimROSWrapper airsim_ros_wrapper(nh, nh_private);
    uint32_t shape = visualization_msgs::Marker::CUBE;
    trajectory_t traject_t_trajectory;

    int width, length, lanes, height; // size of area to scan
    std::vector<Vector3r> path; // piecewise vertices
    mav_trajectory_generation::Trajectory scanning_trajectory; //smoothed trajectory
    yaw_strategy_t yaw_strategy = face_forward;

    //profiling
        ros::Time start_hook_t, end_hook_t;
        signal(SIGINT, sigIntHandlerPrivate);
        ros::ServiceClient record_profiling_data_client = 
          nh.serviceClient<airsim_ros_pkgs::profiling_data_srv>("/record_profiling_data");

        ros::ServiceClient start_profiling_client = 
          nh.serviceClient<airsim_ros_pkgs::start_profiling_srv>("/start_profiling");

        airsim_ros_pkgs::start_profiling_srv start_profiling_srv_inst;
        start_profiling_srv_inst.request.key = "";

        ros::Time loop_start_t(0,0); 
        ros::Time loop_end_t(0,0);

    // visualization setup
        visualization_msgs::Marker points, line_strip, line_list, drone_point;
        points.header.frame_id = line_strip.header.frame_id = line_list.header.frame_id = drone_point.header.frame_id = "world_enu";
        points.header.stamp = line_strip.header.stamp = line_list.header.stamp = drone_point.header.stamp = ros::Time::now();
        points.ns = line_strip.ns = line_list.ns = "points_and_lines";
        drone_point.ns = "drone_position";
        points.action = line_strip.action = line_list.action = drone_point.action = visualization_msgs::Marker::ADD;
        points.pose.orientation.w = line_strip.pose.orientation.w = line_list.pose.orientation.w = drone_point.pose.orientation.w= 1.0;

        points.id = 0;
        line_strip.id = 1;
        line_list.id = 2;
        drone_point.id = 3;

        points.type = visualization_msgs::Marker::POINTS;
        drone_point.type = shape;
        line_strip.type = visualization_msgs::Marker::LINE_STRIP;
        line_list.type = visualization_msgs::Marker::LINE_LIST;

        points.scale.x = 0.2;
        points.scale.y = 0.2;

        drone_point.scale.x = 1;
        drone_point.scale.y = 1;
        drone_point.scale.z = 1;

        line_strip.scale.x = 0.1;
        line_list.scale.x = 0.1;

        points.color.g = 1.0f;
        points.color.a = 1.0;

        drone_point.color.g = 0.1;
        drone_point.color.a = 1.0;
        drone_point.color.b = 1.0;

        line_strip.color.b = 1.0;
        line_strip.color.a = 1.0;

        line_list.color.r = 1.0;
        line_list.color.a = 1.0;
        

    // airsim_ros_wrapper setup
        if (airsim_ros_wrapper.is_used_img_timer_cb_queue_)
        {
            airsim_ros_wrapper.img_async_spinner_.start();
        }

        if (airsim_ros_wrapper.is_used_lidar_timer_cb_queue_)
        {
            airsim_ros_wrapper.lidar_async_spinner_.start();
        }

        airsim_ros_wrapper.takeoff_jin();


    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("scanning_visualization_marker", 100);
    int scanning_loop_rate = 100;
    ros::Rate loop_rate(scanning_loop_rate);
    marker_pub.publish(drone_point);


    for (State state = setup; ros::ok(); ) {
        ros::spinOnce();
        State next_state = invalid;
        loop_start_t = ros::Time::now();

        if(state == setup){
            get_goal(width, length, lanes, height);
            std::cout << width << " " << length << "    " << lanes << " " << height << endl;
            std::cout << "----------------------" << endl;
            std::cout << "----------------------" << endl;
            max_x = length;
            max_y = width;
            max_z = height*2;

            airsim_ros_pkgs::profiling_data_srv profiling_data_srv_inst;
            profiling_data_srv_inst.request.key = "start_profiling";
            if (ros::service::waitForService("/record_profiling_data", 10)){ 
                if(!record_profiling_data_client.call(profiling_data_srv_inst)){
                    ROS_ERROR_STREAM("could not probe data using stats manager");
                    ros::shutdown();
                }
            }
           
            next_state = waiting;
        }
        else if(state == waiting){
            start_hook_t = ros::Time::now(); 

            //simple path
            path = get_turn_points(width, length, lanes, height);
            path.insert(path.begin(),Vector3r(0,0,height));

            //smooth trajectory
            scanning_trajectory = get_scanning_trajectory(path);

            //trajectory for airsim drone to fly with
            auto multiDof_trajectory = get_multiDOF_trajectory(scanning_trajectory);
            traject_t_trajectory = create_trajectory(multiDof_trajectory,true);
            cout << traject_t_trajectory.size() << endl;

            end_hook_t = ros::Time::now(); 
            g_planning_time_acc += ((end_hook_t - start_hook_t).toSec()*1e9);
            g_planning_ctr++;

            // prepare visulization
                for(int i = 0; i < traject_t_trajectory.size(); i++){
                    auto j = traject_t_trajectory[i];
                    geometry_msgs::Point p;
                    p.x = j.x;
                    p.y = j.y;
                    p.z = j.z;
                    points.points.push_back(p);
                    line_strip.points.push_back(p);

                    // The line list needs two points for each line
                    line_list.points.push_back(p);
                    p.z += 1.0;
                    line_list.points.push_back(p);
                }

            //visualize trajectory
            points.header.stamp = line_strip.header.stamp = line_list.header.stamp = drone_point.header.stamp = ros::Time::now();
            
            marker_pub.publish(points);
            marker_pub.publish(line_strip);
            marker_pub.publish(line_list);
            next_state = flying;
        }
        else if(state == flying){
            follow_trajectory(airsim_ros_wrapper, &traject_t_trajectory, nullptr, 
                    yaw_strategy, true ,max_velocity);

            //visualize drone position
            auto current_pos = airsim_ros_wrapper.getPosition();
            drone_point.pose.position.x = current_pos.y();
            drone_point.pose.position.y = current_pos.x();
            drone_point.pose.position.z = (-1)*current_pos.z();
            marker_pub.publish(drone_point);

            next_state = trajectory_done(traject_t_trajectory) ? completed : flying;
        }
        else if (state == completed){
            ROS_INFO("scanned the entire space and returned successfully");

            auto AllStates = airsim_ros_wrapper.getMultirotorState();
            msr::airlib::TripStats g_end_stats;
            g_end_stats = AllStates.getTripStats();
            std::cout << "end_voltage: " << g_end_stats.voltage << "," << std::endl;
            std::cout << "end StateOfCharge: " << g_end_stats.state_of_charge << std::endl;
            std::cout << "rotor energy consumed: " << g_end_stats.energy_consumed << std::endl; 

            airsim_ros_wrapper.end();

            g_mission_status = "completed";
            log_data_before_shutting_down();
            //signal_supervisor(g_supervisor_mailbox, "kill"); 
            ros::shutdown(); 
        }
        else if(state == invalid){
            cout << "invalid state !" << endl;
        }
        state = next_state;

        if (clct_data){
            if(!g_start_profiling) { 
                if (ros::service::waitForService("/start_profiling", 10)){ 
                    if(!start_profiling_client.call(start_profiling_srv_inst)){
                        ROS_ERROR_STREAM("could not probe data using stats manager");
                        ros::shutdown();
                    }
                    g_start_profiling = start_profiling_srv_inst.response.start; 
                }
            }
            else{
                loop_end_t = ros::Time::now(); 
                g_accumulate_loop_time += (((loop_end_t - loop_start_t).toSec())*1e9);
                g_main_loop_ctr++;
            }
        }
    }

	return 0;
}