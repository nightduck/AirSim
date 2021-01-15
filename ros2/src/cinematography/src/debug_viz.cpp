#include "rclcpp/rclcpp.hpp"
#include <boost/make_shared.hpp>
#include "builtin_interfaces/msg/time.hpp"
#include "cinematography_msgs/msg/multi_do_farray.hpp"
#include "cinematography_msgs/msg/vision_measurements.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std::chrono_literals;
using std::placeholders::_1;

class DebugViz : public rclcpp::Node {
private:
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub;
    rclcpp::Subscription<cinematography_msgs::msg::MultiDOFarray>::SharedPtr drone_traj_sub;
    rclcpp::Subscription<cinematography_msgs::msg::MultiDOFarray>::SharedPtr actor_traj_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub;
    rclcpp::Subscription<cinematography_msgs::msg::VisionMeasurements>::SharedPtr vm_sub;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr actor_traj_pub;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr drone_traj_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub;

    std::recursive_mutex m;
    rclcpp::Clock clock = rclcpp::Clock(RCL_SYSTEM_TIME);
    std::string world_frame_id_ = "world_ned";

    cv::Mat lastFrame;

    void fetchDronePose(const geometry_msgs::msg::Pose::SharedPtr pose) {
        geometry_msgs::msg::PoseStamped ps;

        ps.pose = *pose;
        ps.header.frame_id = world_frame_id_;
        ps.header.stamp = clock.now();

        pose_pub->publish(ps);
    }

    geometry_msgs::msg::PoseArray convertTraj(const cinematography_msgs::msg::MultiDOFarray::SharedPtr traj) {
        geometry_msgs::msg::PoseArray traj_out;

        traj_out.header.frame_id = "world_ned";
        traj_out.header.stamp = this->now();

        traj_out.poses.reserve(traj->points.size());

        for(cinematography_msgs::msg::MultiDOF md : traj->points) {
            geometry_msgs::msg::Pose p;
            p.position.x = md.x;
            p.position.y = md.y;
            p.position.z = md.z;
            p.orientation.w = cos(md.yaw/2);
            p.orientation.x = 0;
            p.orientation.y = 0;
            p.orientation.z = sin(md.yaw/2);

            traj_out.poses.push_back(p);
        }

        return traj_out;
    }

    void fetchActorTraj(const cinematography_msgs::msg::MultiDOFarray::SharedPtr traj) {
        actor_traj_pub->publish(convertTraj(traj));
    }

    void fetchDroneTraj(const cinematography_msgs::msg::MultiDOFarray::SharedPtr traj) {
        drone_traj_pub->publish(convertTraj(traj));
    }

    void fetchImage(const sensor_msgs::msg::Image::SharedPtr img) {
        m.lock();
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
        lastFrame = cv_ptr->image;
        m.unlock();
    }

    void fetchVisionMeasurements(const cinematography_msgs::msg::VisionMeasurements::SharedPtr vm) {
        int baseline = 0;
        float font_scale = 0.5;
        int thickness = 2;

        cv::Point topLeftCorner = cv::Point((vm->centerx - vm->width/2) * lastFrame.cols,
                                            (vm->centery - vm->height/2) * lastFrame.rows);
        cv::Point bottomRightCorner = cv::Point((vm->centerx + vm->width/2) * lastFrame.cols,
                                                (vm->centery + vm->height/2) * lastFrame.rows);

        int length = lastFrame.cols / 10;
        cv::Point center = cv::Point(vm->centerx * lastFrame.cols, vm->centery * lastFrame.rows);
        cv::Point endpoint = cv::Point(vm->centerx * lastFrame.cols + length * sin(vm->hde),
                                       vm->centery * lastFrame.rows + length * cos(vm->hde));


        cv_bridge::CvImage cv_msg;
        cv_msg.header.frame_id = "world_ned";
        cv_msg.header.stamp = this->now();
        cv_msg.encoding = sensor_msgs::image_encodings::BGR8;

        m.lock();

        if (vm->width != 0) {
            // draw rectangle
            cv::rectangle(lastFrame, topLeftCorner, bottomRightCorner, cv::Scalar(50, 205, 50), 2); 

            //draw hde line
            cv::line(lastFrame, center, endpoint, cv::Scalar(0,0,255), 3);
        }

        cv_msg.image = lastFrame;
        img_pub->publish(cv_msg.toImageMsg());
        m.unlock();
    }
    

public:
    DebugViz() : Node("debug_viz") {
        pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("pose_out", 1);
        actor_traj_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("actor_traj_out", 1);
        drone_traj_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("drone_traj_out", 1);
        img_pub = this->create_publisher<sensor_msgs::msg::Image>("img_out", 20);
        
        pose_sub = this->create_subscription<geometry_msgs::msg::Pose>("pose_in", 1,
            std::bind(&DebugViz::fetchDronePose, this, _1));
        actor_traj_sub = this->create_subscription<cinematography_msgs::msg::MultiDOFarray>("actor_traj_in", 1,
            std::bind(&DebugViz::fetchActorTraj, this, _1));
        drone_traj_sub = this->create_subscription<cinematography_msgs::msg::MultiDOFarray>("drone_traj_in", 1,
            std::bind(&DebugViz::fetchDroneTraj, this, _1));
        img_sub = this->create_subscription<sensor_msgs::msg::Image>("img_in", 1,
            std::bind(&DebugViz::fetchImage, this, _1));
        vm_sub = this->create_subscription<cinematography_msgs::msg::VisionMeasurements>("vm_in", 1,
            std::bind(&DebugViz::fetchVisionMeasurements, this, _1));
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor exec;
    auto ros2wrapper = std::make_shared<DebugViz>();
    exec.add_node(ros2wrapper);
    exec.spin();
    rclcpp::shutdown();

    return 0;
}