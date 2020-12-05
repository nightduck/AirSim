#include "rclcpp/rclcpp.hpp"
#include "rclcpp/logging.hpp"
#include "cinematography_msgs/msg/bounding_box.hpp"
#include "cinematography_msgs/msg/vision_measurements.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <math.h>

#include "tkDNN/tkdnn.h"
#include "tkDNN/NetworkRT.h"

#define BATCH_SIZE 1

using std::placeholders::_1;

class HeadingEstimation : public rclcpp::Node {
private:
    rclcpp::Publisher<cinematography_msgs::msg::VisionMeasurements>::SharedPtr hde_pub;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sat_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_sub;

    // NOTE: These variables makes this node not safe for multithreaded spinning
    geometry_msgs::msg::TransformStamped t;         // Dummy variable for math-ing
    geometry_msgs::msg::Vector3Stamped unit_vect;
    geometry_msgs::msg::QuaternionStamped unit_quat;

    rclcpp::Clock clock = rclcpp::Clock(RCL_SYSTEM_TIME);

    tk::dnn::NetworkRT* netRT;
    dnnType *input_d;
    dnnType *out_d;
    const int nBatches = 1;

    #ifdef OPENCV_CUDACONTRIB
        cv::cuda::GpuMat bgr[3];
        cv::cuda::GpuMat imagePreproc;
    #else
        cv::Mat bgr[3];
        cv::Mat imagePreproc;
        dnnType *input;
    #endif

    void flatten(tf2::Quaternion &quat) {
        quat.setX(0);
        quat.setY(0);
        double length = sqrt(quat.w() * quat.w() + quat.z() * quat.z());
        quat.setW(quat.w() / length);
        quat.setZ(quat.z() / length);
    }

    void preprocess(cv::Mat &frame, const int bi) {
        //resize image, remove mean, divide by std
        cv::Mat frame_nomean;
        resize(frame, frame, cv::Size(netRT->input_dim.w, netRT->input_dim.h));
        frame.convertTo(frame_nomean, CV_32FC3, 1, -127);
        frame_nomean.convertTo(imagePreproc, CV_32FC3, 1 / 128.0, 0);

        //copy image into tensor and copy it into GPU
        cv::split(imagePreproc, bgr);
        for (int i = 0; i < netRT->input_dim.c; i++){
            int idx = i * imagePreproc.rows * imagePreproc.cols;
            memcpy((void *)&input[idx + netRT->input_dim.tot()*bi], (void *)bgr[i].data, imagePreproc.rows * imagePreproc.cols * sizeof(dnnType));
        }
        checkCuda(cudaMemcpyAsync(input_d+ netRT->input_dim.tot()*bi, input + netRT->input_dim.tot()*bi, netRT->input_dim.tot() * sizeof(dnnType), cudaMemcpyHostToDevice, netRT->stream));
    }

    void postprocess(){
        out_d = (float*)malloc(netRT->output_dim.tot() * sizeof(dnnType));
        checkCuda(cudaMemcpy(out_d, netRT->output, netRT->output_dim.tot() * sizeof(dnnType), cudaMemcpyDeviceToHost));

        out_d[0] = (out_d[0] * 2) - 1;
        out_d[1] = (out_d[1] * 2) - 1;
    }


    float* infer(cv::Mat in) {
        preprocess(in, 0);

        float total_time = 0;
        TKDNN_TSTART
        netRT->infer(netRT->input_dim, input_d);
        TKDNN_TSTOP
        total_time+= t_ns;

        postprocess();

        return out_d;
    }

    void processImage(const cinematography_msgs::msg::BoundingBox::SharedPtr msg) {
        int res;
        get_parameter("resolution", res);

        // Pad image to 192x192 square with black bars
        sensor_msgs::msg::Image img = msg->image;
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::BGR8);
        cv::Mat new_img;

        // Pad to square
        if (cv_ptr->image.rows != cv_ptr->image.cols) {
            int max_dim = (cv_ptr->image.rows > cv_ptr->image.cols) ? cv_ptr->image.rows : cv_ptr->image.cols;
            int vert_padding = (max_dim-cv_ptr->image.cols)/2;
            int horz_padding = (max_dim-cv_ptr->image.rows)/2;
            int right_px = (cv_ptr->image.cols % 2 != 0) ? 1 : 0;
            int bottom_px = (cv_ptr->image.rows % 2 != 0) ? 1 : 0;
            cv::copyMakeBorder(cv_ptr->image, new_img, horz_padding, horz_padding + bottom_px,
                    vert_padding, vert_padding + right_px, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
            cv_ptr->image = new_img;
        }

        float* array = infer(cv_ptr->image);

        double hde0 = array[0];   // Output #1
        double hde1 = array[1];   // Output #2
        // TODO: Normalize to [-1,1] range

        // Get the rotation of the actor relative to camera
        double angle = atan2(hde1, hde0);
        // tf2::Quaternion quat_actor_rel;
        // quat_actor_rel.setRPY(0, 0, angle);

        // // Convert drone's orientation to usable format
        // tf2::Quaternion quat_drone;
        // tf2::fromMsg(msg->drone_pose.orientation, quat_drone);

        // // Express the drone's offset within frame as quaternion rotations
        // tf2::Quaternion horiz_offset = tf2::Quaternion(0, 0, sin(msg->fov * (msg->centerx - 0.5) / 2), cos(msg->fov * (msg->centerx - 0.5) / 2));
        // tf2::Quaternion vert_offset = tf2::Quaternion(0, sin(msg->fov * (msg->centery + msg->height/2 - 0.5) / -2), 0, cos(msg->fov * (msg->centery + msg->height/2 - 0.5) / -2));
        // tf2::Quaternion actor_direction = horiz_offset * quat_drone * vert_offset;   // Get the orientation in the direction of the actor (centered on bottom center)

        // // Compute a unit vector pointing at the actor
        // t.transform.rotation = tf2::toMsg(actor_direction);
        // t.transform.translation = geometry_msgs::msg::Vector3();
        // geometry_msgs::msg::Vector3Stamped actor_ray;
        // tf2::doTransform(unit_vect, actor_ray, t);

        // // Project actor ray to ground, resulting vector being position of actor
        // double scale = abs(msg->drone_pose.position.z / actor_ray.vector.z);
        // actor_ray.vector.x *= scale;
        // actor_ray.vector.y *= scale;
        // actor_ray.vector.z *= scale;

        // // Flatten drone's quaternions (so they only represent yaw), and use that to calculate the
        // // actor's absolute rotation
        // flatten(actor_direction);
        // flatten(quat_drone);
        // tf2::Quaternion quat_actor_absolute = actor_direction * quat_drone;


        // NOTE: RESUME HERE. Verify the actor's absolute rotation is being calcualted correctly from it's relative rotation
        // Combine actor's position and rotation (adjusted from relative to absolute)
        cinematography_msgs::msg::VisionMeasurements vm;
        vm.hde = angle;
        vm.centerx = msg->centerx;
        vm.centery = msg->centery;
        vm.width = msg->width;
        vm.height = msg->height;
        vm.fov = msg->fov;
        vm.drone_pose = msg->drone_pose;

        // geometry_msgs::msg::PoseStamped actor_pose;
        // actor_pose.pose.position.x = actor_ray.vector.x + msg->drone_pose.position.x;
        // actor_pose.pose.position.y = actor_ray.vector.y + msg->drone_pose.position.y;
        // actor_pose.pose.position.z = actor_ray.vector.z + msg->drone_pose.position.z;
        // actor_pose.pose.orientation = tf2::toMsg(quat_actor_absolute);

        // // Publish to rviz, as debugging step rviz_pose;
        // actor_pose.header.frame_id = "world_ned";
        // actor_pose.header.stamp = clock.now();
        hde_pub->publish(vm);
    }


public:
    HeadingEstimation() : Node("heading_estimation") {
        declare_parameter("resolution", 192);

        unit_vect.vector.x = unit_quat.quaternion.x = unit_quat.quaternion.w = 1;
        unit_vect.vector.y = unit_vect.vector.z = unit_quat.quaternion.y = unit_quat.quaternion.z = 0;

        hde_pub = this->create_publisher<cinematography_msgs::msg::VisionMeasurements>("vision_measurements", 50);
        bb_sub = this->create_subscription<cinematography_msgs::msg::BoundingBox>("bounding_box", 50, std::bind(&HeadingEstimation::processImage, this, _1));

        netRT = new tk::dnn::NetworkRT(NULL, "hde_deer_airsim.rt");
        // These 3 lines fix a bug in the NetworkRT constructor
        // netRT->input_dim = tk::dnn::dataDim_t(1, 3, 192, 192, 1);
        netRT->output_dim.w = 1;    // There's a bug where this will be set to zero. It should be 1 minimum
        
        // Allocate memory for input buffers
        checkCuda(cudaMallocHost(&input, sizeof(dnnType) * netRT->input_dim.tot() * nBatches));
        checkCuda(cudaMalloc(&input_d, sizeof(dnnType) * netRT->input_dim.tot() * nBatches));
    }
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    sleep(10);
    rclcpp::spin(std::make_shared<HeadingEstimation>());
    rclcpp::shutdown();

    return 0;
}