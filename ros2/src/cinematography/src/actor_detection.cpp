#include "rclcpp/rclcpp.hpp"
#include "cinematography_msgs/msg/gimbal_angle_quat_cmd.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include "cinematography_msgs/msg/bounding_box.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
//#include <darknet/include/darknet.h>
#include <math.h>

#include <iostream>
#include <vector>
#include "tkDNN/tkdnn.h"
#include "tkDNN/Yolo3Detection.h"

using std::placeholders::_1;

#define CV_RES cv::Size(192, 192)
#define ACTOR_CLASS 1

class ActorDetection : public rclcpp::Node {
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr objdet_pub;
    rclcpp::Publisher<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_pub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera;
    msr::airlib::MultirotorRpcLibClient* airsim_client;
    tf2::Quaternion gimbal_setpoint;
    std::string camera_name = "front_center_custom";  // TODO: Set these as parameters
    std::string vehicle_name = "drone_1";
    float fov = M_PI/2;
    int gimbal_msg_count = 0;

    tk::dnn::Yolo3Detection *detNN;
    tk::dnn::Network *net;
    std::vector<tk::dnn::Layer*> outputs;

    std::string airsim_hostname;

    int difffilter(const std::vector<uint8_t> &v, std::vector<uint8_t> &w, int m, int n);

    void processImage(const sensor_msgs::msg::Image::SharedPtr msg) {
        // TODO: Add parameter to specify which type our actor is (deer, car, person, etc)
        // TODO: Run image through YOLO model and get bounding box coordiantes of object that is most confidently our actor

        std::vector<uint8_t> array;
        array.reserve(msg->data.size());
        //difffilter(msg->data, array, msg->width, msg->height);

        // Scale image to 192x192 square with black bars
        cv::Mat dest;
        cv_bridge::CvImagePtr cv_ptr_in = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::resize(cv_ptr_in->image, dest, CV_RES);
        

        // Perform inference
        std::vector<cv::Mat> dnn_input;
        dnn_input.push_back(cv_ptr_in->image);
        detNN->update(dnn_input);

        // Draw on boxes and publish for debugging purposes
        detNN->draw(dnn_input);
        cv_ptr_in->image = dnn_input[0];
        objdet_pub->publish(cv_ptr_in->toImageMsg());

        tk::dnn::box mostLikely;
        mostLikely.cl = -1;
        mostLikely.prob = 0;
        for(auto b: detNN->detected) {
            if (b.cl == ACTOR_CLASS && b.prob > mostLikely.prob) {
                mostLikely = b;
            }
        }


        return;

        // // Get 
        // for(auto d:detected){
        //     if(checkClass(d.cl, cam->dataset)){
        //         convertCameraPixelsToMapMeters((d.x + d.w / 2)*scale_x, (d.y + d.h)*scale_y, d.cl, *cam, north, east);
        //         tracking::obj_m obj;
        //         obj.frame   = 0;
        //         obj.cl      = d.cl;
        //         obj.x       = north;
        //         obj.y       = east;
        //         cur_frame.push_back(obj);
        //     }
        // }


        // //=================DUMMY LOAD==========================
        // uint8_t prev = msg->data[0];
        // int len = msg->data.size();
        // array[0] = prev;
        // for (int i = 1; i < 1000000; i++) {
        //     array[i%len] = msg->data[i%len] * prev;
        //     prev = array[i%len];
        // }
        // //==================DUMMY LOAD=========================
        float centerx = array[0] / 512.0 + 0.25;
        float centery = array[1] / 512.0 + 0.25;
        float width = array[2] / 1024.0;
        float height = array[3] / 1024.0;

        //DEBUGGING
        centerx = 0.45;
        centery = 0.5;
        width=0.15;
        height=0.2;

        // Use centering coordinates to move gimbal so actor is in center frame (OPTIONAL: Add parameter to specify
        // where in frame the actor should be)
        tf2::Quaternion horiz_adjustment = tf2::Quaternion(0, 0, sin(fov * (centerx - 0.5) / 2), cos(fov * (centerx - 0.5) / 2));
        tf2::Quaternion vert_adjustment = tf2::Quaternion(0, sin(fov * (centery + height/2 - 0.5) / 2), 0, cos(fov * (centery + height/2 - 0.5) / 2));
        tf2::Quaternion actor_direction = horiz_adjustment * gimbal_setpoint * vert_adjustment;   // Get the orientation in the direction of the actor (centered on bottom center)

        vert_adjustment = tf2::Quaternion(0, sin(fov * (centery - 0.5) / 2), 0, cos(fov * (centery - 0.5) / 2));
        gimbal_setpoint = horiz_adjustment * gimbal_setpoint * vert_adjustment;             // Get new gimbal setpoint (centered on actor)

        msr::airlib::Quaternionr alq = msr::airlib::Quaternionr(gimbal_setpoint.w(),
                        gimbal_setpoint.x(), gimbal_setpoint.y(), gimbal_setpoint.z());
        airsim_client->simSetCameraOrientation(camera_name, alq, vehicle_name);

        // Extract stk::dnn::Yolo3Detection *detNNubimage, package with centering coordinates, and publish to /bounding_box
        cinematography_msgs::msg::BoundingBox bb;
        bb.actor_direction = tf2::toMsg(actor_direction);

        int px_left = (centerx - width/2) * msg->width;
        int px_top = (centery - height/2) * msg->height;
        int px_width = width * msg->width;
        int px_height = height * msg->height;
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat cropped = cv_ptr->image(cv::Rect(px_left, px_top, px_width, px_height));
        cv_ptr->image = cropped;
        bb.image = *cv_ptr->toImageMsg();

        bb_pub->publish(bb);
    }

public:
    // TODO: Pass IP address of airsim as parameter
    ActorDetection() : Node("actor_detection") {
        declare_parameter<std::string>("airsim_hostname", "localhost");
        get_parameter("airsim_hostname", airsim_hostname);
        
        airsim_client = new msr::airlib::MultirotorRpcLibClient(airsim_hostname);
        gimbal_setpoint = tf2::Quaternion(0,0,0,1);
        bb_pub = this->create_publisher<cinematography_msgs::msg::BoundingBox>("bounding_box", 50);
        objdet_pub = this->create_publisher<sensor_msgs::msg::Image>("obj_detect_output", 50);
        camera = this->create_subscription<sensor_msgs::msg::Image>("camera", 50, std::bind(&ActorDetection::processImage, this, _1));

        // // Load in Yolo4 network
        // net = tk::dnn::darknetParser("yolo4/yolo-deer.cfg", "yolo4/layers", "yolo4/classes.names");
        
        // tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, net->getNetworkRTName("yolo4/yolo-deer.rt"));
        detNN = new tk::dnn::Yolo3Detection();
        detNN->init("yolo4/yolo3_fp32.rt", 2, 1);


        // // Generate list of output layers 
        // for(int i=0; i<net->num_layers; i++) {
        //     if(net->layers[i]->final)
        //         outputs.push_back(net->layers[i]);
        // }
        // // no final layers, set last as output
        // if(outputs.size() == 0) {
        //     outputs.push_back(net->layers[net->num_layers-1]);
        // }
    }

    ~ActorDetection() {
        delete net;
        delete airsim_client;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ActorDetection>());
    rclcpp::shutdown();

    return 0;
}
