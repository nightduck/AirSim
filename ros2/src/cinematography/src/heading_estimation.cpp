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

// #include "tkDNN/tkdnn.h"
// #include "tkDNN/NetworkRT.h"
#include <NvInfer.h>
#include "cudaWrapper.h"
#include "ioHelper.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <math.h>
#include <cmath>
#include <unistd.h>


#define BATCH_SIZE 1

using std::placeholders::_1;

class HeadingEstimation : public rclcpp::Node {
private:
    rclcpp::Publisher<cinematography_msgs::msg::VisionMeasurements>::SharedPtr hde_pub;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sat_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<cinematography_msgs::msg::BoundingBox>::SharedPtr bb_sub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bb_pub;

    // NOTE: These variables makes this node not safe for multithreaded spinning
    geometry_msgs::msg::TransformStamped t;         // Dummy variable for math-ing
    geometry_msgs::msg::Vector3Stamped unit_vect;
    geometry_msgs::msg::QuaternionStamped unit_quat;

    std::string trt_engine_filename;

    rclcpp::Clock clock = rclcpp::Clock(RCL_SYSTEM_TIME);

    // tk::dnn::NetworkRT* netRT;
    // //tk::dnn::Network* net;
    // dnnType *input_d;
    const int nBatches = 1;

    #ifdef OPENCV_CUDACONTRIB
        cv::cuda::GpuMat bgr[3];
        cv::cuda::GpuMat imagePreproc;
    #else
        cv::Mat bgr[3];
        cv::Mat imagePreproc;
        //dnnType *input;
    #endif

    //==========================AI=============================================
    nvinfer1::Logger gLogger;

    // Declaring cuda engine.
    std::unique_ptr<nvinfer1::ICudaEngine, nvinfer1::Destroy<nvinfer1::ICudaEngine>> engine{nullptr};
    // Declaring execution context.
    std::unique_ptr<nvinfer1::IExecutionContext, nvinfer1::Destroy<nvinfer1::IExecutionContext>> context{nullptr};
    std::vector<float> inputTensor;
    std::vector<float> outputTensor;
    // std::vector<float> referenceTensor;
    void* bindings[2]{0};
    // std::vector<std::string> inputFiles;
    cudawrapper::CudaStream stream;

    nvinfer1::ICudaEngine* getCudaEngine(std::string enginePath) {
        nvinfer1::ICudaEngine* engine{nullptr};

        std::string buffer = nvinfer1::readBuffer(enginePath);
        if (buffer.size())
        {
            // Try to deserialize the engine
            std::unique_ptr<nvinfer1::IRuntime, nvinfer1::Destroy<nvinfer1::IRuntime>> runtime{nvinfer1::createInferRuntime(gLogger)};
            engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
        }

        return engine;
    }

    static int getBindingInputIndex(nvinfer1::IExecutionContext* context)
    {
        return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
    }

    void launchInference(nvinfer1::IExecutionContext* context, cudaStream_t stream, std::vector<float> const& inputTensor, std::vector<float>& outputTensor, void** bindings, int batchSize)
    {
        int inputId = getBindingInputIndex(context);
        cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueueV2(bindings, stream, nullptr);
        cudaMemcpyAsync(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    //==========================AI=============================================

    void flatten(tf2::Quaternion &quat) {
        quat.setX(0);
        quat.setY(0);
        double length = sqrt(quat.w() * quat.w() + quat.z() * quat.z());
        quat.setW(quat.w() / length);
        quat.setZ(quat.z() / length);
    }

    void preprocess(cv::Mat &frame, const int bi) {
        //resize image, remove mean, divide by std
        nvinfer1::Dims dims_o{engine->getBindingDimensions(0)};
        resize(frame, frame, cv::Size(dims_o.d[2], dims_o.d[3]));

        frame.convertTo(imagePreproc, CV_32FC3, 1 / 255.0, 0);

        //copy image into tensor and copy it into GPU
        cv::split(imagePreproc, bgr);
        int channels = engine->getBindingDimensions(0).d[1];
        inputTensor.clear();
        for (int i = 0; i < channels; i++){
            inputTensor.insert(inputTensor.end(), (float*)bgr[i].data, (float*)bgr[i].dataend);
        }
    }

    void postprocess(){
        // float* out = (float*)malloc(netRT->output_dim.tot() * sizeof(dnnType));
        // checkCuda(cudaDeviceSynchronize());
        // checkCuda(cudaMemcpy(out, output_d, netRT->output_dim.tot() * sizeof(dnnType), cudaMemcpyDeviceToHost));
        // checkCuda(cudaDeviceSynchronize());

        outputTensor[0] = (outputTensor[0] * 2) - 1;
        outputTensor[1] = (outputTensor[1] * 2) - 1;
    }


    void infer(cv::Mat in) {
        preprocess(in, 0);

        launchInference(context.get(), stream, inputTensor, outputTensor, bindings, BATCH_SIZE);
        cudaStreamSynchronize(stream);

        postprocess();
    }

    void processImage(const cinematography_msgs::msg::BoundingBox::SharedPtr msg) {
        int res;
        get_parameter("resolution", res);

        // If nothing detected, passthru
        if (msg->width == 0) {
            cinematography_msgs::msg::VisionMeasurements vm;
            vm.header.frame_id = "world_ned";
            vm.header.stamp = clock.now();
            vm.centerx = msg->centerx;
            vm.centery = msg->centery;
            vm.width = msg->width;
            vm.height = msg->height;
            vm.fov = msg->fov;
            vm.drone_pose = msg->drone_pose;
            vm.hde = 0;
            hde_pub->publish(vm);
            return;
        }

        // Pad image to 192x192 square with black bars
        sensor_msgs::msg::Image img = msg->image;
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg->image, msg->image.encoding);
        cv::Mat new_img;


        // Combine actor's position and rotation (adjusted from relative to absolute)
        cinematography_msgs::msg::VisionMeasurements vm;
        vm.header.frame_id = "world_ned";
        vm.header.stamp = clock.now();
        vm.centerx = msg->centerx;
        vm.centery = msg->centery;
        vm.width = msg->width;
        vm.height = msg->height;
        vm.fov = msg->fov;
        vm.drone_pose = msg->drone_pose;

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

        // DEBUG
        bb_pub->publish(cv_ptr->toImageMsg());

        infer(cv_ptr->image);

        double hde0 = outputTensor[0];   // Output #1
        double hde1 = outputTensor[1];   // Output #2

        // Get the rotation of the actor relative to camera
        double angle = atan2(hde1, hde0);
    
        vm.hde = angle;
        hde_pub->publish(vm);
        return;       
    }


public:
    HeadingEstimation() : Node("heading_estimation") {
        declare_parameter<std::string>("tensorrt_engine", "hde_deer_airsim.rt");
        get_parameter("tensorrt_engine", trt_engine_filename);
        declare_parameter("resolution", 192);

        unit_vect.vector.x = unit_quat.quaternion.x = unit_quat.quaternion.w = 1;
        unit_vect.vector.y = unit_vect.vector.z = unit_quat.quaternion.y = unit_quat.quaternion.z = 0;

        engine.reset(getCudaEngine(trt_engine_filename));
        if (!engine) {
            RCLCPP_ERROR(this->get_logger(), "Unable to create HDE engine\n");
            return;
        }

        for (int i = 0; i < engine->getNbBindings(); ++i)
        {
            nvinfer1::Dims dims{engine->getBindingDimensions(i)};
            size_t size = std::accumulate(dims.d+1, dims.d + dims.nbDims, BATCH_SIZE, std::multiplies<size_t>());
            // Create CUDA buffer for Tensor.
            cudaMalloc(&bindings[i], BATCH_SIZE * size * sizeof(float));

            // Resize CPU buffers to fit Tensor.
            if (engine->bindingIsInput(i)){
                inputTensor.resize(size);
            }
            else
                outputTensor.resize(size);
        }

        // Create Execution Context.
        context.reset(engine->createExecutionContext());
        nvinfer1::Dims dims_i{engine->getBindingDimensions(0)};
        nvinfer1::Dims4 inputDims{BATCH_SIZE, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
        context->setBindingDimensions(0, inputDims);

        //netRT = new tk::dnn::NetworkRT(NULL, trt_engine_filename.c_str());
        // These 3 lines fix a bug in the NetworkRT constructor
        
        // Allocate memory for input buffers
        // checkCuda(cudaMallocHost(&input, sizeof(dnnType) * netRT->input_dim.tot() * nBatches));
        // checkCuda(cudaMalloc(&input_d, sizeof(dnnType) * netRT->input_dim.tot() * nBatches));

        hde_pub = this->create_publisher<cinematography_msgs::msg::VisionMeasurements>("vision_measurements", 50);
        bb_sub = this->create_subscription<cinematography_msgs::msg::BoundingBox>("bounding_box", 50, std::bind(&HeadingEstimation::processImage, this, _1));
        bb_pub = this->create_publisher<sensor_msgs::msg::Image>("bounding_box_out", 50);
    }
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HeadingEstimation>());
    rclcpp::shutdown();

    return 0;
}