#include <iostream>

#include "cuda/tsdf_handler.cuh"
#include "transformer.hpp"
#include "publisher.hpp"

typedef Eigen::Matrix<float, 3, 1> Vector3f;

rclcpp::Clock::SharedPtr clock_;

TSDFHandler * tsdf_handler;
Transformer * transformer;
Publisher * publisher;

//keep track of lidar position transformed from lidar to world frame
Vector3f lidar_position_transformed;

//keep track of average run time
double average, count = 0.0;

//arrays to hold voxel position and sdf + weight data to be published
Vector3f publish_voxels_pos[PUBLISH_VOXELS_MAX_SIZE];
Voxel publish_voxels_data[PUBLISH_VOXELS_MAX_SIZE];

//callback for point cloud subscriber
void callback(sensor_msgs::msg::PointCloud2::SharedPtr point_cloud_2)
{
    auto start = std::chrono::high_resolution_clock::now();

    //convert lidar position coordinates to same frame as point cloud
    try{
      transformer->getLidarPositionInPointCloudFrame(*point_cloud_2, lidar_position_transformed);
    } //if error converting lidar position don't incorporate the data into the TSDF
    catch(tf2::LookupException & e){
      return;
    }
    catch(tf2::ExtrapolationException & e){
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    transformer->convertPointCloud2ToPointCloudXYZ(point_cloud_2, point_cloud_xyz);
    printf("Point Cloud Size: %lu\n", point_cloud_xyz->size());

    //used to keep track of number of voxels to publish after integrating this lidar scan into the TSDF
    int publish_voxels_size = 0;

    tsdf_handler->processPointCloudAndUpdateVoxels(point_cloud_xyz, &lidar_position_transformed, publish_voxels_pos, &publish_voxels_size, publish_voxels_data);

    //publish voxels. The number of voxels to publish is passed in publish_voxels_size
    publisher->publish(publish_voxels_size);

    //the time to integrate this lidar scan into the TSDF
    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout << "Overall Duration: ";
    std::cout << duration.count() << std::endl;

    //the avg time taken to integrate every lidar scan thus far
    std::cout << "Average Duration: ";
    average += duration.count();
    count++;
    std::cout << average/count << std::endl; 
    std::cout << "---------------------------------------------------------------" << std::endl;
}

int main(int argc, char ** argv)
{

  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("tsdf_node");

  node->declare_parameter<float>("voxel_size", .5);
  node->declare_parameter<float>("truncation_distance", 4.0);
  node->declare_parameter<float>("max_weight", 10000.0);
  node->declare_parameter<float>("publish_distance_squared", 2500.00);
  node->declare_parameter<float>("garbage_collect_distance_squared", 250000.0);
  node->declare_parameter<std::string>("lidar_frame", "lidar");
  float voxel_size, max_weight, publish_distance_squared, truncation_distance, garbage_collect_distance_squared;
  std::string lidar_frame;
  node->get_parameter("voxel_size", voxel_size);
  node->get_parameter("truncation_distance", truncation_distance);
  node->get_parameter("max_weight", max_weight);
  node->get_parameter("publish_distance_squared", publish_distance_squared);
  node->get_parameter("garbage_collect_distance_squared", garbage_collect_distance_squared);
  node->get_parameter("lidar_frame", lidar_frame);

  //use parameters to intialize global variables in tsdf handler and on GPU
  initGlobalVars(voxel_size, truncation_distance, max_weight, publish_distance_squared, garbage_collect_distance_squared);

  clock_ = node->get_clock();

  auto lidar_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "lidar", 500, callback
  );

  auto tsdf_pub = node->create_publisher<tsdf_package_msgs::msg::Tsdf>("tsdf", 10);

  tsdf_handler = new TSDFHandler();
  //source frame set to lidar frame
  transformer = new Transformer(lidar_frame, clock_);
  publisher = new Publisher(tsdf_pub, truncation_distance, voxel_size, publish_voxels_pos, publish_voxels_data);

  rclcpp::spin(node);

  rclcpp::shutdown();

  delete tsdf_handler;
  delete transformer;
  delete publisher;
  
  return 0;
}