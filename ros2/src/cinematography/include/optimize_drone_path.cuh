#ifndef _OPTIMIZE_DRONE_PATH_CUH
#define _OPTIMIZE_DRONE_PATH_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Dense>
#include <stdio.h> 

struct MultiDOF{
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
    double ax;
    double ay;
    double az;
    double yaw;
    double duration;
    MultiDOF(double x, double y, double z, double vx, double vy, double vz, double ax, double ay, double az, double yaw, double duration){
        this->x = x;
        this->y = y;
        this->z = z;
        this->vx = vx;
        this->vy = vy;
        this->vz = vz;
        this->ax = ax;
        this->ay = ay;
        this->az = az;
        this->yaw = yaw;
        this->duration = duration;
    }
};

struct Voxel{
    float sdf;
    Eigen::Matrix<double, 3, 1> position;
    __device__ __host__
    Voxel():position(0,0,0), sdf(0){}
    __device__ __host__
    Voxel(float sdf, float x, float y, float z){
        this->sdf = sdf;
        position(0) = x;
        position(1) = y;
        position(2) = z;
    }

        __host__ __device__
    bool operator==(const Voxel &other) const{ //may need to include voxel size param
            // return (position(0) == other.position(0) && position(1)==other.position(1) && position(2)==other.position(2));

        float epsilon = .5 / 4; //should equal voxel size / 4 TODO can set to .1/4 and expect voxels won't be smaller than that

        double diff_x = position(0) - other.position(0);
        double diff_y = position(1) - other.position(1);
        double diff_z = position(2) - other.position(2);
        //have to use an epsilon value due to floating point precision errors
        if((fabs(diff_x) < epsilon) && (fabs(diff_y) < epsilon) && (fabs(diff_z) < epsilon))
            return true;

        return false;

    }
};

struct Key{
    double x;
    double y;
    double z;

    __host__
    Key():x(0),y(0),z(0){}

    __host__
    Key(double x, double y, double z){
        this->x = x;
        this->y = y;
        this->z = z;                                                                       
    }

    __host__
    bool operator==(const Key &other) const{
        return (x == other.x && y==other.y && z==other.z);
    }
};

namespace std {
    template<>
    struct hash<Key>{
        __host__
        std::size_t operator()(const Key& k) const
        {
            using std::size_t;
            using std::hash;
            return(hash<double>()(k.x) ^ hash<double>()(k.y) ^ hash<double>()(k.z));
        }
    };
}

Eigen::Matrix<double, Eigen::Dynamic, 3> obstacle_avoidance_gradient_cuda(std::vector<MultiDOF>  & drone_traj, double & truncation_distance, double & voxel_size);

Eigen::Matrix<double, Eigen::Dynamic, 3> occlusion_avoidance_gradient_cuda(std::vector<MultiDOF>  & drone_traj, 
std::vector<MultiDOF> & actor_traj, double & truncation_distance, double & voxel_size);

void allocate_set();
void init_set_cuda(std::vector<Voxel> & voxels);
#endif