#ifndef _OPTIMIZE_DRONE_PATH_CUH
#define _OPTIMIZE_DRONE_PATH_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Dense>
#include <math.h>
#include <stdio.h> 
#include <unordered_map>

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

void optimize_trajectory_cuda(std::vector<MultiDOF> & drone_traj, std::vector<MultiDOF> & actor_traj, std::vector<Voxel> & voxels);

#endif