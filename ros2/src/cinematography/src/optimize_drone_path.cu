#include "optimize_drone_path.cuh"
#include <stdgpu/unordered_map.cuh>
#include <stdgpu/unordered_set.cuh>
#include <stdgpu/memory.h>

//error function for cpu called after kernel calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace stdgpu {
    template<>
    struct hash<Voxel>{
        inline STDGPU_HOST_DEVICE std::size_t
        operator()(const Voxel& v) const
        {
            Eigen::Matrix<double, 3, 1> p = v.position;
            return ((int)(p(0)*73856093) ^ (int)(p(1)*19349669) ^ (int)(p(2)*83492791));
            // using stdgpu::hash;
            // return(hash<double>()(k.x) ^ hash<double>()(k.y) ^ hash<double>()(k.z));
        }
    };
}

stdgpu::unordered_set<Voxel> set;

__global__ 
void init_set(stdgpu::unordered_set<Voxel> set, Voxel * voxels, int * voxels_size){
    int thread_index = (blockIdx.x*512 + threadIdx.x);
    if(thread_index >= *voxels_size){
        return;
    }
    Voxel v = voxels[thread_index];
    set.insert(v);
}

__device__
double floor_fun_cuda(const double & x, const double & scale){
    return floor(x*scale) / scale;
}

/*
* Given world point return center of volume given by volume_size
*/
__device__
Eigen::Matrix<double, 3, 1> get_volume_center_from_point_cuda(Eigen::Matrix<double, 3, 1> point, double volume_size){
    double scale = 1/volume_size;
    double half_volume_size = volume_size / 2;
    Eigen::Matrix<double, 3, 1> volume_center;
    volume_center(0) = floor_fun_cuda(point(0), scale) + half_volume_size;
    volume_center(1) = floor_fun_cuda(point(1), scale) + half_volume_size;
    volume_center(2) = floor_fun_cuda(point(2), scale) + half_volume_size;
    return volume_center;
}

/*
* Check if two Eigen::Matrix<double, 3, 1> are equal
*/
__device__
bool check_floating_point_vectors_equal_cuda(Eigen::Matrix<double, 3, 1> A, Eigen::Matrix<double, 3, 1> B, double epsilon){
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
__device__
void get_voxels(const MultiDOF point_start, const MultiDOF point_end, Eigen::Matrix<double, 3, 1> * voxels, const double & volume_size, int & size){
    const double half_volume_size = volume_size / 2;
    const double epsilon = volume_size / 4;
    const double volume_size_plus_epsilon = volume_size + epsilon;
    const double volume_size_minus_epsilon = volume_size - epsilon;
    Eigen::Matrix<double, 3, 1> start(point_start.x, point_start.y, point_start.z);
    Eigen::Matrix<double, 3, 1> end(point_end.x, point_end.y, point_end.z);
    //   equation of line = u+tv
    Eigen::Matrix<double, 3, 1> u(point_start.x, point_start.y, point_start.z);
    Eigen::Matrix<double, 3, 1> v(point_end.x - point_start.x, point_end.y - point_start.y, point_end.z - point_start.z);
    double stepX = v(0) > 0 ? volume_size : -1 * volume_size;
    double stepY = v(1) > 0 ? volume_size : -1 * volume_size;
    double stepZ = v(2) > 0 ? volume_size : -1 * volume_size;
    Eigen::Matrix<double, 3, 1> start_voxel_center = get_volume_center_from_point_cuda(start, volume_size);
    double tMaxX = fabs(v(0) < 0 ? (start_voxel_center(0) - half_volume_size - u(0)) / v(0) : (start_voxel_center(0) + half_volume_size - u(0)) / v(0));
    double tMaxY = fabs(v(1) < 0 ? (start_voxel_center(1) - half_volume_size - u(1)) / v(1) : (start_voxel_center(1) + half_volume_size - u(1)) / v(1));
    double tMaxZ = fabs(v(2) < 0 ? (start_voxel_center(2) - half_volume_size - u(2)) / v(2) : (start_voxel_center(2) + half_volume_size - u(2)) / v(2));
    double tDeltaX = fabs(volume_size / v(0));
    double tDeltaY = fabs(volume_size / v(1));
    double tDeltaZ = fabs(volume_size / v(2));
    Eigen::Matrix<double, 3, 1> current_vol(start(0), start(1), start(2));
    Eigen::Matrix<double, 3, 1> current_vol_center = get_volume_center_from_point_cuda(current_vol, volume_size);
    Eigen::Matrix<double, 3, 1> end_voxel_center = get_volume_center_from_point_cuda(end, volume_size);

    while(!check_floating_point_vectors_equal_cuda(current_vol_center, end_voxel_center, epsilon)){
        //add traversed voxel key to list of voxel keys
        Eigen::Matrix<double, 3, 1> voxel = current_vol_center;
        voxels[size] = voxel;
        size++;
        
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
        current_vol_center = get_volume_center_from_point_cuda(current_vol, volume_size);
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
    Eigen::Matrix<double, 3, 1> voxel = current_vol_center;
    voxels[size] = voxel;
    size++;
}

/*
* Cost of sdf value
*/
__device__
double get_cost(const double & sdf, const double & truncation_distance){
    if(fabs(sdf) >= truncation_distance){
        return 0;
    }
    else if(sdf > 0){
        return pow((sdf - truncation_distance), 2) / (2* truncation_distance);
    }else{
        return sdf * -1 + .5 * truncation_distance;
    }

}

__device__
inline double get_voxel_cost(const Eigen::Matrix<double, 3, 1> & voxel_pos, const double & volume_size, stdgpu::unordered_set<Voxel> set, double * truncation_distance){

    Voxel voxel(0, voxel_pos(0), voxel_pos(1), voxel_pos(2));
    const Voxel * result = set.find(voxel);
    if(result != set.end()){
        return get_cost(result->sdf, * truncation_distance);
    }
    return 0; //voxel does not exist so it is in free space(or inside an object) and return 0 cost
}

/*
* Compute cost gradient for a voxel. Check cost values of voxel and voxels around
*/
__device__
Eigen::Matrix<double, 3, 1> get_voxel_cost_gradient(const Eigen::Matrix<double, 3, 1> & voxel_pos, const double & volume_size, stdgpu::unordered_set<Voxel> set, double * truncation_distance){
    double cost = get_voxel_cost(voxel_pos, volume_size, set, truncation_distance);
    Eigen::Matrix<double, 3, 1> gradient_val;

    Eigen::Matrix<double, 3, 1> xNext(voxel_pos(0) + volume_size, voxel_pos(1), voxel_pos(2));
    Eigen::Matrix<double, 3, 1> xPrev(voxel_pos(0) - volume_size, voxel_pos(1), voxel_pos(2));
    double xDiffNext = get_voxel_cost(xNext, volume_size, set, truncation_distance) - cost;
    double xDiffPrev = cost - get_voxel_cost(xPrev, volume_size, set, truncation_distance);
    gradient_val(0) = (xDiffNext + xDiffPrev) / 2;

    Eigen::Matrix<double, 3, 1> yNext(voxel_pos(0), voxel_pos(1) + volume_size, voxel_pos(2));
    Eigen::Matrix<double, 3, 1> yPrev(voxel_pos(0), voxel_pos(1) - volume_size, voxel_pos(2));
    double yDiffNext = get_voxel_cost(yNext, volume_size, set, truncation_distance) - cost;
    double yDiffPrev = cost - get_voxel_cost(yPrev, volume_size, set, truncation_distance);
    gradient_val(1) = (yDiffNext + yDiffPrev) / 2;

    Eigen::Matrix<double, 3, 1> zNext(voxel_pos(0), voxel_pos(1), voxel_pos(2) + volume_size);
    Eigen::Matrix<double, 3, 1> zPrev(voxel_pos(0), voxel_pos(1), voxel_pos(2) - volume_size);
    double zDiffNext = get_voxel_cost(zNext, volume_size, set, truncation_distance) - cost;
    double zDiffPrev = cost - get_voxel_cost(zPrev, volume_size, set, truncation_distance);
    gradient_val(2) = (zDiffNext + zDiffPrev) / 2;

    return gradient_val;
}

__global__
void process_obstacle_avoidance_voxels(Eigen::Matrix<double, 3, 1> * voxels, int * size_p, stdgpu::unordered_set<Voxel> set, double * velocity_mag, 
    Eigen::Matrix<double, 3, 3> * identity_minus_p_hat_multiplied, double * obs_grad, int * point_index, double * truncation_distance, double * voxel_size){
    int thread_index = (blockIdx.x*256 + threadIdx.x);
    if(thread_index >=*size_p){
        return;
    }
    Eigen::Matrix<double, 3, 1> cost_function_gradient = get_voxel_cost_gradient(voxels[thread_index], * voxel_size, set, truncation_distance); //TODO: volume_size
    Eigen::Matrix<double, 3, 1> gradient_multiplied_result = (*identity_minus_p_hat_multiplied) * cost_function_gradient;       
    Eigen::Matrix<double, 3, 1> grad_j_obs = gradient_multiplied_result * (*velocity_mag);
    
    if(isnan(grad_j_obs(0)) || isnan(grad_j_obs(1)) || isnan(grad_j_obs(2))){
        grad_j_obs(0) = 0;
        grad_j_obs(1) = 0;
        grad_j_obs(2) = 0;
    }

    int obs_grad_index = (* point_index) * 3;

    atomicAdd(&(obs_grad[obs_grad_index]), grad_j_obs(0));
    atomicAdd(&(obs_grad[obs_grad_index + 1]), grad_j_obs(1));
    atomicAdd(&(obs_grad[obs_grad_index + 2]), grad_j_obs(2));

}

__global__
void obstacle_avoidance_gradient(MultiDOF * drone_traj, double * obs_grad, int * n, stdgpu::unordered_set<Voxel> set, double * truncation_distance, double * voxel_size){
    int thread_index = (blockIdx.x*128 + threadIdx.x);
    if(thread_index >= *n-1){
        return;
    }

    MultiDOF point_start = drone_traj[thread_index];
    MultiDOF point_end = drone_traj[thread_index+1];

    Eigen::Matrix<double, 3, 1> * voxels = new Eigen::Matrix<double, 3, 1>[500]; //figure out how to bound this
    int size = 0;
    get_voxels(point_start, point_end, voxels, * voxel_size, size);

    double velocity_mag = sqrt(pow(point_end.vx, 2) + pow(point_end.vy ,2) + pow(point_end.vz , 2));
    double * velocity_mag_p = new double(velocity_mag);
    Eigen::Matrix<double, 3, 1> p_hat(point_end.vx/velocity_mag, point_end.vy/velocity_mag, point_end.vz/velocity_mag);
    if(isnan(p_hat(0)) || isnan(p_hat(1)) || isnan(p_hat(2))){
      p_hat(0) = 0;
      p_hat(1) = 0;
      p_hat(2) = 0;
    }

    Eigen::Matrix<double, 3, 3> p_hat_multiplied = p_hat * p_hat.transpose();

    Eigen::Matrix<double, 3, 3> * identity_minus_p_hat_multiplied_p = new Eigen::Matrix3d();
    * identity_minus_p_hat_multiplied_p = Eigen::Matrix3d::Identity(3,3) - p_hat_multiplied;

    int threads_per_block = 256;
    int num_blocks = size / threads_per_block + 1;
    int * size_p = new int(size);
    int * thread_index_p = new int(thread_index);
    process_obstacle_avoidance_voxels<<<num_blocks, threads_per_block>>>(voxels, size_p, set, velocity_mag_p, 
        identity_minus_p_hat_multiplied_p, obs_grad, thread_index_p, truncation_distance, voxel_size);
    cudaDeviceSynchronize();

    int obs_grad_index = thread_index * 3;

    obs_grad[obs_grad_index]/=size;
    obs_grad[obs_grad_index+1]/=size;
    obs_grad[obs_grad_index+2]/=size;

    free(voxels);
    free(velocity_mag_p);
    free(size_p);
    free(thread_index_p);
    free(identity_minus_p_hat_multiplied_p);
}

__global__
void process_occlusion_avoidance_voxels(Eigen::Matrix<double, 3, 1> * voxels, int * size_p, stdgpu::unordered_set<Voxel> set, 
     double * occ_grad, int * point_index, MultiDOF * point_start_p, MultiDOF * point_end_p, double * truncation_distance, double * voxel_size){
    int thread_index = (blockIdx.x*256 + threadIdx.x);
    if(thread_index >=*size_p){
        return;
    }

    MultiDOF point_start = * point_start_p;
    MultiDOF point_end = * point_end_p;

    Eigen::Matrix<double, 3, 1> actor_point_velocity(point_end.vx, point_end.vy, point_end.vz);

    Eigen::Matrix<double, 3, 1> drone_point_velocity(point_start.vx, point_start.vy, point_start.vz);
    double drone_point_velocity_mag = drone_point_velocity.norm();
    Eigen::Matrix<double, 3, 1> normalized_drone_point_velocity = drone_point_velocity/drone_point_velocity_mag; 
    if(isnan(normalized_drone_point_velocity(0)) || isnan(normalized_drone_point_velocity(1)) || isnan(normalized_drone_point_velocity(2))){
        normalized_drone_point_velocity(0) = 0;
        normalized_drone_point_velocity(1) = 0;
        normalized_drone_point_velocity(2) = 0;
    }
    
    Eigen::Matrix<double, 1, 3> normalized_drone_point_velocity_transpose = normalized_drone_point_velocity.transpose();

    Eigen::Matrix<double, 3, 1> L(point_end.x - point_start.x, point_end.y - point_start.y, point_end.z - point_start.z);
    double L_mag = L.norm();
    Eigen::Matrix<double, 3, 1> normalized_L = L/L_mag;
    Eigen::Matrix<double, 1, 3> normalized_L_transpose = normalized_L.transpose();
    Eigen::Matrix<double, 3, 1> L_velocity = actor_point_velocity - drone_point_velocity;

    //used for determining the value of τ
    double increment = 1.0/(*size_p-1); 

    Eigen::Matrix<double, 3, 1> cost_function_gradient = get_voxel_cost_gradient(voxels[thread_index], * voxel_size, set, truncation_distance);
    // Eigen::Matrix<double, 3, 1> cost_function_gradient(1,1,1);
    Eigen::Matrix<double, 3, 1> inner_first_term = actor_point_velocity/drone_point_velocity_mag - normalized_drone_point_velocity;
    double temp1 = inner_first_term(0);
    double temp2 = inner_first_term(1);
    double temp3 = inner_first_term(2);
    inner_first_term*=thread_index*increment;
    inner_first_term +=normalized_drone_point_velocity;
    Eigen::Matrix<double, 3, 3> inner_first_term_matrix = inner_first_term * normalized_drone_point_velocity_transpose;
    inner_first_term_matrix = Eigen::Matrix3d::Identity(3,3) - inner_first_term_matrix;
    Eigen::Matrix<double, 1, 3> first_term = cost_function_gradient.transpose() * L_mag * drone_point_velocity_mag * inner_first_term_matrix;

    Eigen::Matrix<double, 1, 3> inner_second_term = normalized_L_transpose + normalized_L_transpose * L_velocity * normalized_drone_point_velocity_transpose;
    Eigen::Matrix<double, 1, 3> second_term = get_voxel_cost(voxels[thread_index], * voxel_size, set, truncation_distance) * drone_point_velocity_mag * inner_second_term;
    // Eigen::Matrix<double, 1, 3> second_term(1,1,1);
    Eigen::Matrix<double, 1, 3> grad_j_occ = first_term - second_term;

    if(isnan(grad_j_occ(0)) || isnan(grad_j_occ(1)) || isnan(grad_j_occ(2))){
        grad_j_occ(0) = 0;
        grad_j_occ(1) = 0;
        grad_j_occ(2) = 0;
    }

    int occ_grad_index = (* point_index) * 3;

    atomicAdd(&(occ_grad[occ_grad_index]), grad_j_occ(0));
    atomicAdd(&(occ_grad[occ_grad_index + 1]), grad_j_occ(1));
    atomicAdd(&(occ_grad[occ_grad_index + 2]), grad_j_occ(2));

}

__global__ 
void occlusion_avoidance_gradient(MultiDOF * drone_traj, MultiDOF * actor_traj, double * occ_grad, int * n, stdgpu::unordered_set<Voxel> set, 
double * truncation_distance, double * voxel_size){
    int thread_index = (blockIdx.x*128 + threadIdx.x);
    if(thread_index >= *n-1){
        return;
    }
    
    MultiDOF * point_start_p = new MultiDOF(drone_traj[thread_index+1]); 
    // * point_start_p = drone_traj[thread_index+1];

    MultiDOF * point_end_p = new MultiDOF(actor_traj[thread_index+1]); 
    // * point_end_p = actor_traj[thread_index+1];

    Eigen::Matrix<double, 3, 1> * voxels = new Eigen::Matrix<double, 3, 1>[500]; //figure out how to bound this
    int size = 0;
    get_voxels(* point_start_p, * point_end_p, voxels, * voxel_size, size);

    int threads_per_block = 256;
    int num_blocks = size / threads_per_block + 1;
    int * size_p = new int(size);
    int * thread_index_p = new int(thread_index);
    process_occlusion_avoidance_voxels<<<num_blocks, threads_per_block>>>(voxels, size_p, set, occ_grad, 
        thread_index_p, point_start_p, point_end_p, truncation_distance, voxel_size);
    cudaDeviceSynchronize();

    int occ_grad_index = thread_index * 3;

    occ_grad[occ_grad_index]/=size;
    occ_grad[occ_grad_index+1]/=size;
    occ_grad[occ_grad_index+2]/=size;

    free(voxels);
    free(size_p);
    free(thread_index_p);
    free(point_start_p);
    free(point_end_p);
}

void allocate_set(){
    set = stdgpu::unordered_set<Voxel>::createDeviceObject(5000000);
}

void init_set_cuda(std::vector<Voxel> & voxels){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    set.clear();

    int * voxels_size_h = new int(voxels.size());

    if(*voxels_size_h == 0){
        return;
    }

    // printf("voxels size: %d\n", * voxels_size_h);
    int * voxels_size_d;
    cudaMalloc(&voxels_size_d, sizeof(*voxels_size_h));
    cudaMemcpy(voxels_size_d, voxels_size_h, sizeof(*voxels_size_h), cudaMemcpyHostToDevice);

    Voxel * voxels_h = &voxels[0];
    Voxel * voxels_d;
    cudaMalloc(&voxels_d, sizeof(*voxels_h) * (*voxels_size_h));
    cudaMemcpy(voxels_d, voxels_h, sizeof(*voxels_h) * (*voxels_size_h), cudaMemcpyHostToDevice);

    int num_threads = 512;
    int num_blocks = *voxels_size_h / num_threads + 1;
    // printf("max size: %d\n", set.max_size());
    init_set<<<num_blocks,num_threads>>>(set, voxels_d, voxels_size_d);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // printf("valid? : %d\n", set.valid());
    printf("set size: %d\n", set.size());
    printf("set load factor: %f\n", set.load_factor());
    // printf("set max load factor: %f\n", set.max_load_factor());
    printf("num buckets: %d\n", set.bucket_count());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("set generation: %f\n", milliseconds);

}

Eigen::Matrix<double, Eigen::Dynamic, 3> obstacle_avoidance_gradient_cuda(std::vector<MultiDOF>  & drone_traj, double & truncation_distance, double & voxel_size){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int * n_h = new int(drone_traj.size());
    Eigen::Matrix<double, Eigen::Dynamic, 3> obs_grad((*n_h)-1,3);

    if(set.size() == 0){
        for(int i=0;i<(*n_h)-1; ++i){
            obs_grad(i,0) = 0;
            obs_grad(i,1) = 0;
            obs_grad(i,2) = 0;
        }
        return obs_grad;
    }
    
    int * n_d;
    cudaMalloc(&n_d, sizeof(*n_h));
    cudaMemcpy(n_d, n_h, sizeof(*n_h), cudaMemcpyHostToDevice);

    MultiDOF * drone_traj_h = &drone_traj[0];
    MultiDOF * drone_traj_d;

    cudaMalloc(&drone_traj_d, sizeof(*drone_traj_h)*(*n_h));
    cudaMemcpy(drone_traj_d, drone_traj_h, sizeof(*drone_traj_h)*(*n_h), cudaMemcpyHostToDevice);

    double * truncation_distance_h = new double(truncation_distance);
    double * truncation_distance_d;
    cudaMalloc(&truncation_distance_d, sizeof(*truncation_distance_h));
    cudaMemcpy(truncation_distance_d, truncation_distance_h, sizeof(*truncation_distance_h), cudaMemcpyHostToDevice);

    double * voxel_size_h = new double(voxel_size);
    double * voxel_size_d;
    cudaMalloc(&voxel_size_d, sizeof(*voxel_size_h));
    cudaMemcpy(voxel_size_d, voxel_size_h, sizeof(*voxel_size_h), cudaMemcpyHostToDevice);

    int obs_grad_size = ((*n_h)-1) * 3;
    double obs_grad_h[obs_grad_size];
    double * obs_grad_d;
    cudaMalloc(&obs_grad_d, sizeof(*obs_grad_h) * obs_grad_size);
    cudaMemset(obs_grad_d, 0, sizeof(*obs_grad_h) * obs_grad_size);

    int num_threads = 128;
    int num_blocks = *n_h / num_threads + 1;

    obstacle_avoidance_gradient<<<num_blocks, num_threads>>>(drone_traj_d, obs_grad_d, n_d, set, truncation_distance_d, voxel_size_d);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    cudaMemcpy(obs_grad_h, obs_grad_d, sizeof(*obs_grad_h) * obs_grad_size, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());

    for(int i=0;i<(*n_h)-1; ++i){
        int obs_grad_h_index = 3*i;
        obs_grad(i,0) = obs_grad_h[obs_grad_h_index];
        obs_grad(i,1) = obs_grad_h[obs_grad_h_index + 1];
        obs_grad(i,2) = obs_grad_h[obs_grad_h_index + 2];
    }

    free(n_h);
    free(truncation_distance_h);
    free(voxel_size_h);
    cudaFree(n_d);
    cudaFree(drone_traj_d);
    cudaFree(obs_grad_d);
    cudaFree(truncation_distance_d);
    cudaFree(voxel_size_d);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("obstacle avoidance gradient duration: %f\n", milliseconds);

    return obs_grad;
}

Eigen::Matrix<double, Eigen::Dynamic, 3> occlusion_avoidance_gradient_cuda(std::vector<MultiDOF>  & drone_traj, 
    std::vector<MultiDOF> & actor_traj, double & truncation_distance, double & voxel_size){
    
    int * n_h = new int(drone_traj.size());
    Eigen::Matrix<double, Eigen::Dynamic, 3> occ_grad((*n_h)-1,3);
    
    if(set.size() == 0){
        for(int i=0;i<(*n_h)-1; ++i){
            occ_grad(i,0) = 0;
            occ_grad(i,1) = 0;
            occ_grad(i,2) = 0;
        }
        return occ_grad;
    }

    int * n_d;
    cudaMalloc(&n_d, sizeof(*n_h));
    cudaMemcpy(n_d, n_h, sizeof(*n_h), cudaMemcpyHostToDevice);

    MultiDOF * drone_traj_h = &drone_traj[0];
    MultiDOF * actor_traj_h = &actor_traj[0];
    MultiDOF * drone_traj_d;
    MultiDOF * actor_traj_d;

    cudaMalloc(&drone_traj_d, sizeof(*drone_traj_h)*(*n_h));
    cudaMemcpy(drone_traj_d, drone_traj_h, sizeof(*drone_traj_h)*(*n_h), cudaMemcpyHostToDevice);
    cudaMalloc(&actor_traj_d, sizeof(*actor_traj_h)*(*n_h));
    cudaMemcpy(actor_traj_d, actor_traj_h, sizeof(*actor_traj_h)*(*n_h), cudaMemcpyHostToDevice);

    double * truncation_distance_h = new double(truncation_distance);
    double * truncation_distance_d;
    cudaMalloc(&truncation_distance_d, sizeof(*truncation_distance_h));
    cudaMemcpy(truncation_distance_d, truncation_distance_h, sizeof(*truncation_distance_h), cudaMemcpyHostToDevice);

    double * voxel_size_h = new double(voxel_size);
    double * voxel_size_d;
    cudaMalloc(&voxel_size_d, sizeof(*voxel_size_h));
    cudaMemcpy(voxel_size_d, voxel_size_h, sizeof(*voxel_size_h), cudaMemcpyHostToDevice);

    int occ_grad_size = ((*n_h)-1) * 3;
    double occ_grad_h[occ_grad_size];
    double * occ_grad_d;
    cudaMalloc(&occ_grad_d, sizeof(*occ_grad_h) * occ_grad_size);
    cudaMemset(occ_grad_d, 0, sizeof(*occ_grad_h) * occ_grad_size);

    int num_threads = 128;
    int num_blocks = *n_h / num_threads + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    occlusion_avoidance_gradient<<<num_blocks, num_threads>>>(drone_traj_d, actor_traj_d, occ_grad_d, n_d, set, truncation_distance_d, voxel_size_d);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("occlusion avoidance gradient duration: %f\n", milliseconds);

    cudaMemcpy(occ_grad_h, occ_grad_d, sizeof(*occ_grad_h) * occ_grad_size, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());

    for(int i=0;i<(*n_h)-1; ++i){
        int occ_grad_h_index = 3*i;
        occ_grad(i,0) = occ_grad_h[occ_grad_h_index];
        occ_grad(i,1) = occ_grad_h[occ_grad_h_index + 1];
        occ_grad(i,2) = occ_grad_h[occ_grad_h_index + 2];
    }

    free(n_h);
    free(truncation_distance_h);
    free(voxel_size_h);
    cudaFree(n_d);
    cudaFree(drone_traj_d);
    cudaFree(actor_traj_d);
    cudaFree(occ_grad_d);
    cudaFree(truncation_distance_d);
    cudaFree(voxel_size_d);

    return occ_grad;
}