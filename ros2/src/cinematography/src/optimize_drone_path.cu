#include "optimize_drone_path.cuh"

__device__
double floor_fun(const double & x, const double & scale){
    return floor(x*scale) / scale;
}

/*
* Given world point return center of volume given by volume_size
*/
__device__
Eigen::Matrix<double, 3, 1> get_volume_center_from_point(Eigen::Matrix<double, 3, 1> point, double volume_size){
    double scale = 1/volume_size;
    double half_volume_size = volume_size / 2;
    Eigen::Matrix<double, 3, 1> volume_center;
    volume_center(0) = floor_fun(point(0), scale) + half_volume_size;
    volume_center(1) = floor_fun(point(1), scale) + half_volume_size;
    volume_center(2) = floor_fun(point(2), scale) + half_volume_size;
    return volume_center;
}

/*
* Check if two Eigen::Matrix<double, 3, 1> are equal
*/
__device__
bool check_floating_point_vectors_equal(Eigen::Matrix<double, 3, 1> A, Eigen::Matrix<double, 3, 1> B, double epsilon){
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
    Eigen::Matrix<double, 3, 1> start_voxel_center = get_volume_center_from_point(start, volume_size);
    double tMaxX = fabs(v(0) < 0 ? (start_voxel_center(0) - half_volume_size - u(0)) / v(0) : (start_voxel_center(0) + half_volume_size - u(0)) / v(0));
    double tMaxY = fabs(v(1) < 0 ? (start_voxel_center(1) - half_volume_size - u(1)) / v(1) : (start_voxel_center(1) + half_volume_size - u(1)) / v(1));
    double tMaxZ = fabs(v(2) < 0 ? (start_voxel_center(2) - half_volume_size - u(2)) / v(2) : (start_voxel_center(2) + half_volume_size - u(2)) / v(2));
    double tDeltaX = fabs(volume_size / v(0));
    double tDeltaY = fabs(volume_size / v(1));
    double tDeltaZ = fabs(volume_size / v(2));
    Eigen::Matrix<double, 3, 1> current_vol(start(0), start(1), start(2));
    Eigen::Matrix<double, 3, 1> current_vol_center = get_volume_center_from_point(current_vol, volume_size);
    Eigen::Matrix<double, 3, 1> end_voxel_center = get_volume_center_from_point(end, volume_size);

    while(!check_floating_point_vectors_equal(current_vol_center, end_voxel_center, epsilon)){
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
        current_vol_center = get_volume_center_from_point(current_vol, volume_size);
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
inline double get_cost(const double & sdf){
    double truncation_distance = 4;
    if(fabs(sdf) >= truncation_distance){
        return 0;
    }
    else if(sdf > 0){
        return pow((sdf - truncation_distance), 2) / (2* truncation_distance);
    }else{
        return sdf * -1 + .5 * truncation_distance;
    }

}

/*
* Compute cost gradient for a voxel. Check cost values of voxel and voxels around
*/
__device__
Eigen::Matrix<double, 3, 1> get_cost_gradient(const Eigen::Matrix<double, 3, 1> & voxel, const double & volume_size, Voxel * voxels, int * voxels_size){

    //todo: implement an unordered map like data structure for gpu

    Eigen::Matrix<double, 3, 1> xNext(voxel(0) + volume_size, voxel(1), voxel(2));
    Eigen::Matrix<double, 3, 1> xPrev(voxel(0) - volume_size, voxel(1), voxel(2));

    Eigen::Matrix<double, 3, 1> yNext(voxel(0), voxel(1) + volume_size, voxel(2));
    Eigen::Matrix<double, 3, 1> yPrev(voxel(0), voxel(1) - volume_size, voxel(2));

    Eigen::Matrix<double, 3, 1> zNext(voxel(0), voxel(1), voxel(2) + volume_size);
    Eigen::Matrix<double, 3, 1> zPrev(voxel(0), voxel(1), voxel(2) - volume_size);

    double voxel_cost = 0;
    double x_next_cost = 0;
    double x_prev_cost = 0;
    double y_next_cost = 0;
    double y_prev_cost = 0;
    double z_next_cost = 0;
    double z_prev_cost = 0;

    bool voxel_found = false;
    bool x_next_found = false;
    bool x_prev_found = false;
    bool y_next_found = false;
    bool y_prev_found = false;
    bool z_next_found = false;
    bool z_prev_found = false;

    double epsilon = volume_size / 4;

    for(size_t i = 0;i < *voxels_size; ++i){
        Voxel v = voxels[i];
        Eigen::Matrix<double, 3, 1> pos = v.position;
        float sdf = v.sdf;
        if(!voxel_found && check_floating_point_vectors_equal(pos, voxel, epsilon)){
            voxel_cost = get_cost(sdf);
            voxel_found = true;
        }
        else if(!x_next_found && check_floating_point_vectors_equal(pos, xNext, epsilon)){
            x_next_cost = get_cost(sdf);
            x_next_found = true;
        }
        else if(!x_prev_found && check_floating_point_vectors_equal(pos, xPrev, epsilon)){
            x_prev_cost = get_cost(sdf);
            x_prev_found = true;
        }
        else if(!y_next_found && check_floating_point_vectors_equal(pos, yNext, epsilon)){
            y_next_cost = get_cost(sdf);
            y_next_found = true;
        }
        else if(!y_prev_found && check_floating_point_vectors_equal(pos, yPrev, epsilon)){
            y_prev_cost = get_cost(sdf);
            y_prev_found = true;
        }
        else if(!z_next_found && check_floating_point_vectors_equal(pos, zNext, epsilon)){
            z_next_cost = get_cost(sdf);
            z_next_found = true;
        }
        else if(!z_prev_found && check_floating_point_vectors_equal(pos, zPrev, epsilon)){
            z_prev_cost = get_cost(sdf);
            z_prev_found = true;
        }
    }

    Eigen::Matrix<double, 3, 1> gradient_val;
    gradient_val(0) = ((x_next_cost - voxel_cost) + (voxel_cost - x_prev_cost)) / 2;
    gradient_val(1) = ((y_next_cost - voxel_cost) + (voxel_cost - y_prev_cost)) / 2;
    gradient_val(2) = ((y_next_cost - voxel_cost) + (voxel_cost - y_prev_cost)) / 2;
    return gradient_val;
}

__global__
void process_obstacle_avoidance_voxels(Eigen::Matrix<double, 3, 1> * voxels, int * size_p, Voxel * voxels_data, int * voxels_size, double * velocity_mag, 
    Eigen::Matrix<double, 3, 3> * identity_minus_p_hat_multiplied, Eigen::Matrix<double, Eigen::Dynamic, 3> * obs_grad, int * point_index){
    int thread_index = (blockIdx.x*256 + threadIdx.x);
    if(thread_index >=*size_p){
        return;
    }
    Eigen::Matrix<double, 3, 1> cost_function_gradient = get_cost_gradient(voxels[thread_index], 0.5, voxels_data, voxels_size); //TODO: volume_size
    Eigen::Matrix<double, 3, 1> gradient_multiplied_result = (*identity_minus_p_hat_multiplied) * cost_function_gradient;       
    Eigen::Matrix<double, 3, 1> grad_j_obs = gradient_multiplied_result * (*velocity_mag);
    atomicAdd(&((*obs_grad)(* point_index, 0)), grad_j_obs(0));
    atomicAdd(&((*obs_grad)(* point_index, 1)), grad_j_obs(1));
    atomicAdd(&((*obs_grad)(* point_index, 2)), grad_j_obs(2));

}

__global__
void obstacle_avoidance_gradient(MultiDOF * drone_traj, Eigen::Matrix<double, Eigen::Dynamic, 3> * obs_grad, int * n, Voxel * voxels_data, int * voxels_size){
    int thread_index = (blockIdx.x*128 + threadIdx.x);
    if(thread_index >= *n-1){
        return;
    }

    MultiDOF point_start = drone_traj[thread_index];
    MultiDOF point_end = drone_traj[thread_index+1];

    Eigen::Matrix<double, 3, 1> * voxels = new Eigen::Matrix<double, 3, 1>[500]; //figure out how to bound this
    int size = 0;
    get_voxels(point_start, point_end, voxels, 0.5, size);

    double velocity_mag = sqrt(pow(point_end.vx, 2) + pow(point_end.vy ,2) + pow(point_end.vz , 2));
    double * velocity_mag_p = new double(velocity_mag);
    Eigen::Matrix<double, 3, 1> p_hat(point_end.vx/velocity_mag, point_end.vy/velocity_mag, point_end.vz/velocity_mag);
    if(isnan(p_hat(0)), isnan(p_hat(1)), isnan(p_hat(2))){
      p_hat(0) = 0;
      p_hat(1) = 0;
      p_hat(2) = 0;
    }

    Eigen::Matrix<double, 3, 3> p_hat_multiplied = p_hat * p_hat.transpose();

    Eigen::Matrix<double, 3, 3> * identity_minus_p_hat_multiplied_p;
    * identity_minus_p_hat_multiplied_p = Eigen::Matrix3d::Identity(3,3) - p_hat_multiplied;

    int threads_per_block = 256;
    int num_blocks = size / threads_per_block + 1;
    int * size_p = new int(size);
    int * thread_index_p = new int(thread_index);
    process_obstacle_avoidance_voxels<<<num_blocks, threads_per_block>>>(voxels, size_p, voxels_data, voxels_size, velocity_mag_p, identity_minus_p_hat_multiplied_p, obs_grad, thread_index_p);

    (*obs_grad)(thread_index, 0)/=size;
    (*obs_grad)(thread_index, 1)/=size;
    (*obs_grad)(thread_index, 2)/=size;

    free(voxels);
    free(velocity_mag_p);
    free(size_p);
    free(thread_index_p);
    free(identity_minus_p_hat_multiplied_p);
}

void optimize_trajectory_cuda(std::vector<MultiDOF>  & drone_traj, std::vector<MultiDOF> & actor_traj, std::vector<Voxel> & voxels){
    int n = drone_traj.size();
    int * n_h = &n;
    int * n_d;
    cudaMalloc(&n_d, sizeof(*n_h));
    cudaMemcpy(n_d, n_h, sizeof(*n_h), cudaMemcpyHostToDevice);

    MultiDOF * drone_traj_h = &drone_traj[0];
    MultiDOF * actor_traj_h = &actor_traj[0];
    MultiDOF * drone_traj_d;
    MultiDOF * actor_traj_d;

    cudaMalloc(&drone_traj_d, sizeof(*drone_traj_h)*n);
    cudaMemcpy(drone_traj_d, drone_traj_h, sizeof(*drone_traj_h)*n, cudaMemcpyHostToDevice);
    cudaMalloc(&actor_traj_d, sizeof(*actor_traj_h)*n);
    cudaMemcpy(actor_traj_d, actor_traj_h, sizeof(*actor_traj_h)*n, cudaMemcpyHostToDevice);

    int voxels_size = voxels.size();
    int * voxels_size_h = &voxels_size;
    int * voxels_size_d;
    cudaMalloc(&voxels_size_h, sizeof(*voxels_size_h));
    cudaMemcpy(voxels_size_d, voxels_size_h, sizeof(*voxels_size_h), cudaMemcpyHostToDevice);

    Voxel * voxels_h = &voxels[0];
    Voxel * voxels_d;
    cudaMalloc(&voxels_d, sizeof(*voxels_h) * voxels_size);
    cudaMemcpy(voxels_d, voxels_h, sizeof(*voxels_h) * voxels_size, cudaMemcpyHostToDevice);

    // Eigen::Matrix<double, Eigen::Dynamic, 3> * obs_grad_h = new Eigen::MatrixXd(n-1,3);
    Eigen::Matrix<double, Eigen::Dynamic, 3> * obs_grad_d;
    cudaMalloc(&obs_grad_d, 24 * (n-1));
    // cudaMemcpy(obs_grad_d, obs_grad_h, sizeof(* obs_grad_h) * (n-1), cudaMemcpyHostToDevice);

    // Eigen::Matrix<double, Eigen::Dynamic, 3> * obs_grad_h = new Eigen::MatrixXd(n-1,3);
    Eigen::Matrix<double, Eigen::Dynamic, 3> * occ_grad_d;
    cudaMalloc(&occ_grad_d, 24 * (n-1));
    // cudaMemcpy(obs_grad_d, obs_grad_h, sizeof(* obs_grad_h) * (n-1), cudaMemcpyHostToDevice);
    int num_threads = 128;
    int num_blocks = n / num_threads + 1;

    //pass truncation distance and volume size
    for(size_t i = 0; i < 100; i++){
        obstacle_avoidance_gradient<<<num_blocks, num_threads>>>(drone_traj_d, obs_grad_d, n_d, voxels_d, voxels_size_d);
        cudaDeviceSynchronize();
    }

    // cudaMemcpy(obs_grad_h, obs_grad_d, sizeof(* obs_grad_h) * (n-1), cudaMemcpyDeviceToHost);

    cudaFree(n_d);
    cudaFree(drone_traj_d);
    cudaFree(actor_traj_d);
    cudaFree(voxels_size_d);
    cudaFree(voxels_d);
    cudaFree(obs_grad_d);
    cudaFree(occ_grad_d);

}