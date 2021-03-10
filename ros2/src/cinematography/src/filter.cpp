
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include "UKF/Types.h"
#include "UKF/Integrator.h"
#include "UKF/StateVector.h"
#include "UKF/MeasurementVector.h"
#include "UKF/Core.h"
#include "filter.h"

static double fov = 90;
static int width = 672;
static int height = 672;

enum StateFields {
    CameraPosition,
    CameraQuat,
    CameraVelocity,
    CameraAngVelocity,
    CameraAcceleration,
    CameraAngAcceleration,
    ActorPosition,
    ActorYaw,
    ActorVelocity,
    ActorYawVelocity,
    ActorAcceleration,
    ActorYawAcceleration
};

using TrajectoryStateVector = UKF::StateVector<
    UKF::Field<CameraPosition, UKF::Vector<3>>,
    UKF::Field<CameraQuat, UKF::Vector<4>>,
    UKF::Field<CameraVelocity, UKF::Vector<3>>,
    UKF::Field<CameraAngVelocity, UKF::Vector<3>>,
    UKF::Field<CameraAcceleration, UKF::Vector<3>>,
    UKF::Field<CameraAngAcceleration, UKF::Vector<3>>,
    UKF::Field<ActorPosition, UKF::Vector<3>>,
    UKF::Field<ActorYaw, real_t>,
    UKF::Field<ActorVelocity, UKF::Vector<3>>,
    UKF::Field<ActorYawVelocity, real_t>,
    UKF::Field<ActorAcceleration, UKF::Vector<3>>,
    UKF::Field<ActorYawAcceleration, real_t>
>;


namespace UKF {
    template <> template<>
    TrajectoryStateVector TrajectoryStateVector::derivative<>() const {
        TrajectoryStateVector temp;

        /* Position derivative */
        temp.set_field<CameraPosition>(get_field<CameraVelocity>());
        UKF::Vector<3> ang_vel = get_field<CameraAngVelocity>();
        float cr = std::cos(ang_vel[0] * 0.5);
        float sr = std::sin(ang_vel[0] * 0.5);
        float cp = std::cos(ang_vel[1] * 0.5);
        float sp = std::sin(ang_vel[1] * 0.5);
        float cy = std::cos(ang_vel[2] * 0.5);
        float sy = std::sin(ang_vel[2] * 0.5);
        temp.set_field<CameraQuat>(
            UKF::Vector<4>(cr * cp * cy + sr * sp * sy, //w
                            sr * cp * cy - cr * sp * sy, //x
                            cr * sp * cy + sr * cp * sy, //y
                            cr * cp * sy - sr * sp * cy  //z
                            ));

        /* Velocity derivative */
        temp.set_field<CameraVelocity>(get_field<CameraAcceleration>());
        temp.set_field<CameraAngVelocity>(get_field<CameraAngAcceleration>());

        /* Acceleration derivative */
        temp.set_field<CameraAcceleration>(Vector<3>(0,0,0));
        temp.set_field<CameraAngAcceleration>(0);

        /* Position derivative */
        temp.set_field<ActorPosition>(get_field<ActorVelocity>());
        temp.set_field<ActorYaw>(get_field<ActorYawVelocity>());

        /* Velocity derivative */
        temp.set_field<ActorVelocity>(get_field<ActorAcceleration>());
        temp.set_field<ActorYawVelocity>(get_field<ActorYawAcceleration>());

        /* Acceleration derivative */
        temp.set_field<ActorAcceleration>(Vector<3>(0,0,0));
        temp.set_field<ActorYawAcceleration>(0);

        return temp;
    }
}

enum MeasurementFields {
    BoundingBox,
    Depth,
    DronePosition,
    DroneYaw,
    DronePitch,
    HDE
};

using MeasurementVector = UKF::DynamicMeasurementVector<
    UKF::Field<BoundingBox, UKF::Vector<2>>,
    UKF::Field<Depth, real_t>,
    UKF::Field<DronePosition, UKF::Vector<3>>,
    UKF::Field<DroneYaw, real_t>,
    UKF::Field<DronePitch, real_t>,
    UKF::Field<HDE, real_t>
>;

using MotionForecastingCore = UKF::Core<
    TrajectoryStateVector,
    MeasurementVector,
    UKF::IntegratorRK4
>;

UKF::Vector<4> flatten(UKF::Vector<4> quat) {
    double length = sqrt(quat[0] * quat[0] + quat[3] * quat[3]);
    return UKF::Vector<4>(quat[0] / length, 0, 0, quat[3] / length);
}

float getYaw(UKF::Vector<4> q) {
    q = flatten(q);
    double siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2]);
    double cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3]);
    return std::atan2(siny_cosp, cosy_cosp);
}

float getPitch(UKF::Vector<4> q) {
    UKF::Vector<4> f = flatten(q);

    float dot_product = f[0] * q[0] + f[1] * q[1] + f[2] * q[2] + f[3] * q[3];
    float angle = acos(2*dot_product*dot_product - 1);
    return angle;
}

namespace UKF {
    template <> template <>
    UKF::Vector<2> MeasurementVector::expected_measurement
    <TrajectoryStateVector, BoundingBox>(
            const TrajectoryStateVector& state) {
        double fx = width/fov;
        double fy = height/fov;
        int cu = width/2;
        int cv = height/2;

        UKF::Vector<3> diff = state.get_field<ActorPosition>() - state.get_field<CameraPosition>();

        float camera_yaw = getYaw(state.get_field<CameraQuat>());
        float camera_pitch = getPitch(state.get_field<CameraQuat>());

        float actor_yaw = std::atan2(diff[0], diff[1]);
        float actor_pitch = std::atan2(diff[2], std::sqrt(diff[1]*diff[1] + diff[0]*diff[0]));

        int u = fx*(actor_yaw - camera_yaw) + cu;
        int v = fy*(actor_pitch - camera_pitch) + cv;

        return UKF::Vector<2>(u, v);
    }
    template <> template <>
    real_t MeasurementVector::expected_measurement
    <TrajectoryStateVector, Depth>(
            const TrajectoryStateVector& state) {
        // Calculate vector r_q/c between drone and actor, and get the magnitude of that
        UKF::Vector<3> diff = state.get_field<CameraPosition>() - state.get_field<ActorPosition>();
        return diff.norm();
    }
    template <> template <>
    UKF::Vector<3> MeasurementVector::expected_measurement
    <TrajectoryStateVector, DronePosition>(
            const TrajectoryStateVector& state) {
        return state.get_field<CameraPosition>();
    }
    template <> template <>
    real_t MeasurementVector::expected_measurement
    <TrajectoryStateVector, DroneYaw>(
            const TrajectoryStateVector& state) {
        UKF::Vector<4> q = state.get_field<CameraQuat>();
        return getYaw(q);
    }
    template <> template <>
    real_t MeasurementVector::expected_measurement
    <TrajectoryStateVector, DronePitch>(
            const TrajectoryStateVector& state) {
        UKF::Vector<4> q = state.get_field<CameraQuat>();
        return getPitch(q);
    }
    template <> template <>
    real_t MeasurementVector::expected_measurement
    <TrajectoryStateVector, HDE>(
            const TrajectoryStateVector& state) {
        float yaw_actor = state.get_field<ActorYaw>();
        float yaw_camera = getYaw(state.get_field<CameraQuat>());
        return yaw_actor - yaw_camera;
    }
}

static MotionForecastingCore filter;
static MeasurementVector meas;

void ukf_init(float x, float y, float z, float yaw) {
    filter.state.set_field<CameraPosition>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<CameraQuat>(UKF::Vector<4>(cos(yaw/2), 0, 0, sin(yaw/2)));
    filter.state.set_field<CameraVelocity>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<CameraAngVelocity>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<CameraAcceleration>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<CameraAngAcceleration>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<ActorPosition>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<ActorYaw>(0);
    filter.state.set_field<ActorVelocity>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<ActorYawVelocity>(0);
    filter.state.set_field<ActorAcceleration>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<ActorYawAcceleration>(0);
    filter.covariance = TrajectoryStateVector::CovarianceMatrix::Identity();

    filter.process_noise_covariance = TrajectoryStateVector::CovarianceMatrix::Identity() * 0.1;
    filter.measurement_covariance << 1e-2, 1e-2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 1e-2;
}


cinematography_msgs::msg::MultiDOF get_state(rclcpp::Duration duration) {
    cinematography_msgs::msg::MultiDOF point;
    point.duration = duration.seconds();
    point.x = filter.state.get_field<ActorPosition>()[0];
    point.y = filter.state.get_field<ActorPosition>()[1];
    point.z = filter.state.get_field<ActorPosition>()[2];
    point.yaw = filter.state.get_field<ActorYaw>();
    point.vx = filter.state.get_field<ActorVelocity>()[0];
    point.vy = filter.state.get_field<ActorVelocity>()[1];
    point.vz = filter.state.get_field<ActorVelocity>()[2];
    point.ax = filter.state.get_field<ActorAcceleration>()[0];
    point.ay = filter.state.get_field<ActorAcceleration>()[1];
    point.az = filter.state.get_field<ActorAcceleration>()[2];

    return point;
}

cinematography_msgs::msg::MultiDOF get_state(rclcpp::Duration duration, MotionForecastingCore filter) {
    cinematography_msgs::msg::MultiDOF point;
    point.duration = duration.seconds();
    point.x = filter.state.get_field<ActorPosition>()[0];
    point.y = filter.state.get_field<ActorPosition>()[1];
    point.z = filter.state.get_field<ActorPosition>()[2];
    point.yaw = filter.state.get_field<ActorYaw>();
    point.vx = filter.state.get_field<ActorVelocity>()[0];
    point.vy = filter.state.get_field<ActorVelocity>()[1];
    point.vz = filter.state.get_field<ActorVelocity>()[2];
    point.ax = filter.state.get_field<ActorAcceleration>()[0];
    point.ay = filter.state.get_field<ActorAcceleration>()[1];
    point.az = filter.state.get_field<ActorAcceleration>()[2];

    return point;
}

// forecast length must be at least 2 (now and next point)
std::vector<cinematography_msgs::msg::MultiDOF> ukf_iterate(rclcpp::Duration point_duration, int forecast_length) {
    if (forecast_length < 2) {
        return std::vector<cinematography_msgs::msg::MultiDOF>();
    }

    std::vector<cinematography_msgs::msg::MultiDOF> path;
    path.reserve(forecast_length);

    path[0] = get_state(point_duration);
    
    filter.step(point_duration.seconds(), meas);
    path[1] = get_state(point_duration);

    MotionForecastingCore forecaster = filter;
    for(int i = 2; i < forecast_length; i++) {
        forecaster.step(point_duration.seconds(), MeasurementVector());
        path[i] = get_state(point_duration, forecaster);
    }

    return path;
}

void ukf_meas_clear() {
    meas = MeasurementVector();
}

void ukf_set_fov(float f) {
    fov = f;
}

void ukf_set_bb(float width, float height) {
    meas.set_field<BoundingBox>(UKF::Vector<2>(width, height));
}

void ukf_set_depth(float depth) {
    meas.set_field<Depth>(depth);
}

void ukf_set_position(float x, float y, float z) {
    meas.set_field<DronePosition>(UKF::Vector<3>(x, y, z));
}

void ukf_set_yaw(float yaw) {
    meas.set_field<DroneYaw>(yaw);
}

void ukf_set_pitch(float pitch) {
    meas.set_field<DronePitch>(pitch);
}

void ukf_set_hde(float hde) {
    meas.set_field<HDE>(hde);
}