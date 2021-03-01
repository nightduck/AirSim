
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

double fov = 90;
int width = 672;
int height = 672;
int actor_area = 1806336;

enum StateFields {
    MagicVector,
    CameraYaw,
    CameraPitch,
    ActorPosition,
    ActorYaw,
    ActorVelocity,
    ActorYawVelocity,
    ActorAcceleration,
    ActorYawAcceleration
};

using TrajectoryStateVector = UKF::StateVector<
    UKF::Field<MagicVector, UKF::Vector<3>>,
    UKF::Field<CameraYaw, real_t>,
    UKF::Field<CameraPitch, real_t>,
    UKF::Field<ActorPosition, UKF::Vector<3>>,
    UKF::Field<ActorYaw, real_t>,
    UKF::Field<ActorVelocity, UKF::Vector<3>>,
    UKF::Field<ActorYawVelocity, real_t>,
    UKF::Field<ActorAcceleration, UKF::Vector<3>>,
    UKF::Field<ActorYawAcceleration, real_t>
>;


namespace UKF {
    template <> template<>
    TrajectoryStateVector TrajectoryStateVector::derivative<Vector<3>, Vector<3>>(
            const Vector<3>& droneVelocity,
            const Vector<3>& droneAngularVelocity) const {
        TrajectoryStateVector temp;
        /* Magic number derivatives */
        float x1 = get_field<MagicVector>()[0];
        float x2 = get_field<MagicVector>()[1];
        float x3 = get_field<MagicVector>()[2];
        Vector<3> actorVelocity = get_field<ActorVelocity>();
        float vqx = actorVelocity[0];
        float vqy = actorVelocity[1];
        float vqz = actorVelocity[2];
        float vcx = droneVelocity[0];
        float vcy = droneVelocity[1];
        float vcz = droneVelocity[2];
        float wcx = droneAngularVelocity[0];
        float wcy = droneAngularVelocity[1];
        float wcz = droneAngularVelocity[2];
        float mn1 = vqx * x3 - vqz * x1 * x3 + wcx * x2 - wcy - wcy * x1 * x1 - wcx * x1 * x2 + x3 * (vcz * x1 - vcx);
        float mn2 = vqy * x3 - vqz * x2 * x3 - wcz * x2 + wcx - wcx * x2 * x2 - wcy * x1 * x2 + x3 * (vcz * x2 - vcy);
        float mn3 = -vqz * x3 * x3 + vcz * x3 * x3 - (wcy * x1 - wcx * x2) * x3;

        temp.set_field<MagicVector>(Vector<3>(mn1, mn2, mn3));

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
    DronePosition,
    DroneYaw,
    DronePitch,
    HDE
};

using MeasurementVector = UKF::DynamicMeasurementVector<
    UKF::Field<BoundingBox, UKF::Vector<3>>,
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

namespace UKF {
    template <> template <>
    UKF::Vector<3> MeasurementVector::expected_measurement
    <TrajectoryStateVector, BoundingBox, Vector<3>, Vector<3>>(
            const TrajectoryStateVector& state,
            const Vector<3>& droneVelocity,
            const Vector<3>& droneAngularVelocity) {
        double fx = fov/width;
        double fy = fov/height;
        int cu = width/2;
        int cv = height/2;

        int u = fx*state.get_field<BoundingBox>()[0] + cu;
        int v = fy*state.get_field<BoundingBox>()[1] + cv;
        int a = actor_area * fx * fy * std::pow(state.get_field<BoundingBox>()[2],2);

        return UKF::Vector<3>(u, v, a);
    }
    template <> template <>
    UKF::Vector<3> MeasurementVector::expected_measurement
    <TrajectoryStateVector, DronePosition, Vector<3>, Vector<3>>(
            const TrajectoryStateVector& state,
            const Vector<3>& droneVelocity,
            const Vector<3>& droneAngularVelocity) {
        float x1 = state.get_field<MagicVector>()[0];
        float x2 = state.get_field<MagicVector>()[1];
        float x3 = state.get_field<MagicVector>()[2];
        float pitch = state.get_field<CameraPitch>();
        float yaw = state.get_field<CameraYaw>();

        // Get r_q/c vector from magic numbers and then use camera pitch and yaw to correct it into
        // NED coordinate frame
        UKF::Vector<3> rqc = UKF::Vector<3>(1/x3, x1/x3, x2/x3);
        Eigen::Matrix<real_t, 3, 3> r_pitch = Eigen::MatrixXd::Identity(3,3).cast<real_t>();
        Eigen::Matrix<real_t, 3, 3> r_yaw = Eigen::MatrixXd::Identity(3,3).cast<real_t>();
        r_pitch(0,0) = std::cos(pitch);
        r_pitch(0,2) = std::sin(pitch);
        r_pitch(2,0) = -std::sin(pitch);
        r_pitch(2,2) = std::cos(pitch);
        r_pitch(0,0) = std::cos(yaw);
        r_pitch(0,1) = -std::sin(yaw);
        r_pitch(1,0) = std::sin(yaw);
        r_pitch(1,1) = std::cos(yaw);
        Eigen::Matrix<real_t, 3, 3> correct_to_ned = r_pitch * r_yaw;
        correct_to_ned = correct_to_ned.inverse();
        rqc = correct_to_ned * rqc;

        UKF::Vector<3> rq = state.get_field<ActorPosition>();

        // return r_c.  r_c = r_q - r_q/c
        return rq - rqc;
    }
    template <> template <>
    real_t MeasurementVector::expected_measurement
    <TrajectoryStateVector, DroneYaw, Vector<3>, Vector<3>>(
            const TrajectoryStateVector& state,
            const Vector<3>& droneVelocity,
            const Vector<3>& droneAngularVelocity) {
        return state.get_field<CameraYaw>();
    }
    template <> template <>
    real_t MeasurementVector::expected_measurement
    <TrajectoryStateVector, DronePitch, Vector<3>, Vector<3>>(
            const TrajectoryStateVector& state,
            const Vector<3>& droneVelocity,
            const Vector<3>& droneAngularVelocity) {
        return state.get_field<CameraPitch>();
    }
    template <> template <>
    real_t MeasurementVector::expected_measurement
    <TrajectoryStateVector, HDE, Vector<3>, Vector<3>>(
            const TrajectoryStateVector& state,
            const Vector<3>& droneVelocity,
            const Vector<3>& droneAngularVelocity) {
        float yaw_actor = state.get_field<ActorYaw>();
        float yaw_drone = state.get_field<DroneYaw>();
        float x1 = state.get_field<MagicVector>()[0];
        double fx = fov/width;

        return yaw_drone - yaw_actor + fx * x1;
    }
}

static MotionForecastingCore filter;
static MeasurementVector meas;

void ukf_init() {
    filter.state.set_field<MagicVector>(UKF::Vector<3>());
    filter.state.set_field<CameraYaw>(0);
    filter.state.set_field<CameraPitch>(0);
    filter.state.set_field<ActorPosition>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<ActorYaw>(0);
    filter.state.set_field<ActorVelocity>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<ActorYawVelocity>(0);
    filter.state.set_field<ActorAcceleration>(UKF::Vector<3>(0,0,0));
    filter.state.set_field<ActorYawAcceleration>(0);
    filter.covariance = TrajectoryStateVector::CovarianceMatrix::Zero();
    filter.covariance.diagonal() << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

    filter.process_noise_covariance = TrajectoryStateVector::CovarianceMatrix::Identity() * 0.1;
    filter.measurement_covariance << 1;
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
std::vector<cinematography_msgs::msg::MultiDOF> ukf_iterate(rclcpp::Duration point_duration, int forecast_length, float vx, float vy, float vz, float wx, float wy, float wz) {
    if (forecast_length < 2) {
        return std::vector<cinematography_msgs::msg::MultiDOF>();
    }

    std::vector<cinematography_msgs::msg::MultiDOF> path;
    path.reserve(forecast_length);

    path[0] = get_state(point_duration);
    
    filter.step(point_duration.seconds(), meas, UKF::Vector<3>(vx, vy, vz), UKF::Vector<3>(wx, wy, wz));
    path[1] = get_state(point_duration);

    MotionForecastingCore forecaster = filter;
    for(int i = 2; i < forecast_length; i++) {
        forecaster.step(point_duration.seconds(), MeasurementVector(), UKF::Vector<3>(0,0,0), UKF::Vector<3>(0,0,0));
        path[i] = get_state(point_duration, forecaster);
    }

    return path;
}

void ukf_meas_clear() {
    meas = MeasurementVector();
}

void ukf_set_bb(float width, float height, float area) {
    meas.set_field<BoundingBox>(UKF::Vector<3>(width, height, area));
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