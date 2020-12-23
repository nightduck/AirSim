#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include "UKF/Types.h"
#include "UKF/Integrator.h"
#include "UKF/StateVector.h"
#include "UKF/MeasurementVector.h"
#include "UKF/Core.h"

enum MeasurementFields {
    BoundingBox,
    DronePosition,
    DroneYaw,
    CameraPitch,
    HDE
};

using MeasurementVector = UKF::DynamicMeasurementVector<
    UKF::Field<BoundingBox, UKF::Vector<3>>,
    UKF::Field<DronePosition, UKF::Vector<3>>,
    UKF::Field<DroneYaw, real_t>,
    UKF::Field<CameraPitch, real_t>,
    UKF::Field<HDE, real_t>
>;

enum StateFields {
    MagicVector,
    ActorPosition,
    ActorYaw,
    ActorVelocity,
    ActorYawVelocity
};

using StateVector = UKF::StateVector<
    UKF::Field<MagicVector, UKF::Vector<3>>,
    UKF::Field<ActorPosition, UKF::Vector<3>>,
    UKF::Field<ActorYaw, real_t>,
    UKF::Field<ActorVelocity, UKF::Vector<3>>,
    UKF::Field<ActorYawVelocity, real_t>
>;

using MyCore = UKF::Core<
    StateVector,
    MeasurementFields,
    UKF::IntegratorRK4
>;

template <> template <>
StateVector StateVector::derivative<UKF::Vector<3>>(
        const UKF::Vector<3>& actorVelocity, const UKF::Vector<3>& droneVelocity,
        const UKF::Vector<3>& droneAngularVelocity) const {
    StateVector temp;
    /* Magic number derivatives */
    float x1 = get_field<MagicVector>()[0];
    float x2 = get_field<MagicVector>()[1];
    float x3 = get_field<MagicVector>()[2];
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

    temp.set_field<MagicVector>(UKF::Vector<3>(mn1, mn2, mn3));

    /* Position derivative. */
    temp.set_field<ActorPosition>(get_field<ActorVelocity>());

    /* Velocity derivative. */
    temp.set_field<ActorVelocity>(UKF::Vector<3>(0,0,0));

    return temp;
}