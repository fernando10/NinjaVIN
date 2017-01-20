#pragma once

#include <vinsystem/VINSystem.hpp>

namespace compass
{

void VINSystem::ImuCallback(const hal::ImuMsg &ref)
{

    for (VINSystem* device : VINSystem::callback_devices){
        device->ProcessImuMessage(ref);
    }

}

void VINSystem::ProcessImuMessage(const hal::ImuMsg& ref)
{
    const double timestamp = use_system_time ? ref.system_time() :
                                               ref.device_time();
    Eigen::VectorXd a, w;
    hal::ReadVector(ref.accel(), &a);
    hal::ReadVector(ref.gyro(), &w);
    // std::cerr << "Added accel: " << a.transpose() << " and gyro " <<
    //             w.transpose() << " at time " << timestamp << std::endl;
    imu_buffer.AddElement(ba::ImuMeasurementT<Scalar>(w, a, timestamp));
}

}// namespace compass
