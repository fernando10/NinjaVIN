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
    std::unique_lock<std::mutex>(imu_buffer_mutex_);
    imu_buffer.AddElement(ba::ImuMeasurementT<Scalar>(w, a, timestamp));
}

bool VINSystem::LoadDevices()
{
    LoadCameraAndRig(cam_uri_str_, cmod_str_, camera_device, rig);

    // Load the imu
    if (!imu_uri_str_.empty()) {
        try {
            imu_device = hal::IMU(imu_uri_str_);
        } catch (hal::DeviceException& e) {
            LOG(ERROR) << "Error loading imu device: " << e.what()
                       << " ... proceeding without.";
        }
        VLOG(1) << "Registering IMU callback function.";
        imu_device.RegisterIMUDataCallback(&VINSystem::ImuCallback);
    }
    // Capture an image so we have some IMU data.
    std::shared_ptr<hal::ImageArray> images = hal::ImageArray::Create();
    while (imu_buffer.elements.size() == 0) {
        camera_device.Capture(*images);
    }

    if (!use_system_time) {
        sys_options_.imu_time_offset = imu_buffer.elements.back().time -
                images->Ref().device_time();
        LOG(INFO) << "Setting initial time offset to " << sys_options_.imu_time_offset <<
                     std:: endl;
    }

    return true;
}

}// namespace compass
