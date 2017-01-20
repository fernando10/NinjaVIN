#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sdtrack/track.h>
#include <sdtrack/utils.h>
#include <sdtrack/semi_dense_tracker.h>

#include <CVars/CVarVectorIO.h>

#include <calibu/cam/camera_crtp.h>
#include <calibu/cam/camera_xml.h>
#include <calibu/cam/camera_rig.h>
#include <glog/logging.h>
#include <sdtrack/utils.h>
#include "math_types.h"
#include "timer.h"
#include <HAL/Camera/CameraDevice.h>



bool LoadCameraAndRig(std::string cam_string, std::string cmod,
                      hal::Camera& camera_device,
                      calibu::Rig<Scalar>& rig,
                      bool transform_to_robotics_coords = true) {

  try {
    camera_device = hal::Camera(hal::Uri(cam_string));
  }
  catch (hal::DeviceException& e) {
    LOG(ERROR) << "Error loading camera device: " << e.what();
    return false;
  }

  std::string def_dir("");
  def_dir = camera_device.GetDeviceProperty(hal::DeviceDirectory);

  LOG(INFO) << "Loading camera models...";
  std::string filename = def_dir + "/" + cmod;
  LOG(INFO) << "Loading camera models from " << filename;

  std::shared_ptr<calibu::Rig<Scalar>> xmlrig = calibu::ReadXmlRig(filename);
  if (xmlrig->cameras_.empty()) {
    LOG(FATAL) << "XML Camera rig is empty!";
  }

  std::shared_ptr<calibu::Rig<Scalar>> crig = xmlrig;
  if (transform_to_robotics_coords) {
    crig = calibu::ToCoordinateConvention<Scalar>(
        xmlrig, calibu::RdfRobotics.cast<Scalar>());

    Sophus::SE3t M_rv;
    M_rv.so3() = calibu::RdfRobotics;
    for (std::shared_ptr<calibu::CameraInterface<Scalar>> model : crig->cameras_)
      {
    model->SetPose(model->Pose() * M_rv);
      }

  }

  LOG(INFO) << "Starting Tvs: " << crig->cameras_[0]->Pose().matrix();

  rig.cameras_.clear();
  for (uint32_t cam_id = 0; cam_id < crig->cameras_.size(); ++cam_id) {
    rig.AddCamera(crig->cameras_[cam_id]);
  }

  for (size_t ii = 0; ii < rig.cameras_.size(); ++ii) {
    LOG(INFO) << ">>>>>>>> Camera " << ii << ":" << std::endl
              << "Model: " << std::endl << rig.cameras_[ii]->K()
              << std::endl << "Pose: " << std::endl
              << rig.cameras_[ii]->Pose().matrix();
  }
  return true;
}



namespace sdtrack {
struct TrackerPose {
  TrackerPose() { opt_id.resize(3); }

  std::list<std::shared_ptr<DenseTrack>> tracks;
  Sophus::SE3t t_wp;
  Eigen::Vector3t v_w;
  Eigen::Vector6t b;
  std::vector<uint32_t> opt_id;
  std::vector<Sophus::SE3t> calib_t_wp;
  Eigen::VectorXd cam_params;
  double time;
  uint32_t longest_track;
};

inline void GetBaPoseRange(
    const std::vector<std::shared_ptr<sdtrack::TrackerPose>>& poses,
    const uint32_t num_active_poses, uint32_t& start_pose,
    uint32_t& start_active_pose) {
  start_active_pose =
      poses.size() > num_active_poses ? poses.size() - num_active_poses : 0;
  start_pose = start_active_pose;
  for (uint32_t ii = start_active_pose; ii < poses.size(); ++ii) {
    std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
//     std::cerr << "Start id: " << start_pose << " pose longest track " <<
//                  pose->longest_track << " for pose id " << ii << std::endl;
    start_pose = std::min(ii - (pose->longest_track - 1), start_pose);
  }


//  std::cerr << "Num poses: " << poses.size() << " start pose " <<
//               start_pose << " start active pose " << start_active_pose <<
//               std::endl;
}
}
