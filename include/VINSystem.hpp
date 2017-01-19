#pragma once
#ifndef INCLUDE_VINSYSTEM_HPP_
#define INCLUDE_VINSYSTEM_HPP_

#include <sophus/se3.hpp>
#include <glog/logging.h>
#include<opencv2/core/core.hpp>
#include<string>
#include<thread>
#include <sdtrack/semi_dense_tracker.h>
#include "common/etc_common.h"
#include <ba/BundleAdjuster.h>

#include <HAL/IMU/IMUDevice.h>
#include <HAL/Messages/Matrix.h>
#include <HAL/Camera/CameraDevice.h>

#include <assert.h>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include <unistd.h>

#include <HAL/Camera/CameraDevice.h>
#include <HAL/IMU/IMUDevice.h>
#include <HAL/Messages/Matrix.h>
#include <ba/InterpolationBuffer.h>
#include <sdtrack/utils.h>
#include "common/math_types.h"
#include "common/chi2inv.h"
#include <thread>
#ifdef CHECK_NANS
#include <xmmintrin.h>
#endif




namespace compass
{

struct SystemOptions
{
    SystemOptions(){}
    double gyro_sigma = 1.3088444e-1;
    double gyro_bias_sigma =
       IMU_GYRO_BIAS_SIGMA;
    double accel_sigma = IMU_ACCEL_SIGMA;
    double accel_bias_sigma = IMU_ACCEL_BIAS_SIGMA;
    int pyramid_levels = 4;
    int patch_size = 9;
    int ba_debug_level = -1;
    int vi_ba_debug_level = -1;
    int aac_ba_debug_level = -1;
    uint32_t num_ba_poses = 10u;
    uint32_t num_aac_poses = 20u;
    int num_features = 128;
    int feature_cells = 8;
    bool use_imu_measurements = true;
    bool do_outlier_rejection = true;
    bool reset_outliers = false;
    double outlier_threshold = 2.0;
    bool use_dogleg = true;
    bool regularize_biases_in_batch = false;
    bool calculate_covariance_once = false;
    bool do_keyframing = true;
    bool do_adaptive = true;
    bool do_async_ba = true;
    bool use_imu_for_guess = true;
    bool use_robust_norm_for_proj = true;
    bool use_only_imu = false;
    double adaptive_threshold = 0.1;
    int num_ba_iterations = 200;
    uint32_t min_poses_for_imu = num_ba_poses - 1u;
    double imu_extra_integration_time = 0.3;
    double imu_time_offset = 0.0;
    double tracker_center_weight = 100.0;
    double ncc_threshold =  0.875;
    Eigen::Vector3d gravity_vector = (Eigen::Vector3d)(Eigen::Vector3d(0, 0, -1) *
                                                  ba::Gravity);


};

class VINSystem
{

public:

    /**
     * @brief VINSystem Constructor from settings file
     * @param settings_file_str filesystem path to the .yaml settings file
     */
    VINSystem(const std::string& settings_file_str);

    /**
     * @brief GetSystemOptions
     * @return SystemOptions struct, for manipulation on the application side.
     */
    SystemOptions& GetSystemOptions();

    /**
     * @brief GetLatestPose
     * @return latest possible pose, will be propagated to the latest imu
     *         measurement.
     */
    sdtrack::TrackerPose& GetLatestPose();

    /**
     * @brief GetLatestOptimizedPose
     * @return latest pose that was optimized (bundle adjustment).
     */
    sdtrack::TrackerPose& GetLatestOptimizedPose();


    /**
     * @brief Shutdown Kills the tracking thread and exits cleanly.
     */
    void Shutdown();

    /**
     * @brief ImuCallback Static callback for receiving data
     * @param ref
     */
    static void ImuCallback(const hal::ImuMsg &ref);
    static std::vector<VINSystem*> callback_devices;

    void ProcessImuMessage(const hal::ImuMsg& ref);


private:
    void UpdateCurrentPose();

    void InitTracker();

    void DoBA();

    void DoAAC();

    void Run();

    void BaAndStartNewLandmarks();

    bool LoadDevices();

    void ProcessImage(std::vector<cv::Mat>& images, double timestamp);

    /**
     * @brief PropagatePose prpagates the lates pose with all the currently
     * available IMU measurements to get the latest possible pose estimate.
     * @return Propagated pose.
     */
    sdtrack::TrackerPose& PropagatePose();

    template <typename BaType>
    void DoBundleAdjustment(BaType& ba, bool use_imu, uint32_t& num_active_poses,
                            bool initialize_lm, bool do_adaptive_conditioning,
                            uint32_t id, std::vector<uint32_t>& imu_residual_ids);

private:
    SystemOptions sys_options_;

    // sd_track variables
    uint32_t keyframe_tracks = UINT_MAX;
    double start_time = 0;
    uint32_t frame_count = 0;
    Sophus::SE3d last_t_ba, prev_delta_t_ba, prev_t_ba;

    bool is_keyframe = true, is_prev_keyframe = true;
    bool include_new_landmarks = true;
    bool optimize_landmarks = true;
    bool optimize_pose = true;
    bool is_manual_mode = false;
    bool do_bundle_adjustment = true;
    bool do_start_new_landmarks = true;
    bool use_system_time = false;

    std::string cam_uri_str_;
    std::string imu_uri_str_;
    std::string cmod_str_;

    std::mutex latest_pose_mutex_;



    calibu::Rig<Scalar> old_rig;
    calibu::Rig<Scalar> rig;
    hal::Camera camera_device;
    hal::IMU imu_device;

    sdtrack::SemiDenseTracker tracker;

    std::list<std::shared_ptr<sdtrack::DenseTrack>>* current_tracks = nullptr;
    std::shared_ptr<hal::Image> camera_img;
    std::vector<std::shared_ptr<sdtrack::TrackerPose>> poses;

    // Gps structures
    std::vector<std::shared_ptr<sdtrack::TrackerPose>> gps_poses;
    std::shared_ptr<std::thread> gps_thread;


    // Inertial stuff.
    std::mutex aac_mutex;
    std::shared_ptr<std::thread> aac_thread;
    std::shared_ptr<std::thread> tracking_thread;
    ba::BundleAdjuster<double, 1, 6, 0> bundle_adjuster;
    ba::BundleAdjuster<double, 1, 15, 0> vi_bundle_adjuster;
    ba::BundleAdjuster<double, 1, 15, 0> aac_bundle_adjuster;
    ba::InterpolationBufferT<ba::ImuMeasurementT<Scalar>, Scalar> imu_buffer;
    std::vector<uint32_t> ba_imu_residual_ids, aac_imu_residual_ids;
    int orig_num_aac_poses = 20;
    double prev_cond_error;
    int imu_cond_start_pose_id = -1;
    int imu_cond_residual_id = -1;


    // State variables
    std::vector<cv::KeyPoint> keypoints;



};


} // namespace compass

#endif // INCLUDE_VINSYSTEM_HPP_
