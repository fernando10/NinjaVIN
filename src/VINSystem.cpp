
#include <vinsystem/VINSystem.hpp>
#include <vinsystem/SystemDevices.hpp>

#define POSES_TO_INIT 10

namespace compass
{
// initialization of static variable
std::vector<VINSystem*> VINSystem::callback_devices =
        std::vector<VINSystem*>();

VINSystem::VINSystem(const std::string &settings_file_str)
{
    srand(0);

    // Read in settings file
    cv::FileStorage fsSettings(settings_file_str.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        LOG(FATAL) << "Failed to open settings file at: "
                   << settings_file_str;
    }

    FLAGS_v = fsSettings["Debug.verbose_level"];

    // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
    FLAGS_stderrthreshold = fsSettings["Debug.log_threshold"];

    // Get sensor URI strings
    cam_uri_str_ = static_cast<std::string>(fsSettings["Camera.uri"]);
    cmod_str_ = static_cast<std::string>(fsSettings["Camera.config"]);
    imu_uri_str_ = static_cast<std::string>(fsSettings["IMU.uri"]);

    VLOG(1) << "Got cam URI: " << cam_uri_str_;
    VLOG(1) << "Got cam config URI: " << cmod_str_;
    VLOG(1) << "Got imu URI: " << imu_uri_str_;


    use_system_time =
            static_cast<int>(fsSettings["Devices.use_system_time"]) != 0;

    VLOG(1) << "Use system time: " << use_system_time;

    sys_options_.imu_time_offset = fsSettings["IMU.time_offset"];
    sys_options_.patch_size = fsSettings["Tracker.patch_size"];
    sys_options_.num_features = fsSettings["Tracker.num_features"];

    // Register this instance to receive the sensor messages from HAL.
    VINSystem::callback_devices.push_back(this);

    latest_pose_ = std::make_shared<sdtrack::TrackerPose>();

    // Load camera and IMU
    LoadDevices();
    VLOG(1) << "Finished loading devices.";


    VLOG(1) << "Initializing Tracker";
    InitTracker();

    VLOG(1) << "Starting threads...";
    StartThreads();

}


template <typename BaType>
void VINSystem::DoBundleAdjustment(BaType& ba, bool use_imu, uint32_t& num_active_poses,
                                   bool initialize_lm, bool do_adaptive_conditioning,
                                   uint32_t id, std::vector<uint32_t>& imu_residual_ids)
{
    if (initialize_lm) {
        use_imu = false;
    }

    if (sys_options_.reset_outliers) {
        for (std::shared_ptr<sdtrack::TrackerPose> pose : poses) {
            for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
                track->is_outlier = false;
            }
        }
        sys_options_.reset_outliers = false;
    }

    bundle_adjuster.debug_level_threshold = sys_options_.ba_debug_level;
    vi_bundle_adjuster.debug_level_threshold = sys_options_.vi_ba_debug_level;
    aac_bundle_adjuster.debug_level_threshold = sys_options_.aac_ba_debug_level;

    imu_residual_ids.clear();
    ba::Options<double> options;
    options.gyro_sigma = sys_options_.gyro_sigma;
    options.accel_sigma = sys_options_.accel_sigma;
    options.accel_bias_sigma = sys_options_.accel_bias_sigma;
    options.gyro_bias_sigma = sys_options_.gyro_bias_sigma;
    options.use_dogleg = sys_options_.use_dogleg;
    options.use_sparse_solver = true;
    options.param_change_threshold = 1e-10;
    options.error_change_threshold = 1e-3;
    options.use_robust_norm_for_proj_residuals =
            sys_options_.use_robust_norm_for_proj && !initialize_lm;
    options.projection_outlier_threshold = sys_options_.outlier_threshold;
    options.regularize_biases_in_batch = poses.size() < POSES_TO_INIT ||
            sys_options_.regularize_biases_in_batch;
    options.calculate_inertial_covariance_once = sys_options_.calculate_covariance_once;
    uint32_t num_outliers = 0;
    Sophus::SE3d t_ba;
    // Find the earliest pose touched by the current tracks.
    uint32_t start_active_pose, start_pose_id;

    uint32_t end_pose_id;
    {
        std::lock_guard<std::mutex> lock(aac_mutex);
        end_pose_id = poses.size() - 1;

        GetBaPoseRange(poses, num_active_poses, start_pose_id, start_active_pose);

        if (start_pose_id == end_pose_id) {
            return;
        }

        // Add an extra pose to conditon the IMU
        if (use_imu && sys_options_.use_imu_measurements && start_active_pose == start_pose_id &&
                start_pose_id != 0) {
            start_pose_id--;
            VLOG(3) << "expanding sp from " << start_pose_id - 1 << " to " << start_pose_id << std::endl;
        }
    }

    bool all_poses_active = start_active_pose == start_pose_id;


    // Do a bundle adjustment on the current set
    if (current_tracks && end_pose_id) {

        {
            std::lock_guard<std::mutex> lock(aac_mutex);
            if (use_imu) {
                ba.SetGravity(sys_options_.gravity_vector);
            }
            ba.Init(options, end_pose_id + 1,
                    current_tracks->size() * (end_pose_id + 1));
            for (uint32_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
                ba.AddCamera(rig.cameras_[cam_id]);
            }

            // First add all the poses and landmarks to ba.
            for (uint32_t ii = start_pose_id ; ii <= end_pose_id ; ++ii) {
                std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
                const bool is_active = ii >= start_active_pose && !initialize_lm;
                pose->opt_id[id] = ba.AddPose(
                            pose->t_wp, Eigen::VectorXt(), pose->v_w, pose->b,
                            is_active, pose->time);
                if (ii == start_active_pose && use_imu && all_poses_active) {
                    ba.RegularizePose(pose->opt_id[id], true, true, false, false);
                }

                if (use_imu && ii >= start_active_pose && ii > 0) {
                    std::vector<ba::ImuMeasurementT<Scalar>> meas =
                            imu_buffer.GetRange(poses[ii - 1]->time, pose->time);
                    /*std::cerr << "Adding imu residual between poses " << ii - 1 << " with "
                     " time " << poses[ii - 1]->time <<  " and " << ii <<
                     " with time " << pose->time << " with " << meas.size() <<
                     " measurements" << std::endl;*/

                    imu_residual_ids.push_back(
                                ba.AddImuResidual(poses[ii - 1]->opt_id[id],
                                pose->opt_id[id], meas));
                    // Store the conditioning edge of the IMU.
                    if (do_adaptive_conditioning) {
                        if (imu_cond_start_pose_id == -1 &&
                                !ba.GetPose(poses[ii - 1]->opt_id[id]).is_active &&
                                ba.GetPose(pose->opt_id[id]).is_active) {
                            // std::cerr << "Setting cond pose id to " << ii - 1 << std::endl;
                            imu_cond_start_pose_id = ii - 1;
                            imu_cond_residual_id = imu_residual_ids.back();
                            // std::cerr << "Setting cond residual id to " <<
                            //              imu_cond_residual_id << std::endl;
                        } else if ((uint32_t)imu_cond_start_pose_id == ii - 1) {
                            imu_cond_residual_id = imu_residual_ids.back();
                            // std::cerr << "Setting cond residual id to " <<
                            //              imu_cond_residual_id << std::endl;
                        }
                    }
                }

                if (!sys_options_.use_only_imu) {
                    for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
                        const bool constrains_active =
                                track->keypoints.size() + ii >= start_active_pose;
                        if (track->num_good_tracked_frames <= 1 || track->is_outlier ||
                                !constrains_active) {
                            /*
            std::cerr << "ignoring track " << track->id << " with " <<
                         track->keypoints.size() << "keypoints with ngf " <<
                         track->num_good_tracked_frames << " outlier: " <<
                         track->is_outlier << " constraints " << constrains_active <<
                         std::endl;
                         */
                            track->external_id[id] = UINT_MAX;
                            continue;
                        }

                        Eigen::Vector4d ray;
                        ray.head<3>() = track->ref_keypoint.ray;
                        ray[3] = track->ref_keypoint.rho;
                        ray = sdtrack::MultHomogeneous(
                                    pose->t_wp * rig.cameras_[track->ref_cam_id]->Pose(), ray);
                        bool active = track->id != tracker.longest_track_id() ||
                                !all_poses_active || use_imu || initialize_lm;
                        if (!active) {
                            VLOG(3) << "Landmark " << track->id << " inactive. outlier = " <<
                                         track->is_outlier << " length: " <<
                                         track->keypoints.size() << std::endl;
                        }
                        track->external_id[id] =
                                ba.AddLandmark(ray, pose->opt_id[id], track->ref_cam_id, active);
                    }
                }
            }

            if (!sys_options_.use_only_imu) {
                // Now add all reprojections to ba)
                for (uint32_t ii = start_pose_id ; ii <= end_pose_id ; ++ii) {
                    std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
                    for (std::shared_ptr<sdtrack::DenseTrack> track : pose->tracks) {
                        if (track->external_id[id] == UINT_MAX) {
                            continue;
                        }
                        for (uint32_t cam_id = 0; cam_id < rig.cameras_.size(); ++cam_id) {
                            for (size_t jj = 0; jj < track->keypoints.size() ; ++jj) {
                                if (track->keypoints[jj][cam_id].tracked) {
                                    const Eigen::Vector2d& z = track->keypoints[jj][cam_id].kp;
                                    if (ba.GetNumPoses() > (pose->opt_id[id] + jj)) {
                                        ba.AddProjectionResidual(
                                                    z, pose->opt_id[id] + jj,
                                                    track->external_id[id], cam_id, 2.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


        ba.Solve(sys_options_.num_ba_iterations);

        {
            std::lock_guard<std::mutex> lock(aac_mutex);

            uint32_t last_pose_id =
                    is_keyframe ? poses.size() - 1 : poses.size() - 2;
            std::shared_ptr<sdtrack::TrackerPose> last_pose = is_keyframe ?
                        poses.back() : poses[poses.size() - 2];

            if (last_pose_id <= end_pose_id) {
                // Get the pose of the last pose. This is used to calculate the relative
                // transform from the pose to the current pose.
                last_pose->t_wp = ba.GetPose(last_pose->opt_id[id]).t_wp;
            }
//            VLOG(3) << "last pose t_wp: " << std::endl << last_pose->t_wp.matrix() <<
//                         std::endl;


            // Read out the pose and landmark values.
            for (uint32_t ii = start_pose_id ; ii <= end_pose_id ; ++ii) {
                std::shared_ptr<sdtrack::TrackerPose> pose = poses[ii];
                const ba::PoseT<double>& ba_pose = ba.GetPose(pose->opt_id[id]);

                if (!initialize_lm) {
                    pose->t_wp = ba_pose.t_wp;
                    if (use_imu) {
                        pose->v_w = ba_pose.v_w;
                        pose->b = ba_pose.b;
                    }
                }

                if (!sys_options_.use_only_imu) {
                    // Here the last pose is actually t_wb and the current pose t_wa.
                    last_t_ba = t_ba;
                    t_ba = last_pose->t_wp.inverse() * pose->t_wp;
                    for (std::shared_ptr<sdtrack::DenseTrack> track: pose->tracks) {
                        if (track->external_id[id] == UINT_MAX) {
                            continue;
                        }

                        if (!initialize_lm) {
                            track->t_ba = t_ba;
                        }

                        // Get the landmark location in the world frame.
                        const Eigen::Vector4d& x_w =
                                ba.GetLandmark(track->external_id[id]);
                        double ratio = ba.LandmarkOutlierRatio(track->external_id[id]);

                        if (sys_options_.do_outlier_rejection && poses.size() > POSES_TO_INIT &&
                                !initialize_lm /*&& (do_adaptive_conditioning || !do_async_ba)*/) {
                            if (ratio > 0.3 && track->tracked == false &&
                                    (end_pose_id >= sys_options_.min_poses_for_imu - 1 || use_imu)) {
                                /*
              std::cerr << "Rejecting landmark with outliers : ";
              for (int id: landmark.proj_residuals) {
                typename BaType::ProjectionResidual res =
                    ba.GetProjectionResidual(id);
                std::cerr << res.residual.transpose() << "(" << res.residual.norm() <<
                             "), ";
              }
              std::cerr << std::endl;
              */
                                num_outliers++;
                                track->is_outlier = true;
                            } else {
                                track->is_outlier = false;
                            }
                        }

                        // Make the ray relative to the pose.
                        Eigen::Vector4d x_r =
                                sdtrack::MultHomogeneous(
                                    (pose->t_wp * rig.cameras_[track->ref_cam_id]->Pose()).inverse(), x_w);
                        // Normalize the xyz component of the ray to compare to the original
                        // ray.
                        x_r /= x_r.head<3>().norm();
                        /*
          if (track->keypoints.size() >= min_lm_measurements_for_drawing) {
            std::cerr << "Setting rho for track " << track->id << " with " <<
                         track->keypoints.size() << " kps from " <<
                         track->ref_keypoint.rho << " to " << x_r[3] << std::endl;
          }
          */
                        track->ref_keypoint.rho = x_r[3];
                    }
                }
            }

        }

    }
    const ba::SolutionSummary<Scalar>& summary = ba.GetSolutionSummary();
    // std::cerr << "Rejected " << num_outliers << " outliers." << std::endl;

    if (use_imu && imu_cond_start_pose_id != -1 && do_adaptive_conditioning) {
        const uint32_t cond_dims =
                summary.num_cond_inertial_residuals * BaType::kPoseDim +
                summary.num_cond_proj_residuals * 2;
        //    const uint32_t active_dims = summary.num_inertial_residuals +
        //        summary.num_proj_residuals - cond_dims;
        const Scalar cond_error = summary.cond_inertial_error +
                summary.cond_proj_error;
        //    const Scalar active_error =
        //        summary.inertial_error + summary.proj_error_ - cond_error;

        const double cond_inertial_error =
                ba.GetImuResidual(
                    imu_cond_residual_id).mahalanobis_distance;

        if (prev_cond_error == -1) {
            prev_cond_error = DBL_MAX;
        }

        //    const Scalar cond_chi2_dist = chi2inv(adaptive_threshold, cond_dims);
        const Scalar cond_v_chi2_dist =
                chi2inv(sys_options_.adaptive_threshold, summary.num_cond_proj_residuals * 2);
        const Scalar cond_i_chi2_dist =
                chi2inv(sys_options_.adaptive_threshold, BaType::kPoseDim);
        //    const Scalar active_chi2_dist = chi2inv(adaptive_threshold, active_dims);
        //plot_logs[0].Log(cond_i_chi2_dist, cond_inertial_error);
        //plot_logs[2].Log(cond_v_chi2_dist, summary.cond_proj_error);
        // plot_logs[2].Log(cond_chi2_dist, cond_error);
        // plot_logs[2].Log(poses[start_active_pose]->v_w.norm(),
        //                  poses.back()->v_w.norm());

        /*
    std::cerr << "chi2inv(" << adaptive_threshold << ", " << cond_dims <<
                 "): " << cond_chi2_dist << " vs. " << cond_error <<
                 std::endl;

    std::cerr << "v_chi2inv(" << adaptive_threshold << ", " <<
                 summary.num_cond_proj_residuals * 2 << "): " <<
                 cond_v_chi2_dist << " vs. " <<
                 summary.cond_proj_error << std::endl;

    std::cerr << "i_chi2inv(" << adaptive_threshold << ", " <<
                 BaType::kPoseDim << "):" << cond_i_chi2_dist << " vs. " <<
                 cond_inertial_error << std::endl;

    std::cerr << "ec/Xc: " << cond_error / cond_chi2_dist << " ea/Xa: " <<
                 active_error / active_chi2_dist << std::endl;

    std::cerr << summary.num_cond_proj_residuals * 2 << " cond proj residuals "
                 " with dist: " << summary.cond_proj_error << " vs. " <<
                 summary.num_proj_residuals * 2 <<
                 " total proj residuals with dist: " <<
                 summary.proj_error_ << " and " <<
                 summary.num_cond_inertial_residuals * BaType::kPoseDim <<
                 " total cond imu residuals with dist: " <<
                 summary.cond_inertial_error <<
                 " vs. " << summary.num_inertial_residuals *
                 BaType::kPoseDim << " total imu residuals with dist : " <<
                 summary.inertial_error << std::endl;
    */
        // if (do_adaptive_conditioning) {
        if (num_active_poses > end_pose_id) {
            num_active_poses = orig_num_aac_poses;
            VLOG(3) << "Reached batch solution. resetting number of poses to " <<
                         sys_options_.num_ba_poses << std::endl;
        }

        if (cond_error == 0 || cond_dims == 0) {
            // status = OptStatus_NoChange;
        } else {
            const double cond_total_error =
                    (cond_inertial_error + summary.cond_proj_error);
            const double inertial_ratio = cond_inertial_error / cond_i_chi2_dist;
            const double visual_ratio = summary.cond_proj_error / cond_v_chi2_dist;
            if ((inertial_ratio > 1.0 || visual_ratio > 1.0) &&
                    (cond_total_error <= prev_cond_error) &&
                    (((prev_cond_error - cond_total_error) / prev_cond_error) > 0.00001)) {
                num_active_poses += 30;//(start_active_pose - start_pose);
                // std::cerr << "INCREASING WINDOW SIZE TO " << num_active_poses <<
                //              std::endl;
            } else /*if (ratio < 0.3)*/ {
                num_active_poses = orig_num_aac_poses;
                // std::cerr << "RESETTING WINDOW SIZE TO " << num_active_poses <<
                //              std::endl;
            }
            prev_cond_error = cond_total_error;
        }

    }
}


void VINSystem::Shutdown()
{
    LOG(INFO) << "VIN System shutting down, goodbye!";
    if (aac_thread->joinable()) aac_thread->join();
    if (tracking_thread->joinable()) tracking_thread->join();

}

const std::vector<std::shared_ptr<sdtrack::TrackerPose> >&
VINSystem::GetOptimzedPoses()
{
    return poses;
}

void VINSystem::Run()
{
    VLOG(1) << "Starting main tracking loop.";

    bool capture_success = false;
    std::shared_ptr<hal::ImageArray> images = hal::ImageArray::Create();
    camera_device.Capture(*images);

    while(1)
    {
        capture_success = false;
        capture_success = camera_device.Capture(*images);

        if (capture_success) {
            double timestamp = use_system_time ? images->Ref().system_time() :
                                                 images->Ref().device_time();


            // Wait until we have enough measurements to interpolate this frame's
            // timestamp
            const double start_time = sdtrack::Tic();
            while (imu_buffer.end_time < timestamp &&
                   sdtrack::Toc(start_time) < 0.1) {
                usleep(10);
            }

            std::vector<cv::Mat> cvmat_images;
            for (int ii = 0; (unsigned int) ii < (unsigned int) images->Size() ; ++ii) {
                cvmat_images.push_back(images->at(ii)->Mat());
            }
            ProcessImage(cvmat_images, timestamp);
            {
            std::lock_guard<std::mutex>lck(latest_pose_mutex_);
            std::shared_ptr<sdtrack::TrackerPose> last_pose = poses.back();
            latest_pose_->t_wp = last_pose->t_wp;
            latest_pose_->v_w = last_pose->v_w;
            latest_pose_->b = last_pose->b;
            latest_pose_->cam_params = last_pose->cam_params;
            latest_pose_->time = last_pose->time;
            }
        }else{
            VLOG(2) << "Capture image failed...waiting for a bit.";
            usleep(1000);
        }
    }
}

void VINSystem::UpdateCurrentPose()
{
    std::shared_ptr<sdtrack::TrackerPose> new_pose = poses.back();
    if (poses.size() > 1) {
        new_pose->t_wp = poses[poses.size() - 2]->t_wp * tracker.t_ba().inverse();
    }

    // Also use the current tracks to update the index of the earliest covisible
    // pose.
    size_t max_track_length = 0;
    for (std::shared_ptr<sdtrack::DenseTrack>& track : tracker.GetCurrentTracks()) {
        max_track_length = std::max(track->keypoints.size(), max_track_length);
    }
    new_pose->longest_track = max_track_length;

//    // Update the latest optimized pose with the tracker estimate
//    {
//        std::lock_guard<std::mutex>lck(latest_pose_mutex_);
//        latest_pose_ = new_pose;
//    }
}


void VINSystem::DoAAC()
{
    while (true) {
        if (poses.size() > 10 && sys_options_.do_async_ba) {

            orig_num_aac_poses = sys_options_.num_aac_poses;
            while (true) {
                if (poses.size() > sys_options_.min_poses_for_imu && sys_options_.use_imu_measurements) {
                    DoBundleAdjustment(aac_bundle_adjuster, true, sys_options_.num_aac_poses,
                                       false, sys_options_.do_adaptive, 1, aac_imu_residual_ids);
                }

                if ((int)sys_options_.num_aac_poses == orig_num_aac_poses || !sys_options_.do_adaptive) {
                    break;
                }
            }

            imu_cond_start_pose_id = -1;
            prev_cond_error = -1;
        }
        usleep(1000);
    }
}

void VINSystem::DoBA()
{

    if (poses.size() > sys_options_.min_poses_for_imu && sys_options_.use_imu_measurements) {
        DoBundleAdjustment(vi_bundle_adjuster, true, sys_options_.num_ba_poses, false, false,
                           0, ba_imu_residual_ids);
    } else {
        DoBundleAdjustment(bundle_adjuster, false, sys_options_.num_ba_poses,
                           false, false, 0, ba_imu_residual_ids);
    }
}

void VINSystem::BaAndStartNewLandmarks()
{
    if (!is_keyframe) {
        return;
    }

    double ba_time = sdtrack::Tic();
    if (do_bundle_adjustment) {
        DoBA();
    }
    ba_time = sdtrack::Toc(ba_time);

    if (do_start_new_landmarks) {
        tracker.StartNewLandmarks(0);
    }

    std::shared_ptr<sdtrack::TrackerPose> new_pose = poses.back();

    // Update the tracks on this new pose.
    new_pose->tracks = tracker.GetNewTracks();

    if (!do_bundle_adjustment) {
        tracker.TransformTrackTabs(tracker.t_ba());
    }



}

void VINSystem::ProcessImage(std::vector<cv::Mat>& images, double timestamp)
{
#ifdef CHECK_NANS
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() &
                           ~(_MM_MASK_INVALID | _MM_MASK_OVERFLOW |
                             _MM_MASK_DIV_ZERO));
#endif

    if (frame_count == 0) {
        start_time = sdtrack::Tic();
    }

    frame_count++;


    Sophus::SE3d guess;
    // If this is a keyframe, set it as one on the tracker.
    prev_delta_t_ba = tracker.t_ba() * prev_t_ba.inverse();

    if (is_prev_keyframe) {
        prev_t_ba = Sophus::SE3d();
    } else {
        prev_t_ba = tracker.t_ba();
    }

    // Add a pose to the poses array
    if (is_prev_keyframe) {
        std::shared_ptr<sdtrack::TrackerPose> new_pose(new sdtrack::TrackerPose);
        if (poses.size() > 0) {
            new_pose->t_wp = poses.back()->t_wp * last_t_ba.inverse();
            new_pose->v_w = poses.back()->v_w;
            new_pose->b = poses.back()->b;
        } else {
            if (imu_buffer.elements.size() > 0) {
                Eigen::Vector3t down = -imu_buffer.elements.front().a.normalized();
                VLOG(3) << "Down vector based on first imu meas: " <<
                             down.transpose() << std::endl;

                // compute path transformation
                Eigen::Vector3t forward(1.0, 0.0, 0.0);
                Eigen::Vector3t right = down.cross(forward);
                right.normalize();
                forward = right.cross(down);
                forward.normalize();

                Eigen::Matrix4t base = Eigen::Matrix4t::Identity();
                base.block<1, 3>(0, 0) = forward;
                base.block<1, 3>(1, 0) = right;
                base.block<1, 3>(2, 0) = down;
                new_pose->t_wp = Sophus::SE3t(base);
            }
            // Set the initial velocity and bias. The initial pose is initialized to
            // align the gravity plane
            new_pose->v_w.setZero();
            new_pose->b.setZero();
            // corridor
            // new_pose->b << 0.00209809 , 0.00167743, -7.46213e-05 ,
            //     0.151629 ,0.0224114, 0.826392;

            // gw_block
            // new_pose->b << 0.00288919,  0.0023673, 0.00714931 ,
            //     -0.156199,   0.258919,   0.422379;

            // gw_car_block
            // new_pose->b << 0.00217338, -0.00122939,  0.00220202,
            //     -0.175229,  -0.0731785,    0.548693;

        }
        {
            std::unique_lock<std::mutex>(aac_mutex);
            poses.push_back(new_pose);
        }
    }

    // Set the timestamp of the latest pose to this image's timestamp.
    poses.back()->time = timestamp + sys_options_.imu_time_offset;

    guess = prev_delta_t_ba * prev_t_ba;
    if(guess.translation() == Eigen::Vector3d(0,0,0) &&
            poses.size() > 1) {
        guess.translation() = Eigen::Vector3d(0, 0, 0.001);
    }

    if (sys_options_.use_imu_measurements &&
            sys_options_.use_imu_for_guess && poses.size() >= sys_options_.min_poses_for_imu) {
        std::shared_ptr<sdtrack::TrackerPose> pose1 = poses[poses.size() - 2];
        std::shared_ptr<sdtrack::TrackerPose> pose2 = poses.back();
        std::vector<ba::ImuPoseT<Scalar>> imu_poses;
        ba::PoseT<Scalar> start_pose;
        start_pose.t_wp = pose1->t_wp;
        start_pose.b = pose1->b;
        start_pose.v_w = pose1->v_w;
        start_pose.time = pose1->time;
        // Integrate the measurements since the last frame.
        std::vector<ba::ImuMeasurementT<Scalar> > meas =
                imu_buffer.GetRange(pose1->time, pose2->time);
        decltype(vi_bundle_adjuster)::ImuResidual::IntegrateResidual(
                    start_pose, meas, start_pose.b.head<3>(), start_pose.b.tail<3>(),
                    vi_bundle_adjuster.GetImuCalibration().g_vec, imu_poses);

        if (imu_poses.size() > 1) {
            // std::cerr << "Prev guess t_ab is\n" << guess.matrix3x4() << std::endl;
            ba::ImuPoseT<Scalar>& last_pose = imu_poses.back();
            //      guess.so3() = last_pose.t_wp.so3().inverse() *
            //          imu_poses.front().t_wp.so3();
            guess = last_pose.t_wp.inverse() *
                    imu_poses.front().t_wp;
            pose2->t_wp = last_pose.t_wp;
            pose2->v_w = last_pose.v_w;
            poses.back()->t_wp = pose2->t_wp;
            poses.back()->v_w = pose2->v_w;
            poses.back()->b = pose2->b;
            // std::cerr << "Imu guess t_ab is\n" << guess.matrix3x4() << std::endl;
        }
    }

    {
        std::lock_guard<std::mutex> lock(aac_mutex);

        tracker.AddImage(images, guess);
        tracker.EvaluateTrackResiduals(0, tracker.GetImagePyramid(),
                                       tracker.GetCurrentTracks());

        if (!is_manual_mode) {
            tracker.OptimizeTracks(-1, optimize_landmarks, optimize_pose);
            tracker.PruneTracks();
        }
        // Update the pose t_ab based on the result from the tracker.
        UpdateCurrentPose();
    }

    if (sys_options_.do_keyframing) {
        const double track_ratio = (double)tracker.num_successful_tracks() /
                (double)keyframe_tracks;
        const double total_trans = tracker.t_ba().translation().norm();
        const double total_rot = tracker.t_ba().so3().log().norm();

        double average_depth = 0;
        if (current_tracks == nullptr || current_tracks->size() == 0) {
            average_depth = 1;
        } else {
            for (std::shared_ptr<sdtrack::DenseTrack>& track : *current_tracks) {
                average_depth += (1.0 / track->ref_keypoint.rho);
            }
            average_depth /= current_tracks->size();
        }


        bool keyframe_condition = track_ratio < 0.7 ||
                total_trans > 0.2 || total_rot > 0.1
                /*|| tracker.num_successful_tracks() < 64*/;

        VLOG(3) << "\tRatio: " << track_ratio << " trans: " << total_trans <<
                     "av: depth: " << average_depth << " rot: " <<
                     total_rot << std::endl;

        {
            std::lock_guard<std::mutex> lock(aac_mutex);
            if (keyframe_tracks != 0) {
                if (keyframe_condition) {
                    is_keyframe = true;
                } else {
                    is_keyframe = false;
                }


                // If this is a keyframe, set it as one on the tracker.
                prev_delta_t_ba = tracker.t_ba() * prev_t_ba.inverse();

                if (is_keyframe) {
                    tracker.AddKeyframe();
                }
                is_prev_keyframe = is_keyframe;
            }
        }
    } else {
        std::lock_guard<std::mutex> lock(aac_mutex);
        tracker.AddKeyframe();
    }

    VLOG(3) << "Num successful : " << tracker.num_successful_tracks() <<
                 " keyframe tracks: " << keyframe_tracks << std::endl;

    if (!is_manual_mode) {
        BaAndStartNewLandmarks();
    }

    if (is_keyframe) {
        VLOG(3) << "KEYFRAME." << std::endl;
        keyframe_tracks = tracker.GetCurrentTracks().size();
        VLOG(3) << "New keyframe tracks: " << keyframe_tracks << std::endl;
    } else {
        VLOG(3) << "NOT KEYFRAME." << std::endl;
    }

    current_tracks = &tracker.GetCurrentTracks();

#ifdef CHECK_NANS
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() |
                           (_MM_MASK_INVALID | _MM_MASK_OVERFLOW |
                            _MM_MASK_DIV_ZERO));
#endif

    VLOG(1) << "FRAME : " << frame_count << " KEYFRAME: " << poses.size() <<
                 " FPS: " << frame_count / sdtrack::Toc(start_time) << std::endl;
}





void VINSystem::InitTracker()
{
    sys_options_.patch_size = 9;
    sdtrack::KeypointOptions keypoint_options;
    keypoint_options.gftt_feature_block_size = sys_options_.patch_size;
    keypoint_options.max_num_features = sys_options_.num_features * 2;
    keypoint_options.gftt_min_distance_between_features = 3;
    keypoint_options.gftt_absolute_strength_threshold = 0.005;
    sdtrack::TrackerOptions tracker_options;
    tracker_options.pyramid_levels = sys_options_.pyramid_levels;
    tracker_options.detector_type = sdtrack::TrackerOptions::Detector_GFTT;
    tracker_options.num_active_tracks = sys_options_.num_features;
    tracker_options.use_robust_norm_ = false;
    tracker_options.robust_norm_threshold_ = 30;
    tracker_options.patch_dim = sys_options_.patch_size;
    tracker_options.default_rho = 1.0/5.0;
    tracker_options.feature_cells = sys_options_.feature_cells;
    tracker_options.iteration_exponent = 2;
    tracker_options.center_weight = sys_options_.tracker_center_weight;
    tracker_options.dense_ncc_threshold = sys_options_.ncc_threshold;
    tracker_options.harris_score_threshold = 2e6;
    tracker_options.gn_scaling = 1.0;
    tracker.Initialize(keypoint_options, tracker_options, &rig);

}


void VINSystem::StartThreads()
{

    // Start AAC thread.
    aac_thread = std::shared_ptr<std::thread>
            (new std::thread(&VINSystem::DoAAC, this));

    // Start main trakcing thread
    tracking_thread = std::shared_ptr<std::thread>
            (new std::thread(&VINSystem::Run, this));
    //    Run();
}

bool VINSystem::GetLatestPose(sdtrack::TrackerPose* out_pose, bool integrate_imu)
{
    if(poses.size() == 0){
        // No poses created, nothing to do.
        return false;
    }
    std::lock_guard<std::mutex>lck(latest_pose_mutex_);

    bool use_imu = integrate_imu &&
            sys_options_.use_imu_measurements &&
            sys_options_.use_imu_for_guess &&
            poses.size() >= sys_options_.min_poses_for_imu;

    if(latest_pose_)
    {
        double imu_end_time = 0;
        {
            std::unique_lock<std::mutex>(imu_buffer_mutex_);
            imu_end_time = imu_buffer.end_time;
        }

        if(use_imu && latest_pose_->time < imu_end_time)
        {

            VLOG(2) << "Integrating IMU measurements to get latest pose estimate.";
            std::vector<ba::ImuPoseT<Scalar>> imu_poses;
            ba::PoseT<Scalar> start_pose;
            start_pose.t_wp = latest_pose_->t_wp;
            start_pose.b = latest_pose_->b;
            start_pose.v_w = latest_pose_->v_w;
            start_pose.time = latest_pose_->time;
            // Integrate the measurements since the last frame.
            std::vector<ba::ImuMeasurementT<Scalar> > meas =
                    imu_buffer.GetRange(latest_pose_->time, imu_end_time);
            decltype(vi_bundle_adjuster)::ImuResidual::IntegrateResidual(
                        start_pose, meas, start_pose.b.head<3>(), start_pose.b.tail<3>(),
                        vi_bundle_adjuster.GetImuCalibration().g_vec, imu_poses);

            if (imu_poses.size() > 1) {
                ba::ImuPoseT<Scalar>& last_pose = imu_poses.back();

                out_pose->t_wp = last_pose.t_wp;
                out_pose->v_w = last_pose.v_w;
                out_pose->b = latest_pose_->b;
            }else{
                LOG(ERROR) << "IMU integration returned just one pose, returning last optimized pose instead.";
                *out_pose = *latest_pose_;
            }

        }else
        {
            if(!use_imu)
                VLOG(2) << "Not using IMU, just returning latest optimized pose.";
            else
                VLOG(2) << "Pose estimate is current, no IMU integration necessary.";

            *out_pose = *latest_pose_;
        }
    }else
    {
        VLOG(3) << "Latest pose not set, probably too early.";
        return false;
    }

    return true;
}


SystemOptions& VINSystem::GetSystemOptions(){
    return sys_options_;
}




}
