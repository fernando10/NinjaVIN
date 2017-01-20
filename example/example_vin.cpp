#include <vinsystem/VINSystem.hpp>

using namespace compass;

int main(int argc, char **argv) {

    // Enable decent logging.
    google::InitGoogleLogging(argv[0]);

    if (argc != 2) {
      LOG(ERROR)<<
      "Usage: ./" << argv[0] << " configuration-yaml-file";
      return -1;
    }

    LOG(INFO) << "Initializing SLAM System...";

    // Start SLAM system, we can ask for poses at any time now.
    // It's advisable to wait a bit until the system has initialized the
    // pose estimates.
    VINSystem SLAMSystem(argv[1]);

    sdtrack::TrackerPose latest_pose;
    while(1)
    {
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();

        // Copy the latest pose.
        SLAMSystem.GetLatestPose(&latest_pose);

        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();

        double ttrack=
            std::chrono::duration_cast<std::chrono::duration<double> >
            (t2 - t1).count();

        LOG(INFO) << "Got pose->  trans: ["
                  << latest_pose.t_wp.translation().transpose() << "]" <<
                     " | rot: [" <<
                     latest_pose.t_wp.rotationMatrix().eulerAngles(0,1,2).transpose()
                  << "] in: " << ttrack << "s";

        // Wait for a bit...
        usleep(1000);
    }

    return 0;


}

