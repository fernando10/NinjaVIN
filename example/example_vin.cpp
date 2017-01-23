#include <vinsystem/VINSystem.hpp>

#ifdef HAVE_SLAMViewer
#include <slamviewer/SLAMViewer.hpp>
#endif

using namespace compass;

#ifdef HAVE_SLAMViewer
using namespace slamviewer;
#endif

int main(int argc, char **argv) {

    // Enable decent logging.
    google::InitGoogleLogging(argv[0]);

    FLAGS_v = 1;

#ifdef HAVE_SLAMViewer
    // Create a GUI
    SLAMViewerOptions viewer_options;
    viewer_options.window_name = "NinjaCar - Localizer";

    SLAMViewer* viewer = new SLAMViewer(viewer_options);

    ViewerPath& path = viewer->CreatePath("car_path", Colors::GREEN);

#endif

    if (argc < 2) {
        LOG(ERROR)<<
                     "Usage: ./" << argv[0] << " configuration-yaml-file";
        return -1;
    }

    LOG(INFO) << "Initializing SLAM System...";

    // Start SLAM system, we can ask for poses at any time now.
    VINSystem SLAMSystem(argv[1]);

    sdtrack::TrackerPose latest_pose;
    while(1)
    {

        std::chrono::steady_clock::time_point t1 =
                std::chrono::steady_clock::now();

        // Copy the latest pose.
        if( !SLAMSystem.GetLatestPose(&latest_pose)){
            // Failed to get pose...normal at system initialization
            usleep(1000);
            continue;
        }

        std::chrono::steady_clock::time_point t2 =
                std::chrono::steady_clock::now();

        double ttrack=
                std::chrono::duration_cast<std::chrono::duration<double> >
                (t2 - t1).count();

//        LOG(INFO) << "Got pose->  trans: ["
//                  << latest_pose.t_wp.translation().transpose() << "]" <<
//                     " | speed: [" <<
//                     latest_pose.v_w.transpose()
//                  << "] in: " << ttrack << "s";

#ifdef HAVE_SLAMViewer
        // Render pose
        {
            std::unique_lock<std::mutex>lck(viewer->viewer_mutex);
            path.poses->push_back(latest_pose.t_wp);
        }
#endif


        // Wait for a bit... (20Hz)
        usleep(50000);
    }

    return 0;


}

