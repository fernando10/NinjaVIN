#include <iostream>
#include <HAL/Camera/CameraDevice.h>
#include <HAL/IMU/IMUDevice.h>
#include <HAL/Messages/Matrix.h>
#include <stdlib.h>
#include <Eigen/Core>
#include <okvis/VioParametersReader.hpp>
#include <okvis/ThreadedKFVio.hpp>

// For pose viewer
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <memory>
#include <functional>
#include <atomic>

using namespace std;


/*-----------COMMAND LINE FLAGS-----------------------------------------------*/
DEFINE_string(config, "", "path to yaml configuration file");
DEFINE_string(cam, "", "camera device");
DEFINE_string(imu, "", "imu device");
DEFINE_double(start_delay, 0, "delay in seconds to start getting data");
/*----------------------------------------------------------------------------*/

hal::Camera camera_device;
hal::IMU imu_device;
std::shared_ptr<hal::Image> camera_img;
int image_width;
int image_height;
double image_timestamp;
std::mutex imu_buffer_mutex;

bool LoadCameras();
bool LoadIMU();

struct IMUMeasurement
{
    okvis::Time timestamp;
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
};

std::deque<IMUMeasurement> imu_measurements;


class PoseViewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    constexpr static const double imageSize = 500.0;
    PoseViewer()
    {
        cv::namedWindow("OKVIS Top View");
        _image.create(imageSize, imageSize, CV_8UC3);
        drawing_ = false;
        showing_ = false;
    }
    // this we can register as a callback
    void publishFullStateAsCallback(
            const okvis::Time & /*t*/, const okvis::kinematics::Transformation & T_WS,
            const Eigen::Matrix<double, 9, 1> & speedAndBiases,
            const Eigen::Matrix<double, 3, 1> & /*omega_S*/)
    {

        // just append the path
        Eigen::Vector3d r = T_WS.r();
        Eigen::Matrix3d C = T_WS.C();
        _path.push_back(cv::Point2d(r[0], r[1]));
        _heights.push_back(r[2]);
        // maintain scaling
        if (r[0] - _frameScale < _min_x)
            _min_x = r[0] - _frameScale;
        if (r[1] - _frameScale < _min_y)
            _min_y = r[1] - _frameScale;
        if (r[2] < _min_z)
            _min_z = r[2];
        if (r[0] + _frameScale > _max_x)
            _max_x = r[0] + _frameScale;
        if (r[1] + _frameScale > _max_y)
            _max_y = r[1] + _frameScale;
        if (r[2] > _max_z)
            _max_z = r[2];
        _scale = std::min(imageSize / (_max_x - _min_x), imageSize / (_max_y - _min_y));

        // draw it
        while (showing_) {
        }
        drawing_ = true;
        // erase
        _image.setTo(cv::Scalar(10, 10, 10));
        drawPath();
        // draw axes
        Eigen::Vector3d e_x = C.col(0);
        Eigen::Vector3d e_y = C.col(1);
        Eigen::Vector3d e_z = C.col(2);
        cv::line(
                    _image,
                    convertToImageCoordinates(_path.back()),
                    convertToImageCoordinates(
                        _path.back() + cv::Point2d(e_x[0], e_x[1]) * _frameScale),
                cv::Scalar(0, 0, 255), 1, CV_AA);
        cv::line(
                    _image,
                    convertToImageCoordinates(_path.back()),
                    convertToImageCoordinates(
                        _path.back() + cv::Point2d(e_y[0], e_y[1]) * _frameScale),
                cv::Scalar(0, 255, 0), 1, CV_AA);
        cv::line(
                    _image,
                    convertToImageCoordinates(_path.back()),
                    convertToImageCoordinates(
                        _path.back() + cv::Point2d(e_z[0], e_z[1]) * _frameScale),
                cv::Scalar(255, 0, 0), 1, CV_AA);

        // some text:
        std::stringstream postext;
        postext << "position = [" << r[0] << ", " << r[1] << ", " << r[2] << "]";
        cv::putText(_image, postext.str(), cv::Point(15,15),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);
        std::stringstream veltext;
        veltext << "velocity = [" << speedAndBiases[0] << ", " << speedAndBiases[1] << ", " << speedAndBiases[2] << "]";
        cv::putText(_image, veltext.str(), cv::Point(15,35),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1);

        drawing_ = false; // notify
    }
    void display()
    {
        while (drawing_) {
        }
        showing_ = true;
        cv::imshow("NinjaCar Top View", _image);
        showing_ = false;
        cv::waitKey(1);
    }
private:
    cv::Point2d convertToImageCoordinates(const cv::Point2d & pointInMeters) const
    {
        cv::Point2d pt = (pointInMeters - cv::Point2d(_min_x, _min_y)) * _scale;
        return cv::Point2d(pt.x, imageSize - pt.y); // reverse y for more intuitive top-down plot
    }
    void drawPath()
    {
        for (size_t i = 0; i + 1 < _path.size(); ) {
            cv::Point2d p0 = convertToImageCoordinates(_path[i]);
            cv::Point2d p1 = convertToImageCoordinates(_path[i + 1]);
            cv::Point2d diff = p1-p0;
            if(diff.dot(diff)<2.0){
                _path.erase(_path.begin() + i + 1);  // clean short segment
                _heights.erase(_heights.begin() + i + 1);
                continue;
            }
            double rel_height = (_heights[i] - _min_z + _heights[i + 1] - _min_z)
                    * 0.5 / (_max_z - _min_z);
            cv::line(
                        _image,
                        p0,
                        p1,
                        rel_height * cv::Scalar(255, 0, 0)
                        + (1.0 - rel_height) * cv::Scalar(0, 0, 255),
                        1, CV_AA);
            i++;
        }
    }
    cv::Mat _image;
    std::vector<cv::Point2d> _path;
    std::vector<double> _heights;
    double _scale = 1.0;
    double _min_x = -0.5;
    double _min_y = -0.5;
    double _min_z = -0.5;
    double _max_x = 0.5;
    double _max_y = 0.5;
    double _max_z = 0.5;
    const double _frameScale = 0.2;  // [m]
    std::atomic_bool drawing_;
    std::atomic_bool showing_;
};


int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    if(!FLAGS_cam.empty()){
        LOG(INFO)  << "Initializing camera...";
        LoadCameras();
    }else{
        LOG(FATAL) << "Camera URL empty...cannot continue.";
    }

    if (!FLAGS_imu.empty()){
        LOG(INFO) << "Initializing IMU...";
        LoadIMU();
    }

    okvis::VioParametersReader vio_parameters_reader(FLAGS_config);
    okvis::VioParameters parameters;
    vio_parameters_reader.getParameters(parameters);

    okvis::ThreadedKFVio slam_system(parameters);

    okvis::Duration deltaT(FLAGS_start_delay);


    PoseViewer poseViewer;
    slam_system.setFullStateCallback(
                std::bind(&PoseViewer::publishFullStateAsCallback, &poseViewer,
                          std::placeholders::_1, std::placeholders::_2,
                          std::placeholders::_3, std::placeholders::_4));

    slam_system.setBlocking(true);

    //LOG(INFO)parameters.nCameraSystem.T_SC(0)->r;

    bool capture_success = false;
    std::shared_ptr<hal::ImageArray> images = hal::ImageArray::Create();

    cv::Mat im;
    okvis::Time start(0.0);
    okvis::Time prev_img_time(0.0);

    while(true)
    {
        slam_system.display();
        poseViewer.display();

        // add images
        okvis::Time t;
        capture_success = camera_device.Capture(*images);

        if(capture_success)
        {
            camera_img = images->at(0);
            image_width = camera_img->Width();
            image_height = camera_img->Height();
            image_timestamp = images->Ref().device_time();

            std::vector<cv::Mat> cvmat_images;
            for (int ii = 0; ii < images->Size() ; ++ii) {
                cvmat_images.push_back(images->at(ii)->Mat());
            }

            im = cvmat_images.at(0);

            if(im.empty())
            {
                LOG(ERROR) << "Failed to load image.";
                return 1;
            }

            t = okvis::Time(image_timestamp);
            if (start == okvis::Time(0.0)) {
                start = t;
                prev_img_time = t;
                VLOG(2) << "Start time: " << start.toSec();
            }


            // get all IMU measurements till then
            okvis::Time t_imu = start;
            do {
                IMUMeasurement imu_meas;
                {
                    std::lock_guard<std::mutex>lock(imu_buffer_mutex);

                    imu_meas = imu_measurements.front();
                    imu_measurements.pop_front();
                }

                t_imu = imu_meas.timestamp;

                if (t_imu - start + okvis::Duration(1.0) > deltaT) {
                    VLOG(3) << "Adding IMU meas. with time: " << t_imu.toSec();
                    slam_system.addImuMeasurement(t_imu, imu_meas.accel,
                                                  imu_meas.gyro);
                }

            }
            while (t_imu <= t);

            // add the image to the frontend for (blocking) processing
            if (t - start > deltaT) {
                VLOG(3) << "Adding Image with time: " << t.toSec();
                slam_system.addImage(t, 0, im);
                prev_img_time = t;
            }

        }

    }

}



void ImuCallback(const hal::ImuMsg& ref) {
    const double timestamp = ref.device_time();
    IMUMeasurement meas;
    Eigen::VectorXd a;
    Eigen::VectorXd g;
    hal::ReadVector(ref.accel(), &a);
    hal::ReadVector(ref.gyro(), &g);
    meas.accel = a;
    meas.gyro = g;
    meas.timestamp.fromSec(timestamp);

    VLOG(4) << "Got IMU meas from file, ts: " << timestamp;

    std::lock_guard<std::mutex>lock(imu_buffer_mutex);
    if(imu_measurements.size() &&
            meas.timestamp < imu_measurements.back().timestamp){
        LOG(ERROR) << "Received IMU measurement from the past....not good...";
    }else{
        imu_measurements.emplace_back(meas);
    }
}

bool LoadIMU()
{
    try{
        imu_device = hal::IMU(FLAGS_imu);
        imu_device.RegisterIMUDataCallback(&ImuCallback);
    }
    catch(hal::DeviceException& e) {
        LOG(ERROR) << "Error loading IMU device: " << e.what();
        return false;
    }

    return true;
}

bool LoadCameras()
{

    try {
        camera_device = hal::Camera(hal::Uri(FLAGS_cam));
    }
    catch (hal::DeviceException& e) {
        LOG(ERROR) << "Error loading camera device: " << e.what();
        return false;
    }

    return true;
}


