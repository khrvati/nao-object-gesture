#ifndef IMAGEACQUISITION
#define IMAGEACQUISITION

#include "opencv2/highgui/highgui.hpp"
#include <string>

#include <alproxies/alvideodeviceproxy.h>
#include <boost/filesystem.hpp>
#include "boost/filesystem/fstream.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

#include <iterator>

namespace AL
{
  class ALBroker;
}


class ImageAcquisition{
    public:
        virtual bool getImage(cv::Mat& outputImage) = 0;
};


class ConnectedCamera : public ImageAcquisition{
    private:
        cv::VideoCapture capture;
    public:
        ConnectedCamera(int id);
        bool getImage(cv::Mat &outputImage);
};


class NAOCamera : public ImageAcquisition{
    private:
        AL::ALVideoDeviceProxy camproxy;
        std::string clientName;
    public:
        NAOCamera(const std::string IP, int port);
        ~NAOCamera();
        bool getImage(cv::Mat &outputImage);
};

class ImgSequence : public ImageAcquisition{
private:
    std::string dirname;
    boost::filesystem::directory_iterator itr;
    boost::posix_time::ptime lastimg;

public:
    ImgSequence(std::string dn);
    bool getImage(cv::Mat &outputImage);
};

#endif
