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

/*! An abstract class used as the base class for all image acquisition purposes*/
class ImageAcquisition{
    public:
        /*! Gets image from source.
          * \param outputImage Matrix to store output into
          */
        virtual bool getImage(cv::Mat& outputImage) = 0;
};


/*! Class used to acquire images from a connected 2D camera by identified by device id.*/
class ConnectedCamera : public ImageAcquisition{
    private:
        /*! OpenCV interface with connected camera.*/
        cv::VideoCapture capture;
    public:
        /*! Standard constructor.
          * If camera is unavailable, an exception is raised.
          * \param id Device id of camera to connect to.
          */
        ConnectedCamera(int id);
        /*! Gets image from connected camera.
          * \param outputImage Matrix to store output into
          */
        bool getImage(cv::Mat &outputImage);
};

/*! Class used to acquire images from NAO connected over network.*/
class NAOCamera : public ImageAcquisition{
    private:
        /*! Proxy to ALVideoDevice*/
        AL::ALVideoDeviceProxy camproxy;
        /*! Subscriber name*/
        std::string clientName;
    public:
        /*! Standard constructor.
          * \param IP IP of NAO to connect to.
          * \param port Port over which to connect
          */
        NAOCamera(const std::string IP, int port);
        /*! Destructor.
          * Terminates the NAO connection and unsubscribes from the video stream.
          */
        ~NAOCamera();
        /*! Gets image from NAO
          * \param outputImage Matrix to store output into
          */
        bool getImage(cv::Mat &outputImage);
};
#endif
