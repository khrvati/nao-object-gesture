#include "ImageAcquisition.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>

#include <alproxies/alvideodeviceproxy.h>
#include <alvision/alimage.h>
#include <alvalue/alvalue.h>
#include <alvision/alvisiondefinitions.h>

#include <boost/filesystem.hpp>
#include "boost/filesystem/fstream.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include <boost/thread/thread_time.hpp>

#include <iterator>

using namespace cv;
using namespace AL;
using namespace boost::filesystem;

ConnectedCamera::ConnectedCamera(int id){
    capture.open(id);
    if (!capture.isOpened())
    {
        printf("--(!)Error opening video capture\n");
    }
}

bool ConnectedCamera::getImage(cv::Mat &outputImage){
    if (!capture.isOpened()){return false;}
    capture.read(outputImage);
}


NAOCamera::NAOCamera(const std::string IP, int port) : camproxy(IP, port){
    clientName = camproxy.subscribeCamera("ObjectGestureRemote", 0, kQVGA, kBGRColorSpace, 20);
    /*camproxy.setParam(3, 40);
    camproxy.setParam(11, 1);
    camproxy.setParam(22, 2);
    camproxy.setParam(12, 0);
    camproxy.setParam(33, -36);*/
}


NAOCamera::~NAOCamera(){
    camproxy.unsubscribe(clientName);
    std::cout << "Disconnecting from NAO camera" << std::endl;
}

bool NAOCamera::getImage(cv::Mat &outputImage){
    try{
        ALValue img = camproxy.getImageRemote(clientName);
        Mat imgHeader = Mat(Size(320, 240), CV_8UC3,  (void*)img[6].GetBinary());
        //imgHeader.data = (uchar*) img[6].GetBinary();
        imgHeader.copyTo(outputImage);
        camproxy.releaseImage(clientName);

    }
    catch (std::exception& e){
        std::cout << e.what() << std::endl;
        return false;
    }
    return true;
}
