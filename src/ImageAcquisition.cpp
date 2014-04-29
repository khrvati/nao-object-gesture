#include "ImageAcquisition.h"
#include "opencv2/highgui/highgui.hpp"
#include <string>

#include <alproxies/alvideodeviceproxy.h>
#include <alvision/alimage.h>
#include <alvalue/alvalue.h>
#include <alvision/alvisiondefinitions.h>

using namespace cv;
using namespace AL;

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
    clientName = camproxy.subscribeCamera("ObjectGestureRemote", 0, kVGA, kBGRColorSpace, 30);
}


NAOCamera::~NAOCamera(){
    camproxy.unsubscribe(clientName);
}

bool NAOCamera::getImage(cv::Mat &outputImage){
    try{
        ALValue img = camproxy.getImageRemote(clientName);
        Mat imgHeader = Mat(Size(640, 480), CV_8UC3,  (void*)img[6].GetBinary());
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

