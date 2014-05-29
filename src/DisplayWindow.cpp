#include "opencv2/highgui/highgui.hpp"


#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/ref.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread/pthread/condition_variable.hpp>
#include <boost/thread/pthread/mutex.hpp>
#include "boost/filesystem/fstream.hpp"

#include "DisplayWindow.hpp"
#include "ImageAcquisition.h"
#include "ImgProcPipeline.hpp"
#include "ObjectTracking.hpp"

#include <chrono>

namespace fs = boost::filesystem;

DisplayWindow::DisplayWindow(String name){
    dirname = "";
    windowName=name;
    dragStartL = Point(-1,-1);
    dragStartR = Point(-1,-1);
    currentPos = Point(-1,-1);
    leftDrag=false;
    rightDrag=false;
    dragging = false;
    clickTime = std::chrono::system_clock::now();
    mode=0;
    namedWindow(name);
    setMouseCallback(name, staticMouseCallback, this);
    running = true;
    t = new boost::thread(boost::ref(*this));
}

DisplayWindow::DisplayWindow(std::string name, std::vector<ProcessingElement*> prcElm, std::vector<std::vector<int > > pipelineVec){
    dirname = "";
    windowName=name;
    dragStartL = Point(-1,-1);
    dragStartR = Point(-1,-1);
    currentPos = Point(-1,-1);
    leftDrag=false;
    rightDrag=false;
    dragging = false;
    processingElements = prcElm;
    pipelineVector = pipelineVec;
    clickTime = std::chrono::system_clock::now();
    mode=0;
    namedWindow(name);
    setMouseCallback(name, staticMouseCallback, this);
    running = true;
    t = new boost::thread(boost::ref(*this));
}

void DisplayWindow::operator ()()
{
    while (true)
    {
        if (!dispImg.empty()){
            Mat endImage;
            imLock.lock();
            process(dispImg, endImage, mode);
            imLock.unlock();
            if (dragging && mode==0){
                rectangle(endImage, dragStartL, currentPos, Scalar(0,0,255));
            }
            imshow(windowName,endImage);
        }
        else {
            imshow(windowName,Mat::zeros(Size(100,100), CV_8UC3));
        }
        int c = waitKey(10);
        onKeyPress(c);

    if((char)c == 27) {break;}
    }
    running = false;
}

void DisplayWindow::display(const Mat image){
    imLock.lock();
    image.copyTo(dispImg);
    imLock.unlock();
}

void DisplayWindow::process(Mat inputImage, Mat& outputImage, int tMode)
{
    inputImage.copyTo(outputImage);
    if (tMode!=0 && pipelineVector.size()<=tMode){
        std::vector<int> pipe = pipelineVector[tMode-1];
        for (int i=0; i<pipe.size(); i++){
            if (processingElements[pipe[i]]->initialized){
                processingElements[pipe[i]]->process(outputImage, &outputImage);
            }
            else{inputImage.copyTo(outputImage); break;}
        }
    }
}


void DisplayWindow::staticMouseCallback(int event, int x, int y, int flags, void* param){
    DisplayWindow *self = static_cast<DisplayWindow*>(param);
    self->mouseCallback(event, x, y, flags, param);
}

void DisplayWindow::mouseCallback(int event, int x, int y, int flags, void* param){
    switch (event){
    case EVENT_LBUTTONDOWN:
        dragStartL = Point(x,y);
        leftDrag = true;
        clickTime = std::chrono::system_clock::now();
        break;
    case EVENT_RBUTTONDOWN:
        dragStartR = Point(x,y);
        rightDrag = true;
        clickTime = std::chrono::system_clock::now();
        break;
    case EVENT_LBUTTONUP:
        if (dragging) {onDragStop();} else {onLeftClick();}
        leftDrag = false;
        dragging = false;
        break;
    case EVENT_RBUTTONUP:
        if (dragging) {onDragStop();} else {onRightClick();}
        rightDrag = false;
        dragging = false;
        break;
    case EVENT_MOUSEMOVE:
        currentPos = Point(x,y);
        auto diff = std::chrono::system_clock::now()-clickTime;
        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
        std::chrono::milliseconds oneClick(250);
        if ((leftDrag || rightDrag) && ms > oneClick) {dragging = true;}
        break;
    }
}

void DisplayWindow::onLeftClick(){
}

void DisplayWindow::onRightClick(){
}

void DisplayWindow:: onDragStop(){
    if (processingElements.size()>0 && leftDrag){
        try{
            int x=min(dragStartL.x,currentPos.x);
            int y=min(dragStartL.y,currentPos.y);
            int width=abs(dragStartL.x-currentPos.x);
            int height=abs(dragStartL.y-currentPos.y);
            Rect imageROI = Rect(x,y,width,height);
            imLock.lock();
            Mat subimage(dispImg, imageROI);
            for (int i=0; i<processingElements.size(); i++){
                if (processingElements[i]->name.compare("ColorHistBackProject")==0){
                    ColorHistBackProject *temp = static_cast<ColorHistBackProject*>(processingElements[i]);
                    temp->histFromImage(subimage);
                }
                if (processingElements[i]->name.compare("BayesColorHistBackProject")==0){
                    BayesColorHistBackProject *temp = static_cast<BayesColorHistBackProject*>(processingElements[i]);
                    temp->histFromImage(subimage);
                }
                if (processingElements[i]->name.compare("GMMColorHistBackProject")==0){
                    GMMColorHistBackProject *temp = static_cast<GMMColorHistBackProject*>(processingElements[i]);
                    temp->histFromImage(subimage);
                }

                if (processingElements[i]->name.compare("ObjectTracker")==0){
                    ObjectTracker *temp = static_cast<ObjectTracker*>(processingElements[i]);
                    Mat mask(Mat::zeros(dispImg.size(), CV_8U));
                    rectangle(mask, imageROI, Scalar(255), -1);
                    vector<Mat> imvec;
                    vector<Mat> maskvec;
                    imvec.push_back(dispImg);
                    maskvec.push_back(mask);
                    temp->addObjectKind(imvec, maskvec);
                }
            }
            imLock.unlock();
        } catch(Exception e){
            std::cout << e.msg << std::endl;
        }
    }
}

bool DisplayWindow::nextImage()
{
    boost::filesystem::directory_iterator end_itr;
    if (itr==end_itr){
        itr = boost::filesystem::directory_iterator(dirname);
    }
    filename = itr->path();
    while (is_directory(filename) && itr!=end_itr){
        itr++;
        if (itr!=end_itr){
            filename = itr->path();
        }
    }
    if (itr==end_itr){
        itr = boost::filesystem::directory_iterator(dirname);
        filename = itr->path();
    }
    while (is_directory(filename) && itr!=end_itr){
        itr++;
        if (itr!=end_itr){
            filename = itr->path();
        }
    }
    if (itr==end_itr){
        return false;
    }
    else {
        return true;
    }
}

void DisplayWindow::onKeyPress(int key){
    if (key!=-1){
        char ckey = (char)key;
        key = (int)ckey;
        std::cout << "Keypress: " << (int)ckey << std::endl;
        if (ckey==10){
            fs::path full_path;
            time_t rawtime;
            struct tm * timeinfo;
            char buffer [80];
            time (&rawtime);
            timeinfo = localtime (&rawtime);
            strftime(buffer, 80, "%F_%H-%M-%S.jpg", timeinfo);

            full_path = fs::system_complete(fs::path(buffer));

            vector<int> compression_params;
            compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
            compression_params.push_back(100);

            std::string spath = full_path.c_str();
            imLock.lock();
            imwrite(spath, dispImg, compression_params);
            imLock.unlock();
        }
        if (key>47 && key<58){
            int numkey = key-48;
            if (numkey<=pipelineVector.size()){
                mode = numkey;
            }
        }
        if (key==81 && !dirname.empty()){
            if (nextImage()){
                Mat im = imread(filename.string());
                display(im);
                itr++;
            }
        }
        if (key==83 && !dirname.empty()){
            if (nextImage()){
                Mat im = imread(filename.string());
                display(im);
                itr++;
            }
        }
    }
}

void DisplayWindow::setImageFolder(string dir)
{
    dirname = dir;
    if (nextImage()){
        std::cout << "Init" <<std::endl;
        Mat im = imread(filename.string());
        display(im);
        itr++;
    }
}
