/*
 * objectTracking.cpp
 * 
 * Copyright 2014 Kruno Hrvatinic <kruno.hrvatinic@fer.hr>
 * 
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "boost/filesystem/path.hpp"
#include "ImgProcPipeline.hpp"
#include "DisplayWindow.hpp"
#include "ObjectTracking.hpp"
#include "ImageAcquisition.h"
#include "NAOObjectGesture.h"

#include <iterator>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace boost::filesystem;

int main(void)
{
    ConnectedCamera camera(1);
    ImageAcquisition* capture = &camera;
    Mat frame;
    
    int hSize[] = {32,32};
    int colorCode = CV_BGR2HLS;
    vector<ProcessingElement*> pipeline;
    
    vector<double> num = {0.87};
    vector<double> den = {1, -0.13};
    LTIFilter ltifilt(num, den, 1/30.0);
    ProcessingElement *generalPtr = static_cast<ProcessingElement*>(&ltifilt);
    pipeline.push_back(generalPtr);
    
    ColorHistBackProject seg(colorCode, hSize);
    generalPtr = static_cast<ProcessingElement*>(&seg);
    pipeline.push_back(generalPtr);
    
    BayesColorHistBackProject bayesSeg(colorCode, hSize);
    generalPtr = static_cast<ProcessingElement*>(&bayesSeg);
    pipeline.push_back(generalPtr);
    
    int hSize2[] = {128, 128};
    GMMColorHistBackProject GMMSeg(colorCode, hSize2);
    generalPtr = static_cast<ProcessingElement*>(&GMMSeg);
    pipeline.push_back(generalPtr);
    
    //4
    SimpleThresholder thresh(0.2);
    generalPtr = static_cast<ProcessingElement*>(&thresh);
    pipeline.push_back(generalPtr);
    
    SimpleBlobDetect sbd;
    generalPtr = static_cast<ProcessingElement*>(&sbd);
    pipeline.push_back(generalPtr);
    
    ObjectTracker objtrack;
    generalPtr = static_cast<ProcessingElement*>(&objtrack);
    pipeline.push_back(generalPtr);

    path dataDir("/home/kruno/trainingData/Dataset");
    path gTruthDir("/home/kruno/trainingData/GroundTruth");
    vector<Mat> images;
    vector<Mat> masks;
    if (exists(dataDir) && exists(gTruthDir) && is_directory(dataDir) && is_directory(gTruthDir)){
        directory_iterator end_itr;
        for(directory_iterator itr(dataDir); itr!=end_itr; ++itr){
            path filename = itr->path().stem();
            for(directory_iterator itr2(gTruthDir); itr2!=end_itr; ++itr2){
                path gTruthName = itr2->path().stem();
                if(filename==gTruthName){
                    string impath = itr->path().string();
                    string maskpath = itr2->path().string();
                    Mat img(imread(impath));
                    Mat mask(imread(maskpath,0));
                    images.push_back(img);
                    masks.push_back(mask);
                }
            }
        }
        objtrack.addObjectKind(images, masks);
        std::cout << "Loaded " << images.size() << " skin images" << std::endl;
    }
    images.clear();
    masks.clear();

    OpticalFlow optFlow;
    generalPtr = static_cast<ProcessingElement*>(&optFlow);
    pipeline.push_back(generalPtr);

    BGSubtractor bgs;
    generalPtr = static_cast<ProcessingElement*>(&bgs);
    pipeline.push_back(generalPtr);

    vector<vector<int>> pipelineIdVector;
    vector<int> temp = {6};
    pipelineIdVector.push_back(temp);
    //temp = {0,2,4,5};
    temp = {8};
    pipelineIdVector.push_back(temp);
    temp = {7};
    pipelineIdVector.push_back(temp);
    
    String windowname="Color Histogram Backpropagation";
    
    DisplayWindow window(windowname, pipeline,pipelineIdVector);
    
    while (capture->getImage(frame))
    {
        if(frame.empty())
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        
        window.display(frame);
        
        int c = waitKey(10);
	if((char)c == 27) {break;} // escape
    }
	
	return 0;
}
