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
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/ini_parser.hpp"
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


    bool usingCamera = true;
    path iniPath = "config.ini";
    ImageAcquisition* capture;
    std::string rDir = "";
    std::string imgseq = "";
    if (!exists(iniPath)){
        ConnectedCamera* camera = new ConnectedCamera(0);
        capture = camera;
    }
    else {
        boost::property_tree::ptree pt;
        boost::property_tree::ini_parser::read_ini(iniPath.string(), pt);
        rDir = pt.get<string>("Local.ImageDirectory", "");
        int localCam = pt.get<int>("Local.UseLocalCamera", 0);
        if (localCam){
            int camid = pt.get<int>("Local.Camera", -1);
            if (camid<0){
                std::cout << "Connecting to default local camera" << std::endl;
                ConnectedCamera* camera = new ConnectedCamera(0);
                capture = camera;
            }
            else {
                std::cout << "Connecting to local camera " << camid << std::endl;
                ConnectedCamera* camera = new ConnectedCamera(camid);
                capture = camera;
            }
        }
        else {
            int uImSeq = pt.get<int>("Local.UseImageSequence", 0);
            if (uImSeq){
                imgseq = pt.get<string>("Local.ImageSequence", "");
                std::cout << "Playing image sequence at " << imgseq <<  std::endl;
                usingCamera = false;
            }
            else {
                std::string ip = pt.get<string>("Remote.IP", "");
                int port = pt.get<int>("Remote.PORT", 0);
                std::cout << "Connecting to NAO at " << ip << ":" << port <<  std::endl;
                NAOCamera* camera = new NAOCamera(ip, port);
                capture = camera;
            }
        }
    }

    int hSize[] = {32,32};
    int colorCode = CV_BGR2HLS;
    vector<ProcessingElement*> pipeline;
    
    vector<double> num = {0.87};
    vector<double> den = {1, -0.13};
    ColorHistBackProject ltifilt(colorCode, hSize);
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

    path rootdir(rDir);
    path dataDir = rootdir/"Dataset";
    path gTruthDir = rootdir/"GroundTruth";
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

    vector<vector<int > > pipelineIdVector;
    vector<int> temp = {6};
    pipelineIdVector.push_back(temp);
    
    String windowname="Color Histogram Backpropagation";
    DisplayWindow window(windowname, pipeline,pipelineIdVector);

    if (!usingCamera){
        window.setImageFolder(imgseq);
        window.t->join();
    }
    else {
        Mat frame;
        while (window.running){
            capture->getImage(frame);
            window.display(frame);
        }
    }


    delete capture;
	return 0;
}
