/*
 * objectTracking.cpp
 * 
 * Copyright 2014 Kruno Hrvatinic <kruno.hrvatinic@fer.hr>
 * 
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ImgProcPipeline.hpp"
#include "DisplayWindow.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(void)
{
    VideoCapture capture;
    Mat frame;
    capture.open( -1 );
    if (!capture.isOpened()) 
    {
	printf("--(!)Error opening video capture\n");
	return -1; 
    }
    
    int hSize[] = {32,32};
    vector<ProcessingElement*> pipeline;
    
    ColorHistBackProject seg(CV_BGR2HSV, hSize);
    ProcessingElement *generalPtr = static_cast<ProcessingElement*>(&seg);
    pipeline.push_back(generalPtr);
    
    BayesColorHistBackProject bayesSeg(CV_BGR2HSV, hSize);
    generalPtr = static_cast<ProcessingElement*>(&bayesSeg);
    pipeline.push_back(generalPtr);
    
    int hSize2[] = {128,128};
    GMMColorHistBackProject GMMSeg(CV_BGR2HSV, hSize2);
    generalPtr = static_cast<ProcessingElement*>(&GMMSeg);
    pipeline.push_back(generalPtr);
    
    SimpleThresholder thresh(0.4);
    generalPtr = static_cast<ProcessingElement*>(&thresh);
    pipeline.push_back(generalPtr);
    
    vector<vector<int>> pipelineIdVector;
    vector<int> temp = {0};
    pipelineIdVector.push_back(temp);
    temp = {1};
    pipelineIdVector.push_back(temp);
    temp = {2};
    pipelineIdVector.push_back(temp);
    
    String windowname="Color Histogram Backpropagation";
    
    DisplayWindow window(windowname, pipeline,pipelineIdVector);
    
    while (capture.read(frame))
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