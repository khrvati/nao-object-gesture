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
    
    int hSize[] = {64,64};
    ColorSegmenter seg(CV_BGR2YUV, hSize);
    ProcessingElement *generalPtr = static_cast<ProcessingElement*>(&seg);
    
    vector<ProcessingElement*> pipeline;
    pipeline.push_back(generalPtr);
    String windowname="Color Histogram Backpropagation";
    
    DisplayWindow window(windowname, pipeline);
    
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