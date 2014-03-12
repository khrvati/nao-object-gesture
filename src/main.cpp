/*
 * objectTracking.cpp
 * 
 * Copyright 2014 Kruno Hrvatinic <kruno.hrvatinic@fer.hr>
 * 
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ColorSegmenter.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/* Function Headers */
void RectangleMouseCallback(int event, int x, int y, int, void*);

/* Global Variables */
String window_name = "Capture - Face detection";
Point pt1(-1,-1), pt2(-1,-1);
bool drawRectangle = false;
bool updateHist = false;
bool showSegmented = false;
Rect imageROI;
ColorSegmenter seg;

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
	
	namedWindow(window_name);
	setMouseCallback(window_name, RectangleMouseCallback, 0);
	
    while (capture.read(frame))
    {
        if(frame.empty())
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
		if (updateHist)
		{try{ 
			Mat subimage(frame, imageROI);
			seg.histFromImage(subimage);
			imshow("otherwindow",subimage);
			} catch (exception& e) {}
			updateHist=false;
		}
        if (drawRectangle)
        {
			rectangle(frame, pt1, pt2, Scalar(0,0,255));
		}
		if (showSegmented) {
			Mat segframe;
			seg.backPropHist(frame, &segframe);
			imshow(window_name, segframe);
		}
		else { imshow( window_name, frame );}
        int c = waitKey(10);
        if((char)c == 27) {break;} // escape
    }
	
	return 0;
}

void RectangleMouseCallback(int event, int x, int y, int, void*)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		pt1.x=x;
		pt1.y=y;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		pt2.x=x;
		pt2.y=y;
	}
	else if (event == EVENT_LBUTTONUP)
	{
		int x=min(pt1.x,pt2.x);
		int y=min(pt1.y,pt2.y);
		int width=abs(pt1.x-pt2.x);
		int height=abs(pt1.y-pt2.y);
		imageROI = Rect(x,y,width,height);
		updateHist=true;
		pt1.x=-1;
		pt1.y=-1;
		pt2.x=-1;
		pt2.y=-1;
	}
	if (event == EVENT_RBUTTONDOWN)
	{
	    showSegmented=showSegmented?false:true;
	}
	if (pt1.x!=-1 && pt2.x!=-1) { drawRectangle=true; }	
	else { drawRectangle=false;}
}
