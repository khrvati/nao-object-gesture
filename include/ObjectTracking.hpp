#ifndef OBJECTTRACKING
#define OBJECTTRACKING

#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

class TrackedObject{
    protected:
	vector<Point> contour;
	vector<Point> listOfPoints;
	Mat histogram;
	RotatedRect ellipse;
    public:
	TrackedObject(const Mat image, const vector<Point> inContour);
  
};


#endif