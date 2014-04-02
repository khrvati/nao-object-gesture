#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ObjectTracking.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

TrackedObject::TrackedObject(const Mat image, const vector<Point> inContour){
    contour = inContour;
    Mat temp = Mat::zeros(image.size(), CV_8U);
    vector<vector<Point>> conts;
    vector<Point> points;
    conts.push_back(contour);
    drawContours(temp, conts, 0, Scalar(255), CV_FILLED);
    for (int i=0; i<temp.rows; i++){
	for (int j=0; j<temp.cols; j++){
	    if (temp.at<int>(i,j)){
		points.push_back(Point(i,j));
	    }
	}
    }
    ellipse = fitEllipse(points);
    listOfPoints = points;
    
}