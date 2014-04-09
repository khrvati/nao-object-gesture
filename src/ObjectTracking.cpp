#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "ObjectTracking.hpp"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include <cmath>
#include <chrono>
#include <ctime>

using namespace cv;
using namespace std;

namespace fs = boost::filesystem;

LogManager::LogManager(){
    fs::path full_path;
    do{
      time_t rawtime;
      struct tm * timeinfo;
      char buffer [80];
      time (&rawtime);
      timeinfo = localtime (&rawtime);
      strftime(buffer, 80, "%F_%H-%M-%S", timeinfo);
      
      full_path = fs::system_complete(fs::path(buffer));
    } while (fs::exists(full_path));
    
    fs::create_directory(full_path);
    rootDir = full_path;
    nextId = 0;
}

int LogManager::getId(){
    boost::filesystem::path filePath;
    filePath = rootDir;
    int id = nextId++;
    std::string filename = "object_";
    filename += id;
    filename += ".csv";
    filePath /= filename;
    logFilePaths.push_back(filePath);
    return nextId++;
}

TrackedObject::TrackedObject(){
    tracked = false;
}

TrackedObject::TrackedObject(const Mat image, const vector<Point> inContour){
    tracked = true;
    contour = inContour;
    imageSize = image.size();
    ellipse = fitEllipse(contour);
    area = contourArea(contour);
    occluded = false;
    kind = -1;
    id = -1;
}

vector<Point> TrackedObject::actualPoints(){
    Mat temp = Mat::zeros(imageSize, CV_8U);
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
    listOfPoints = points;
    return points;
}
  
RotatedRect TrackedObject::useCamShift(const Mat probImage){
    Rect box = ellipse.boundingRect();
    return CamShift(probImage, box, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
}
  
double TrackedObject::compare(TrackedObject other){
    return matchShapes(contour, other.contour, CV_CONTOURS_MATCH_I1, 0 );
}




ObjectTracker::ObjectTracker(): filterLTI({0.87}, {1, -0.13}, 1/30.0){
    vector<double> num = {0.87};
    vector<double> den = {1, -0.13};
    filterLTI = LTIFilter(num, den, 1/30.0);
    frameNumber = 0;
    newKindAdded = false;
}

void ObjectTracker::preprocess(const Mat image, Mat* outputImage, Mat* mask){
    Mat procimg;
    bilateralFilter(image, procimg, 5, 80, 60);
    cvtColor(procimg, procimg, CV_BGR2HLS);
    Scalar lowRange = Scalar(0,40,10);
    Scalar highRange = Scalar(255,220,255);
    inRange(procimg, lowRange, highRange, *mask);
    procimg.convertTo(*outputImage, CV_32F);
}

void ObjectTracker::addObjectKind(const Mat image){
    Mat procimg;
    Mat mask;
    preprocess(image, &procimg, &mask);
    
    int channels[2] = {0,1};
    float c1range[2] = {0,180};
    float c2range[2] = {0,256};
    int histSize[2] = {128,128};
    Histogram objHist(channels, histSize, c1range, c2range);
    objectKinds.push_back(objHist);
    objectKinds.back().fromImage(procimg, mask);
    objectKinds.back().makeGMM(3,4,0.01);
    
    newKindAdded = true;
}

void ObjectTracker::process(const Mat inputImage, Mat* outputImage){
    int reacquireEvery = 3; //perform reacquisition every N frames processed
    Mat procimg;
    Mat mask;
    filterLTI.process(inputImage, &procimg);
    preprocess(procimg, &procimg, &mask);
    
    //first, get the general image histogram and use it to get a normalized probability image of the input image
    int channels[2] = {0,1};
    float c1range[2] = {0,180};
    float c2range[2] = {0,256};
    int histSize[2] = {32,32};
    Histogram imgHist(channels, histSize, c1range, c2range);
    imgHist.fromImage(procimg, mask);
    Mat aprioriColor;
    imgHist.backPropagate(procimg, &aprioriColor);
    medianBlur(aprioriColor, aprioriColor, 5);
    
    //then, apply bayesian histogram backpropagation for each object kind to create K probability images, one for each kind
    double histMax = 0;
    double histMin = 0;
    vector<Mat> probImages;
    for (int i=0; i<objectKinds.size(); i++){
	Mat objProb;
	objectKinds[i].backPropagate(procimg, &objProb);
	medianBlur(objProb, objProb, 5);
	objProb = objProb / aprioriColor;
	minMaxLoc(objProb, &histMin, &histMax, NULL, NULL);
	objProb.convertTo(objProb,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
	probImages.push_back(objProb);
    }
    
    vector<TrackedObject> prospectiveObjects;
    //depending on whether reacquisition is necessary in the current time step, find new prospective objects in different ways
    if (frameNumber%reacquireEvery == 0 || newKindAdded){
	for (int i=0; i<probImages.size(); i++){
	    //perform Otsu tresholding and morphological closing
	    Mat binImg;
	    probImages[i].convertTo(binImg, CV_8U,255,0);
	    threshold(binImg, binImg, 255, 255, THRESH_BINARY+THRESH_OTSU);
	    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7,7));
	    dilate(binImg, binImg, element);
	    erode(binImg, binImg, element); 
	    
	    //find contours of remaining blobs
	    vector<vector<Point>> contours;
	    vector<Vec4i> hierarchy;
	    findContours(binImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	    
	    if (contours.size()>0){
		//get contour areas for elimination of smaller contours
		vector<double> contourAreas;
		contourAreas.push_back(contourArea(contours[0]));
		double maxArea = contourAreas[0];
		double minArea = contourAreas[0];
		for (int k=1; k<contours.size(); k++){
		    double area = contourArea(contours[k]);
		    contourAreas.push_back(area);
		    if (area>maxArea){maxArea=area;}
		    if (area<minArea){minArea=area;}
		}
		//eliminate smaller contours, make larger contours into objects
		double pivot = minArea + 0.5*(maxArea-minArea);
		vector<vector<Point>> largeContours;
		for (int k=0; k<contours.size(); k++){
		    if (contourAreas[k]>pivot){
			TrackedObject newObject(inputImage, contours[k]);
			newObject.kind = i;
			prospectiveObjects.push_back(newObject);
		    }
		}
	    }
	}
    }
    
    //CamShift all known objects into their new positions, deleting merged objects
    vector<RotatedRect> newObjectPositions;
    for (int i=0; i<objects.size(); i++){
	RotatedRect newBB = objects[i].useCamShift(probImages[objects[i].kind]);
	bool addNew = true;
	for (int j=0; j<newObjectPositions.size(); j++){
	    if (compareBB(newBB, newObjectPositions[j])<0){
		  objects.erase(objects.begin()+i);
		  i--;
		  addNew = false;
	    }
	}
	if (addNew){
	    newObjectPositions.push_back(newBB);
	    objects[i].ellipse=newBB;
	}
    }
    
    for (int i=0; i<prospectiveObjects.size(); i++){
	bool addNew = true;
	for (int j=0; j<newObjectPositions.size(); j++){
	    if (compareBB(prospectiveObjects[i].ellipse, newObjectPositions[j])<0){
		  prospectiveObjects.erase(prospectiveObjects.begin()+i);
		  i--;
		  addNew = false;
	    }
	}
	if (addNew){
	    objects.push_back(prospectiveObjects[i]);
	}
    }
    
    
    Mat drawImg;
    inputImage.copyTo(drawImg);
    for (int i=0; i<objects.size(); i++){
	ellipse(drawImg, objects[i].ellipse, Scalar(0,255,0));
    }
    
    drawImg.copyTo(*outputImage);
    
    frameNumber++;
    newKindAdded = false;
}


double compareBB(RotatedRect bb1, RotatedRect bb2){
    Rect r1 = bb1.boundingRect();
    Rect r2 = bb2.boundingRect();
    Rect overlap = r1&r2;
    if (overlap.area()>0){
	return -overlap.area()/min(r2.area(),r1.area());
    }
    else{
	Point pt1 = r1.tl();
	Point pt2 = r2.br();
	double dx = pt2.x - pt1.x;
	double dy = pt2.y - pt1.y;
	if (dx>0){dx -= r1.width+r2.width;} else {dx=-dx;}
	if (dy>0){dy -= r1.height+r2.height;} else {dy=-dy;}
	return min(dx,dy);
    }
}



