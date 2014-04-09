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
    trackingStarted = chrono::system_clock::now();
}

int LogManager::getId(){
    boost::filesystem::path filePath;
    filePath = rootDir;
    int id = nextId++;
    string filename = "object_";
    filename.append(to_string(id));
    filename.append(".csv");
    filePath /= filename;
    logFilePaths.push_back(filePath);
    return id;
}

void LogManager::store(TrackedObject obj){
    if (obj.id >=0 && obj.id < logFilePaths.size()){
	auto diff = std::chrono::system_clock::now()-trackingStarted;
	std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
	int time = ms.count();
	boost::filesystem::ofstream fileStream(logFilePaths[obj.id], ios::out | ios::app);
	fileStream << time << ", " << obj.ellipse.center.x << ", " << obj.ellipse.center.y << endl;
    }
}



TrackedObject::TrackedObject(){
    tracked = false;
}

TrackedObject::TrackedObject(const Mat image, const vector<Point> inContour){
    if (inContour.size()<5) {tracked = false; return;}
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
    //double size = min(ellipse.size.height, ellipse.size.width);
    //Point tl(ellipse.center.x-size/2, ellipse.center.y-size/2);
    //Rect box(tl, Size(size,size));
    Rect box = ellipse.boundingRect();
    return CamShift(probImage, box, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 5, 1 ));
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
    int reacquireEvery = 5; //perform reacquisition every N frames processed
    double minimumAreaCutoff = 500;
    bool bayesCompensate = true;
    Mat procimg;
    Mat mask;
    Mat drawImg;
    inputImage.copyTo(drawImg);
    filterLTI.process(inputImage, &procimg);
    preprocess(procimg, &procimg, &mask);
    
    //first, get the general image histogram and use it to get a normalized probability image of the input image
    Mat aprioriColor;
    if (bayesCompensate){
	int channels[2] = {0,1};
	float c1range[2] = {0,180};
	float c2range[2] = {0,256};
	int histSize[2] = {32,32};
	Histogram imgHist(channels, histSize, c1range, c2range);
	imgHist.fromImage(procimg, mask);
	imgHist.backPropagate(procimg, &aprioriColor);
	medianBlur(aprioriColor, aprioriColor, 5);
    }
    //then, apply bayesian histogram backpropagation for each object kind to create K probability images, one for each kind
    double histMax = 0;
    double histMin = 0;
    vector<Mat> probImages;
    for (int i=0; i<objectKinds.size(); i++){
	Mat objProb;
	objectKinds[i].backPropagate(procimg, &objProb);
	if (bayesCompensate){
	    medianBlur(objProb, objProb, 5);
	    objProb = objProb / aprioriColor;
	    minMaxLoc(objProb, &histMin, &histMax, NULL, NULL);
	    objProb.convertTo(objProb,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
	    probImages.push_back(objProb);
	} else {
	    probImages.push_back(objProb);
	}
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
		    if (contourAreas[k]>pivot && contourAreas[k]>minimumAreaCutoff){
			TrackedObject newObject(inputImage, contours[k]);
			if (newObject.tracked){
			    newObject.kind = i;
			    prospectiveObjects.push_back(newObject);
			}
		    }
		}
	    }
	}
    }
    
    vector<vector<Point>> tempct;
    for (int i=0; i<prospectiveObjects.size(); i++){
	  tempct.push_back(prospectiveObjects[i].contour);
    }
    //drawContours(drawImg, tempct, -1, Scalar(0,255,0), 3);
    tempct.clear();
    
    //CamShift all known objects into their new positions, deleting merged objects
    vector<RotatedRect> newObjectPositions;
    for (int i=0; i<objects.size(); i++){
	RotatedRect newBB = objects[i].useCamShift(probImages[objects[i].kind]);
	//rectangle(drawImg,newBB.boundingRect(),Scalar(0,0,255));
	if (newBB.size.width <= 0 || newBB.size.height <=0){
	    objects.erase(objects.begin()+i);
	    i--;
	    continue;
	}
	bool addNew = true;
	for (int j=0; j<newObjectPositions.size() && addNew; j++){
	    if (intersectingOBB(newBB, newObjectPositions[j])){
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
	//rectangle(drawImg,prospectiveObjects[i].ellipse.boundingRect(),Scalar(255,0,0));
	for (int j=0; j<newObjectPositions.size() && addNew; j++){
	    if (intersectingOBB(prospectiveObjects[i].ellipse, newObjectPositions[j])){
		  prospectiveObjects.erase(prospectiveObjects.begin()+i);
		  i--;
		  addNew = false;
	    }
	}
	if (addNew){
	    objects.push_back(prospectiveObjects[i]);
	}
    }
    
    
    
    for (int i=0; i<objects.size(); i++){
	if (objects[i].id == -1){
	    objects[i].id = logManager.getId();
	}
	logManager.store(objects[i]);
	ellipse(drawImg, objects[i].ellipse, Scalar(0,255,0));
    }
    
    drawImg.copyTo(*outputImage);
    
    frameNumber++;
    newKindAdded = false;
}


//separating axis theorem implementation
bool intersectingOBB(RotatedRect obb1, RotatedRect obb2){
    double radAng1 = -obb1.angle/180.0*3.1415927;
    double radAng2 = -obb2.angle/180.0*3.1415927;
    Mat rot1 = (Mat_<float>(2,2) << cos(radAng1), -sin(radAng1), sin(radAng1), cos(radAng1));
    Mat rot2 = (Mat_<float>(2,2) << cos(radAng2), -sin(radAng2), sin(radAng2), cos(radAng2));
    
    Point2f vertices1[4];
    obb1.points(vertices1);
    Point2f vertices2[4];
    obb2.points(vertices2);
    
    Mat points1(2,4,CV_32F);
    Mat points2(2,4,CV_32F);
    
    for (int i=0; i<4; i++){
	points1.at<float>(0,i)=vertices1[i].x;
	points1.at<float>(1,i)=vertices1[i].y;
	points2.at<float>(0,i)=vertices2[i].x;
	points2.at<float>(1,i)=vertices2[i].y;
    }
    
    Mat pt1rot = rot1*points1;
    Mat pt2rot = rot1*points2;
    double minx1 = 0;
    double maxx1 = 0;
    double minx2 = 0;
    double maxx2 = 0;
    double miny1 = 0;
    double maxy1 = 0;
    double miny2 = 0;
    double maxy2 = 0;
    minMaxLoc(pt1rot.row(0), &minx1, &maxx1);
    minMaxLoc(pt2rot.row(0), &minx2, &maxx2);
    minMaxLoc(pt1rot.row(1), &miny1, &maxy1);
    minMaxLoc(pt2rot.row(1), &miny2, &maxy2);
    
    if (minx1>maxx2 || minx2>maxx1 || miny1>maxy2 || miny2>maxy1){
	return false;
    }
    
    pt1rot = rot2*points1;
    pt2rot = rot2*points2;
    minMaxLoc(pt1rot.row(0), &minx1, &maxx1);
    minMaxLoc(pt2rot.row(0), &minx2, &maxx2);
    minMaxLoc(pt1rot.row(1), &miny1, &maxy1);
    minMaxLoc(pt2rot.row(1), &miny2, &maxy2);
    
    if (minx1>maxx2 || minx2>maxx1 || miny1>maxy2 || miny2>maxy1){
	return false;
    }
    
    return true;
}

double distEllipse2Point(RotatedRect ellipse, Point2f pt){
    Point2f ptshift = pt-ellipse.center;
    double angle = atan2(ptshift.y, ptshift.x)-ellipse.angle;
    Point2f ptedge(ellipse.size.width/2.0*cos(angle), ellipse.size.height/2.0*sin(angle));
    double distEdge = norm(ptedge);
    double distPt = norm(ptshift);
    
    if (distPt<distEdge){
	return -(distEdge-distPt)/distEdge;
    }
    else{
	return distPt-distEdge;
    }
}


