#ifndef OBJECTTRACKING
#define OBJECTTRACKING

#include "opencv2/imgproc/imgproc.hpp"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "ImgProcPipeline.hpp"
#include <chrono>

using namespace std;
using namespace cv;

class TrackedObject{
    protected:
	Size imageSize;
    public:
	vector<Point> listOfPoints;
	vector<TrackedObject*> occluding;
	TrackedObject* occluder;
	bool occluded;
	vector<Point> contour;
	bool tracked;
	int id;
	int kind;
	float area;
	RotatedRect ellipse;
	TrackedObject();
	TrackedObject(const Mat image, const vector<Point> inContour);
	vector<Point> actualPoints();
	
	RotatedRect useCamShift(const Mat probImage);
	double compare(TrackedObject otherObject);
	
};

class LogManager{
    protected:
	boost::filesystem::path rootDir;
	int nextId;
	chrono::time_point<chrono::system_clock> trackingStarted;
    public:
	std::vector<boost::filesystem::path> logFilePaths;
	LogManager();
	void store(TrackedObject obj);
	int getId();
};

class ObjectTracker : public ProcessingElement{
    protected:
	vector<TrackedObject> objects;
	vector<Histogram> objectKinds;
	LogManager logManager;
	LTIFilter filterLTI;
	int frameNumber;
	bool newKindAdded;
	
	//std::vector<ProcessingElement*> processingElements;
	//std::vector<int> probProcessingIdx;
	//std::vector<int> segProcessingIdx;
	
    public:
	ObjectTracker();
	void preprocess(const Mat image, Mat* outputImage, Mat* mask);
	void process(const Mat inputImage, Mat* outputImage);
	void addObjectKind(const Mat image);
	
  
};

bool intersectingOBB(RotatedRect obb1, RotatedRect obb2);

double distEllipse2Point(RotatedRect ellipse, Point pt);


#endif