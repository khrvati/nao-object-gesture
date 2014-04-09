#ifndef OBJECTTRACKING
#define OBJECTTRACKING

#include "opencv2/imgproc/imgproc.hpp"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "ImgProcPipeline.hpp"

using namespace std;
using namespace cv;

class LogManager{
    protected:
	boost::filesystem::path rootDir;
	int nextId;
    public:
	std::vector<boost::filesystem::path> logFilePaths;
	LogManager();
	int getId();
};

class TrackedObject{
    protected:
	vector<Point> listOfPoints;
	Size imageSize;
    public:
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

double compareBB(RotatedRect bb1, RotatedRect bb2);


#endif