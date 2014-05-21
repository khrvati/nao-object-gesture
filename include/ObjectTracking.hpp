#ifndef OBJECTTRACKING
#define OBJECTTRACKING

#include "opencv2/imgproc/imgproc.hpp"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "boost/smart_ptr.hpp"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread_time.hpp>
#include <boost/ref.hpp>
#include "ImgProcPipeline.hpp"
#include "GestureRecognition.hpp"
#include <ctime>

using namespace std;
using namespace cv;

class UpdatableHistogram : public Histogram{
protected:
    int buffersize;
    vector<Mat> buffer;
    Mat offline;
public:
    UpdatableHistogram();
    UpdatableHistogram(int channels[2], int histogramSize[2], float channel1range[2], float channel2range[2], int bufferSize);
    void update(Mat image, double alpha, const Mat mask);
    void fromImage(const vector<Mat> image, const vector<Mat> mask);
    void toImage(std::string rootPath);
    bool fromStored(std::string rootPath);
};

class TrackedObject{
    protected:
        Size imageSize;
    public:
        Trajectory traj;
        int id;
        int kind;
        boost::system_time timeLost;
        bool tracked;
        bool occluded;
        Scalar color;
        vector<boost::shared_ptr<TrackedObject> > occluding;
        vector<boost::shared_ptr<TrackedObject> > occluders;
        vector<Point2i> points;
        vector<Point> contour;
        RotatedRect ellipse;
        RotatedRect actualEllipse;
        float area;
        Point2f estMove;
        TrackedObject();
        TrackedObject(const Mat image, const vector<Point> inContour, bool isContour);
	

        vector<Point> pointsFromContour();
        void update(const Mat image, const vector<Point> inContour, bool isContour);
        void updateArea();
        double getAreaRatio(double compareArea);
        double getArea();
        RotatedRect getEllipse();
        RotatedRect useCamShift(const Mat probImage);
        double compare(boost::shared_ptr<TrackedObject> otherObject);
        void unOcclude();
        void updateTrajectory(Point2f pt, long long time);
};

typedef map<int,boost::shared_ptr<TrackedObject> > objMap;

class ObjectTracker : public ProcessingElement{
    protected:
    int frameNumber;
    int nextObjectIdx;
    public:
    vector<UpdatableHistogram> objectKinds;
    objMap objects;
    //vector<boost::shared_ptr<TrackedObject> > objects;
    vector<RotatedRect> lastFrameBlobs;
    vector<int> largestObjOfKind;
	ObjectTracker();
    void preprocess(const Mat image, Mat& outputImage, Mat& mask);
    void getProbImages(const Mat procimg, const Mat mask, vector<Mat>& outputImages);
	void process(const Mat inputImage, Mat* outputImage);
    bool addObjectKind(const vector<Mat> image, const vector<Mat> outMask);
    bool addObjectKind(const vector<Mat> image, const vector<Mat> outMask, std::string path);
    bool addObjectKind(std::string path);
};

bool intersectingOBB(RotatedRect obb1, RotatedRect obb2);

double distEllipse2Point(RotatedRect ellipse, Point2f pt);

double distLine2Point(Point2d pt1, Point2d pt2, Point2d pt3);

double distRotatedRect(RotatedRect r1, RotatedRect r2);

void occludeBy(boost::shared_ptr<TrackedObject> underObject, boost::shared_ptr<TrackedObject> overObject);

void hysteresisThreshold(const cv::Mat inputImg, cv::Mat& binary, std::vector < std::vector<cv::Point2i> > &blobs, double lowThresh, double hiThresh);


#endif
