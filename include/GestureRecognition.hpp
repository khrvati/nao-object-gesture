#include "opencv2/imgproc/imgproc.hpp"
#include "boost/smart_ptr.hpp"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include <cstdlib>

using namespace std;

class LTIFilter{
protected:
    vector<cv::Point> out;
    vector<cv::Point> in;
    vector<float> numerator;
    vector<float> denominator;
    float discretizationTime;
public:
    LTIFilter();
    LTIFilter(vector<float> num, vector<float> den, float T);
    void process(cv::Point input, cv::Point& output);
};

class Trajectory{
protected:
    LTIFilter filt;
    boost::filesystem::path filename;
    vector<int> rSimplify(float eps, int start, int stop);
public:
    vector<cv::Point> points;
    vector<float> times;
    Trajectory();
    Trajectory(vector<float> num, vector<float> den);
    void logTo(boost::filesystem::path filename);
    void append(cv::Point pt, float time);
    void simplify(float eps);
};

class Gesture{
protected:
    string name;
    vector<int> directionList;
public:
    Gesture();
    Gesture(vector<int> directions);
    vector<int> existsIn(Trajectory &traj);
};
