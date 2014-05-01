#include "opencv2/core/core.hpp"
#include "boost/smart_ptr.hpp"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include <cstdlib>
#include <math.h>
#include "GestureRecognition.hpp"

using namespace std;
#define PI 3.1415926535897932



/*Input numerator coefficients in standard MATLAB format  */
LTIFilter::LTIFilter(){
    discretizationTime = 1;
    numerator = {1};
    denominator = {1};
}

LTIFilter::LTIFilter(vector<float> num, vector<float> den, float T){
    if (num.size()<=den.size()){
        numerator = num;
        denominator = den;
        float norm = den[0];
        for (int i=0; i<numerator.size(); i++){
            numerator[i]/=norm;
        }
        for (int i=0; i<denominator.size(); i++){
            denominator[i]/=norm;
        }
        while (numerator.size()<denominator.size()){
            vector<float>::iterator it;
            it = numerator.begin();
            numerator.insert(it,0.0);
        }
        discretizationTime = T;
        for (int i=0; i<denominator.size(); i++){
            in.push_back(cv::Point(0,0));
            out.push_back(cv::Point(0,0));
        }
    }
}

void LTIFilter::process(cv::Point input, cv::Point &output){
    cv::Point temp;
    vector<cv::Point>::iterator it;
    in.push_back(input);
    for(int i=numerator.size()-1; i>=0; i--){
        temp+=in[numerator.size()-i-1]*numerator[i];
    }
    for(int i=denominator.size()-1; i>0; i--){
        temp-=out[denominator.size()-i-1]*denominator[i];
    }
    it = in.begin();
    in.erase(it);
    it = out.begin();
    out.erase(it);
    out.push_back(temp);
    output = temp;
}




Trajectory::Trajectory(): filt(){}

Trajectory::Trajectory(vector<float> num, vector<float> den): filt(num, den, 1){}

void Trajectory::append(cv::Point pt, float time){
    cv::Point ret;
    filt.process(pt, ret);
    points.push_back(ret);
    times.push_back(time);
    if (!filename.empty()){
        boost::filesystem::ofstream fileStream(filename, ios::out | ios::app);
        fileStream << time << ", " << pt.x << ", " << pt.y << endl;
    }
}

//use angles for eps
void Trajectory::simplify(float eps){
    if (points.size()<2){
        return;
    }
    vector<int> keep = rSimplify(eps ,0, points.size()-1);
    vector<cv::Point> newPts;
    vector<float> newTimes;
    for (int i=0; i<keep.size(); i++){
        newPts.push_back(points[keep[i]]);
        newTimes.push_back(times[keep[i]]);
    }
    points=newPts;
    times = newTimes;
}

vector<int> Trajectory::rSimplify(float eps, int start, int stop){
    float max = 0;
    int idx = -1;
    cv::Point startpt = points[start];
    cv::Point endpt = points[stop];
    for (int i=start+1; i<stop; i++){
        cv::Point cs = points[i]-startpt;
        cv::Point ec = endpt - points[i];
        float dist = abs(ec.x*cs.y-cs.x*ec.y)/cv::norm(ec);
        if (dist > eps && dist>max){
            max = dist;
            idx = i;
        }
    }
    vector<int> ret;
    ret.push_back(start);
    if (idx==-1){
        ret.push_back(stop);
        return ret;
    } else {
        vector<int> app = rSimplify(eps, start, idx);
        for (int i=1; i<app.size(); i++){
            ret.push_back(app[i]);
        }
        app = rSimplify(eps, idx, stop);
        for (int i=0; i<app.size(); i++){
            ret.push_back(app[i]);
        }
        return ret;
    }
}

void Trajectory::logTo(boost::filesystem::path filePath){
    filename = filePath;
}



Gesture::Gesture(vector<int> directions) : directionList(directions){}

vector<int> Gesture::existsIn(Trajectory& traj)
{
    float minDist = 0.5;
    vector<int> retval;
    if (traj.points.size()<3 || directionList.size()<2){
        return retval;
    }

    int state = 0;
    float dist = 0;
    int startpt = 0;
    for (int i=1; i<traj.points.size(); i++){
        cv::Point ptdiff = traj.points[i]-traj.points[i-1];
        float angle = fmod(atan2(ptdiff.y, ptdiff.x),(2*PI));
        int tmp = floor(angle/(PI/4));
        bool inCurrentState = tmp == directionList[state] || tmp == (directionList[state]-1)%8;
        bool inNextState = true;
        bool inFinalState = state != directionList.size()-1;
        if (!inFinalState){
            inNextState = tmp == directionList[state+1] || tmp == (directionList[state+1]-1)%8;
        }

        if (dist<minDist){
            if (inCurrentState){
                dist += cv::norm(ptdiff);
            } else {
                startpt = i;
                dist = 0;
            }
        } else {
            if (!inCurrentState && inFinalState){
                retval.push_back(startpt);
                retval.push_back(i);
                startpt = i;
                dist = 0;
                state = 0;
                continue;
            }

            if (!inCurrentState && !inNextState){
                startpt = i;
                dist = 0;
                continue;
            }

            if (inNextState){
                state++;
                dist = 0;
                continue;
            }

            dist += cv::norm(ptdiff);
        }
    }
    return retval;
}















