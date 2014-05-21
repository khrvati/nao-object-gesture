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
            in.push_back(cv::Point2f(0,0));
            out.push_back(cv::Point2f(0,0));
        }
    }
}

void LTIFilter::process(cv::Point2f input, cv::Point2f &output){
    if (numerator.size()==1 && denominator.size()==1){
        output = input;
        return;
    }
    cv::Point2f temp;
    vector<cv::Point2f>::iterator it;
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

Trajectory::Trajectory(vector<float> num, vector<float> den){
    filt = LTIFilter(num, den, 1);
}

void Trajectory::append(cv::Point2f pt, long long time){
    cv::Point2f ret;
    filt.process(pt, ret);
    points.push_back(ret);
    times.push_back(time);
}

void Trajectory::cutoff(int idx){
    if (idx<points.size()-1 && idx>1){
        points.erase(points.begin(), points.begin()+idx-1);
        times.erase(times.begin(), times.begin()+idx-1);
    }
    else {
        points.clear();
        times.clear();
    }
}

//use angles for eps
void Trajectory::simplify(float eps){
    if (points.size()<2){
        return;
    }
    vector<int> keep = rSimplify(eps ,0, points.size()-1);
    vector<cv::Point2f> newPts;
    vector<long long> newTimes;
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
    cv::Point2f startpt = points[start];
    cv::Point2f endpt = points[stop];
    for (int i=start+1; i<stop; i++){
        cv::Point2f cs = points[i]-startpt;
        cv::Point2f ec = endpt - points[i];
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
        for (int i=1; i<app.size(); i++){
            ret.push_back(app[i]);
        }
        return ret;
    }
}

void Trajectory::logTo(boost::filesystem::path filePath){
    if (!filePath.empty()){
        boost::filesystem3::create_directories(filePath.parent_path());
        boost::filesystem::ofstream fileStream(filePath, ios::out | ios::app);
        for (int i=0; i<points.size(); i++){
            fileStream << times[i] << ", " << points[i].x << ", " << points[i].y << endl;
        }
        fileStream.close();
    }
}



Gesture::Gesture(vector<int> directions) : directionList(directions){}

/* old, segment continuation version
vector<int> Gesture::existsIn(Trajectory& traj, bool lastPt){
    float minDist = 10;
    float angleOverlap = 5.0/180*PI;
    vector<int> retval;
    if (traj.points.size()<3 || directionList.size()<1){
        return retval;
    }

    int state = 0;
    int startpt = 0;
    int pt0 = 0;
    bool validSeg = false;
    for (int i=0; i<traj.points.size(); i++){
        if (!validSeg){
            cv::Point ptdiff = traj.points[i]-traj.points[pt0];
            if (cv::norm(ptdiff)>=minDist){
                float angle = fmod(atan2(-ptdiff.y, ptdiff.x)+PI,(2*PI));
                validSeg = true;
                if (!inState(angle, state, angleOverlap)){
                    state = 0;
                    pt0 = i;
                    startpt = i;
                    validSeg = false;
                }
            }
        }
        else {
            cv::Point ptdiff = traj.points[i]-traj.points[i-1];
            float angle = fmod(atan2(-ptdiff.y, ptdiff.x)+PI,(2*PI));
            if (!inState(angle, state, angleOverlap)){
                state++;
                pt0 = i;
                validSeg = false;
                if (state>=directionList.size()){
                    retval.push_back(startpt);
                    retval.push_back(i);
                    state=0;
                    startpt = i;
                }
            }
        }
    }
    if (lastPt && state == (directionList.size()-1)){
        retval.push_back(startpt);
        retval.push_back(traj.points.size()-1);
    }
    return retval;
}
*/

vector<int> Gesture::existsIn(Trajectory& traj, bool lastPt){
    float minDist = 0.13;
    long long timeMs = 1500;
    float angleOverlap = 5.0/180*PI;
    vector<int> retval;
    if (traj.points.size()<3 || directionList.size()<1){
        return retval;
    }

    int state = 0;
    int startpt = 0;
    int pt0 = 0;
    for (int i=0; i<traj.points.size(); i++){
        cv::Point ptdiff = traj.points[i]-traj.points[pt0];
        long long tdiff = traj.times[i]-traj.times[pt0];
        if (cv::norm(ptdiff)>=minDist || tdiff>timeMs){
            float angle = fmod(atan2(-ptdiff.y,ptdiff.x)+PI,(2*PI));
            if (!inState(angle, state, angleOverlap) || tdiff>timeMs){
                if (state==directionList.size()-1){
                    retval.push_back(startpt);
                    retval.push_back(i);
                    state=0;
                    startpt = i;
                    pt0 = i;
                }
                else {
                    if (inState(angle, state+1, angleOverlap) && tdiff<=timeMs){
                        state++;
                        pt0 = i;
                    } else {
                        state = 0;
                        pt0 = i;
                        startpt = i;
                    }
                }
            }
            else {
                pt0 = i;
            }
        }
    }
    if (lastPt && state == (directionList.size()-1)){
        retval.push_back(startpt);
        retval.push_back(traj.points.size()-1);
    }
    return retval;
}


vector<int> Gesture::existsInDebug(Trajectory& traj, bool lastPt){
    float minDist = 20;
    long long timeMs = 1000;
    float angleOverlap = 5.0/180*PI;
    vector<int> retval;
    if (traj.points.size()<3 || directionList.size()<1){
        return retval;
    }

    int state = 0;
    int startpt = 0;
    int pt0 = 0;
    bool validSeg = false;
    for (int i=0; i<traj.points.size(); i++){
            cv::Point ptdiff = traj.points[i]-traj.points[pt0];
            long long tdiff = traj.times[i]-traj.times[pt0];
            if (cv::norm(ptdiff)>=minDist || tdiff>timeMs){
                float angle = fmod(atan2(ptdiff.y,-ptdiff.x)+PI,(2*PI));
                if (!inState(angle, state, angleOverlap) || tdiff>timeMs){
                    if (state==directionList.size()-1){
                        retval.push_back(i);
                        retval.push_back(1);
                        state=0;
                        startpt = i;
                        pt0 = i;
                    }
                    else {
                        if (inState(angle, state+1, angleOverlap) && tdiff<=timeMs){
                            retval.push_back(i);
                            retval.push_back(1);
                            state++;
                            pt0 = i;
                        } else {
                            state = 0;
                            pt0 = i;
                            startpt = i;
                            validSeg = false;
                            retval.push_back(i);
                            retval.push_back(-1);
                        }
                    }
                }
                else {
                    pt0 = i;
                    retval.push_back(i);
                    retval.push_back(0);
                }
        }
    }
    if (lastPt && state == (directionList.size()-1)){
        retval.push_back(startpt);
        retval.push_back(traj.points.size()-1);
    }
    return retval;
}

bool Gesture::inState(float angle, int state, float angleOverlap){
    float cscenter = directionList[state]*(PI/4);
    if (directionList[state] != 0){
        float cslb = cscenter - PI/8 - angleOverlap;
        float csub = cscenter + PI/8 + angleOverlap;
        return angle>cslb && angle<csub;
    } else {
        float cslb = 2*PI - PI/8 - angleOverlap;
        float csub = PI/8 + angleOverlap;
        return angle > cslb || angle < csub;
    }
}












