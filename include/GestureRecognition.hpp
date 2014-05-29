#ifndef GESTURERECOGNITION
#define GESTURERECOGNITION

#include "opencv2/imgproc/imgproc.hpp"
#include "boost/smart_ptr.hpp"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include <cstdlib>

using namespace std;

class Gesture;

/*! Class for linear time-invariant filtering of cv::Point2f.
  * Supports causal filters of any order. Filter coefficients are stored as floats. Initial
  * conditions for all state variables are set to 0.
  */
class LTIFilter{
protected:
    /*! Record of past output values*/
    vector<cv::Point2f> out;
    /*! Record of past input values*/
    vector<cv::Point2f> in;
    /*! Vector of numerator coefficients in descending order of powers of z*/
    vector<float> numerator;
    /*! Vector of denominator coefficients in descending order of powers of z*/
    vector<float> denominator;
    /*! Discretization time. Not used*/
    float discretizationTime;
public:
    /*! Default constructor. Uninitialized filter simply acts as a gain of 1.*/
    LTIFilter();
    /*! Standard constructor.
      * Takes numerator and denominator vectors in standard MATLAB format. Numerator can be of smaller size than denominator,
      * in which case it is front-filled with zeros until it is the same length.
      * \param num Filter transfer function numerator
      * \param num Filter transfer function denominator
      * \param num Filter discretization time (not used)
      */
    LTIFilter(vector<float> num, vector<float> den, float T);
    /*! Given an input for the current time step, calculate the output.
      * Uses internal memory of past states to calculate output. Be careful not to pass the same value twice.
      * \param input Input point
      * \param output Output point
      */
    void process(cv::Point2f input, cv::Point2f& output);
};

/*! Class used to store, simplify, filter and log object trajectory data.*/
class Trajectory{
protected:
    /*! LTI filter applied to all input points before they are stored*/
    LTIFilter filt;
    /*! Recursive simplification function.
      * Necessary for the implementation of the Ramer-Douglas-Peucker simplification algorithm.
      * \param eps Maximum point distance from segment before segment is broken up
      * \param start Index of starting point of trajectory segment to recursively simplify
      * \param stop Index of end point of trajectory segment to recursively simplify
      */
    vector<int> rSimplify(float eps, int start, int stop);
public:
    /*! List of filtered trajectory points*/
    vector<cv::Point2f> points;
    /*! List of unfiltered trajectory points*/
    vector<cv::Point2f> rawPoints;
    /*! List of trajectory times in standard POSIX milliseconds since epoch format*/
    vector<long long> times;

    /*! Default constructor.
      * Use this constructor when no point filtering is desired.
      */
    Trajectory();
    /*! Constructor with filter initialization.
      * Use this constructor when all input points should be filtered.
      * \param num Filter transfer function numerator
      * \param num Filter transfer function denominator
      */
    Trajectory(vector<float> num, vector<float> den);

    /*! Append point to trajectory.
      * \param pt Float format 2-D point
      * \param time Timestamp in POSIX milliseconds since epoch format
      */
    void append(cv::Point2f pt, long long time);
    /*! Simplify trajectory using the Ramer-Douglas-Peucker algorithm.
      * \param eps Maximum point distance from segment before segment is broken up
      */
    void simplify(float eps);
    /*! Remove all points in trajectory before specified point.
      * Negative values of idx clear the entire trajectory.
      * \param idx Index of last kept point
      */
    void cutoff(int idx);

    /*! Logging function without gesture recognition.
      * Filename folder structure is constructed if it doesn't yet exist. The specified file is erased if it already exists.
      * Output format is .csv, with each line consisting of timestamp, filtered x, filtered y, unfiltered x, unfiltered y
      * \param filename Full path to file (including .csv extension)
      */
    void logTo(boost::filesystem::path filename);
    /*! Logging function with gesture recognition.
      * Function checks trajectory for presence of all the listed gestures in "endpoint terminates gesture" mode. \n
      * Filename folder structure is constructed if it doesn't yet exist. The specified file is erased if it already exists. \n
      * Output format is .csv, with each line consisting of timestamp, filtered x, filtered y, unfiltered x, unfiltered y \n
      * Two extra files are created with extensions .trajectory and .trajectoryfound. These files can be parsed using the (hopefully)
      * included MATLAB script.
      * \param filename Full path to file (including .csv extension)
      * \param gestures Vector of gesture objects to test trajectory against
      */
    void logTo(boost::filesystem::path filePath, std::vector<Gesture> gestures);

};

/*! Class used to test trajectories for the presence of gestures.
  * Gestures are integer lists with each element in range [0,7]. Each list element signifies a direction.
  * A gesture is detected when a continuous segment of the tested trajectory corresponds to all of the direction elements in the list. \n
  * Directions can be calculated from list elements by multiplying them by PI/4. Direction 0 is therefore left, 1 is up-left, 2 is up, 3 is up-right
  * and so forth. \n
  * For example, the letter M would be encoded as {2,7,1,6} when traced left to right, and {2,5,3,6} when traced right to left
  */
class Gesture{
protected:
    /*! List of directions in order.*/
    vector<int> directionList;
    /*! Check if angle falls into the allowed range for a given state.
      * \param angle Angle to test
      * \param state Index of direction list element to test against
      * \param angleOverlap Overlap between neighboring directions to prevent inadvertent state changes.
      */
    bool inState(float angle, int state, float angleOverlap = 0);
public:
    /*! Gesture name*/
    string name;

    /*! Default constructor. Do not use.*/
    Gesture();
    /*! Standard constructor.
      * \param tName Gesture name
      * \param directions List of directions in order
      */
    Gesture(std::string tName, vector<int> directions);

    /*! Check if gesture exists in specified trajectory.
      * If lastPt is set to true, a trajectory which is in the last segment of the gesture will evaluate as
      * if the gesture had been completed. Useful when object leaves the camera's field of vision. \n
      * \param traj Trajectory to test
      * \param lastPt Enable "endpoint terminates gesture" mode
      * \return A list of point index pairs corresponding to each start and end of a gesture
      */
    vector<int> existsIn(Trajectory &traj, bool lastPt);
    /*! Check if gesture exists in specified trajectory and output diagnostic information.
      * If lastPt is set to true, a trajectory which is in the last segment of the gesture will evaluate as
      * if the gesture had been completed. Useful when object leaves the camera's field of vision. \n
      * \param traj Trajectory to test
      * \param lastPt Enable "endpoint terminates gesture" mode
      * \return Debug information
      */
    vector<int> existsInDebug(Trajectory &traj, bool lastPt, float minDist);
};

#endif
