#ifndef IMGPROCPIPELINE
#define IMGPROCPIPELINE

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include <iostream>

using namespace cv;

/*! \brief A class for constructing multivariate gaussian mixture models using the EM algorithm
  *     with built-in support for generalizing histograms.
  *
  * This class can be used to create a gaussian mixture model from any dataset stored as the rows of a
  * cv::Mat. Rows are (N+1)-dimensional vectors, where N is the dimensionality of the GMM to be constructed
  * and the last element represents the number of identical data points to be taken into considerations. This
  * is done to greatly speed up the EM algorithm in the case of many identical data points, such as when
  * processing histograms.
  * The constructed model can be accessed directly for any point in N-dimensional space or via a lookup table,
  * which can be constructed for faster access in the 2-dimensional case.
  */
class GaussianMixtureModel{
    /*! Model dimensionality, N */
    int dimensions;

    /*! Number of mixture components, K */
    int components;

    /*! M-dimensional vector of component weights */
    std::vector<double> weight;

    /*! M-dimensional vector of NxN covariance matrices */
    std::vector<Mat> covarianceMatrix;

    /*! M-dimensional vector of Nx1 gaussian density function means */
    std::vector<Mat> meanVector;

    /*! Temporary variable used to store datapoint - component correspondence */
    Mat componentProbability;

    /*! Returns the value of a gaussian probability density function defined by its covariance matrix and mean vector.
      * \param x N-dimensional point
      * \param covarianceMatrix Covariance matrix of the gaussian probability density function
      * \param meanVector Mean vector of the gaussian probability density function
      */
    double gauss(const Mat x, Mat covarianceMatrix, Mat meanVector);

    /*! Returns the value of a single GMM component.
      * Returns -1 when model not initialized, or for out of range component indices.
      * \param x N-dimensional point
      * \param componentIndex Index of queried component
      */
    double gaussIdx(const Mat x, int componentIndex);

public:
    /*! Lookup table constructed with makeLookup */
    Mat lookup;

    /*! True if model has been constructed, false otherwise */
    bool initialized;

    /*! Default constructor. Does nothing, but sets initialized to false to disincentivize use. */
    GaussianMixtureModel();

    /*! Standard constructor.
      * Defines dimensionality and number of components of resulting gaussian mixture model.
      * \param dims Model dimensionality
      * \param K Number of components
      */
    GaussianMixtureModel(int dims, int K);

    /*! Copy constructor */
    GaussianMixtureModel(const GaussianMixtureModel& other);

    /*! Assignment operator */
    GaussianMixtureModel& operator=(const GaussianMixtureModel& other);

    /*! Runs the EM algorithm on a dataset consisting of (N+1)-dimensional row vectors.
      *
      * \param samples A matrix of size Mx(N+1), where M is the number of unique data points and N is the model's dimensionality.
      *     The last element in the row vector is the number of identical data points.
      * \param maxIterations Maximum number of iterations after which the EM algorithm terminates
      * \param minStepIncrease Percentage increase of log likelihood below which the local optimum is presumed to have been achieved
      */
    void runExpectationMaximization(const Mat samples, int maxIterations, double minStepIncrease);

    /*! Gets the model value at a point
      * \param x N-dimensional query point
      */
    double get(const Mat x);

    /*! Makes a lookup table in histogram format for easier and faster access.
      * Only works for two-dimensional gaussian mixture models. The created uniform histogram is defined by the number of bins in
      * each dimension and the range of values in each dimension. The function stores the histogram in the object's lookup attribute
      * after normalizing it so that the max value equals 1
      *
      * \param histSize Number of bins in each dimension.
      * \param c1range Range of histogram values for the first dimension
      * \param c2range Range of histogram values for the second dimension
      */
    void makeLookup(int histSize[2], float c1range[2], float c2range[2]);

    /*! Given a two-dimensional histogram, generate a GMM that best describes it.
      * The function calls makeLookup after the GMM has been generated.
      *
      * \param histogram A 2-dimensional histogram
      * \param histSize Number of bins in each dimension.
      * \param c1range Range of histogram values for the first dimension
      * \param c2range Range of histogram values for the second dimension
      * \param maxIter Maximum number of iterations after which the EM algorithm terminates
      * \param minStepIncrease Percentage increase of log likelihood below which the local optimum is presumed to have been achieved
      */
    void fromHistogram(const Mat histogram, int histSize[2], float c1range[2], float c2range[2], int maxIter, double minStepIncrease);

};

/*! A class used for simplifying histogram backpropagation, with support for gaussian mixture model construction.
  *
  * This class is a container for all the data needed to construct and backpropagate a histogram using OpenCV's built-in functions.
  * It provides a simple way to use histogram backpropagation with two-dimensional histograms. It also contains an uninitialized
  * gaussian mixture model which can be constructed from the stored histogram whenever needed.
  */
class Histogram{
protected:
    /*! Integer matrix used to store the raw number of pixels*/
    Mat accumulator;

    /*! Number of histogram bins in each dimension*/
    int histSize[2];

    /*! Array of image channels to construct the histogram from*/
    int channels[2];

    /*! Range of histogram values for the first dimension*/
    float c1range[2];

    /*! Range of histogram values for the second dimension*/
    float c2range[2];

    /*! Gaussian mixture model object*/
    GaussianMixtureModel gmm;
public:
    /*! Boolean flag used to check if GMM is initialized*/
    bool gmmReady;

    /*! The actual histogram, normalized so that the maximum value equals 1*/
    Mat normalized;

    /*! Default constructor*/
    Histogram();

    /*! Standard constructor.
      * Use this constructor to initialize the object, then call fromImage to make the histogram.
      *
      * \param channels Array of image channels to construct the histogram from
      * \param histogramSize Number of histogram bins in each dimension
      * \param channel1range Range of histogram values for the first dimension
      * \param channel2range Range of histogram values for the second dimension
      */
    Histogram(int channels[2], int histogramSize[2], float channel1range[2], float channel2range[2]);

    /*! Copy constructor*/
    Histogram(const Histogram& other);

    /*! Assignment operator*/
    Histogram& operator=(const Histogram& other);

    /*! Creates a histogram from a masked image.
      * If the mask is not an empty matrix, the histogram is generated from those pixels at which the mask is equal to
      * 255.
      *
      * \param image Input image already converted to the desired color space
      * \param mask Matrix of type CV_8U which defines the pixels to consider when building the histogram.
      */
    virtual void fromImage(Mat image, const Mat mask);

    /*! Updates an existing histogram.
      * Given a masked image and a learning coefficient alpha, decay the pixel accumulator by *=(1-alpha), add the masked
      * pixels to the accumulator, then construct a new histogram from the accumulated pixels.
      *
      * \param image Input image already converted to the desired color space
      * \param alpha Learning coefficient
      * \param mask Matrix of type CV_8U which defines the pixels to consider when building the histogram.
      */
    virtual void update(Mat image, double alpha, const Mat mask);

    /*! Runs histogram backpropagation on the input image and returns a probability image.
      *
      * \param inputImage Input image already converted to the desired color space
      * \param outputImage Pointer to the output matrix
      */
    void backPropagate(Mat inputImage, Mat* outputImage);

    /*! Makes a gaussian mixture model from the existing histogram and stores a normalized lookup table as the new histogram.
      *
      * \param K Number of components for the gaussian mixture model
      * \param maxIter Maximum number of iterations after which the EM algorithm terminates
      * \param minStepIncrease Percentage increase of log likelihood below which the local optimum is presumed to have been achieved
      */
    void makeGMM(int K, int maxIter, double minStepIncrease);

    /*! Sets the stored histogram size to the new specified values.
      *
      * \param histogramSize New histogram size
      */
    void resize(int histogramSize[2]);
};


/*! An abstract class used as the base class for all image processing pipeline components.
 */
class ProcessingElement{
public:
    std::string name;
    bool initialized;
    virtual void process(const Mat inputImage, Mat* outputImage) = 0;
};

/*! 2D histogram-based image flattening class. Supports HSV, HLS and YUV colorspaces.
  */
class ColorHistBackProject : public ProcessingElement{
protected:
    Mat histogramMask;
    int colorspaceCode;
    Histogram objHistogram;
    void preprocess(const Mat image, Mat* outputImage);
public:
    //bool initialized;
    ColorHistBackProject();
    ColorHistBackProject(int code, const int* histogramSize);
    ColorHistBackProject(int code, const int* histogramSize, String filename);
    virtual void histFromImage(const Mat image);
    void updateHistogram(const Mat image, const Mat mask);
    virtual void process(const Mat inputImage, Mat* outputImage);
};

class BayesColorHistBackProject : public ColorHistBackProject{
public:
    void histFromImage(const Mat image);
    void process(const Mat inputImage, Mat* outputImage);
    BayesColorHistBackProject(int code, const int* histogramSize) : ColorHistBackProject(code, histogramSize) {};
    BayesColorHistBackProject(int code, const int* histogramSize, String filename) : ColorHistBackProject(code, histogramSize,filename) {};
};

class GMMColorHistBackProject : public ColorHistBackProject{
public:
    void histFromImage(const Mat image);
    void process(const Mat inputImage, Mat* outputImage);
    GMMColorHistBackProject();
    GMMColorHistBackProject(int code, const int* histogramSize);
    GMMColorHistBackProject(int code, const int* histogramSize, String filename);

};

class SimpleThresholder : public ProcessingElement{
    float thresholdValue;
public:
    SimpleThresholder();
    SimpleThresholder(float threshValue);
    void process(const Mat inputImage, Mat* outputImage);
};

class SimpleBlobDetect : public ProcessingElement{
public:
    SimpleBlobDetect();
    void process(const Mat inputImage, Mat* outputImage);

};

class OpticalFlow : public ProcessingElement{
protected:
    bool init;
    Mat old;
public:
    OpticalFlow();
    void process(const Mat inputImage, Mat* outputImage);
};

class BGSubtractor : public ProcessingElement{
protected:
    bool init;
    BackgroundSubtractorMOG bgsub;
    Mat old;
public:
    BGSubtractor();
    void process(const Mat inputImage, Mat* outputImage);
};

#endif


