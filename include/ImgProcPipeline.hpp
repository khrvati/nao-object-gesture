#ifndef IMGPROCPIPELINE
#define IMGPROCPIPELINE

#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;


class GaussianMixtureModel{
    int dimensions;
    int components;
    std::vector<double> weight; 
    std::vector<Mat> covarianceMatrix;
    std::vector<Mat> meanVector;
    Mat componentProbability;
    void runExpectationMaximization(const Mat samples, int maxIterations, double minStepIncrease);
    double gauss(const Mat x, Mat covarianceMatrix, Mat meanVector);
    double gaussIdx(const Mat x, int componentIndex);
    public:
	Mat lookup;
	bool initialized;
	GaussianMixtureModel(int dims, int K);
	double get(const Mat x);
	void makeLookup(int histSize[2], float c1range[2], float c2range[2]);
	void fromHistogram(const Mat histogram, int histSize[2], float c1range[2], float c2range[2]); 
  
};

/**
 * An abstract base class for all image processing pipeline components.
 */
class ProcessingElement{
    public:
	bool initialized;
	virtual void process(const Mat inputImage, Mat* outputImage) = 0;
};

/**
 * 2D histogram-based image flattening class. Supports HSV, HLS and YUV colorspaces.
 */
class ColorHistBackProject : public ProcessingElement{
    protected:
	Mat histogram;
	Mat normalizedHistogram;
	Mat histogramMask;
	int colorspaceCode;
	int histSize[2];
	int channels[2];
	float c1range[2];
	float c2range[2];
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
    GaussianMixtureModel* gmm;
    public:
	void histFromImage(const Mat image);
	void process(const Mat inputImage, Mat* outputImage);
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

#endif

