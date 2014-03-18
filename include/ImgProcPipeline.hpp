#ifndef IMGPROCPIPELINE
#define IMGPROCPIPELINE

#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

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

class SimpleThresholder : public ProcessingElement{
    float thresholdValue;
    public:
	SimpleThresholder();
	SimpleThresholder(float threshValue);
	void process(const Mat inputImage, Mat* outputImage);
};


#endif


