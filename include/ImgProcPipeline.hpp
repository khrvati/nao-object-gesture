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
class ColorSegmenter : public ProcessingElement{
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
		ColorSegmenter();
		ColorSegmenter(int code, const int* histogramSize);
		void histFromImage(const Mat image);	
		void updateHistogram(const Mat image, const Mat mask);
		void process(const Mat inputImage, Mat* outputImage);
};
#endif


