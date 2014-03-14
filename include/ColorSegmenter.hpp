#ifndef COLORSEGMENTER
#define COLORSEGMENTER

#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class ColorSegmenter{
	Mat histogram;
	Mat normalizedHistogram;
	int colorspaceCode;
	int histSize[2];
	int channels[2];
	float c1range[2];
	float c2range[2];
	void preprocess(const Mat image, Mat* outputImage);
	
	public:
		bool initialized;
		ColorSegmenter();
		ColorSegmenter(int code, const int* histogramSize);
		void histFromImage(const Mat image);	
		void backPropHist(const Mat image, Mat* outputImage);
		void updateHistogram(const Mat image, const Mat mask);
};
#endif


