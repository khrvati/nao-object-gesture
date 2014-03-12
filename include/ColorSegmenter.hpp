#ifndef COLORSEGMENTER
#define COLORSEGMENTER

#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class ColorSegmenter{
	Mat histogram;
	Mat normalizedHistogram;
	public:
		bool initialized;
		ColorSegmenter();
		void histFromImage(const Mat image);	
		void backPropHist(const Mat image, Mat* outputImage);
		void updateHistogram(const Mat image, const Mat mask);
};
#endif


