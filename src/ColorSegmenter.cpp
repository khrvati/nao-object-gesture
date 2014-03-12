#include "opencv2/imgproc/imgproc.hpp"
#include "ColorSegmenter.hpp"
#include <iostream>

using namespace cv;

ColorSegmenter::ColorSegmenter(){
	initialized=false;
}

/* The extracted histogram assumes a YUV image */
void ColorSegmenter::histFromImage(const Mat image)
{
	Mat yuv;
	cvtColor(image, yuv, CV_BGR2HSV);
	int histSize[] = {32, 32};
	int channels[] = {0, 1};
	float hranges[] = {0, 180};
	float sranges[] = {0, 256};
	const float* ranges[] = {hranges, sranges};
	calcHist(&yuv, 1, channels, Mat(), histogram, 2, histSize, ranges, true, false);

	double histMax = 0;
	double histMin = 0;
	minMaxLoc(histogram, &histMin, &histMax, NULL, NULL);
	
	normalizedHistogram = (histogram-histMin)/(histMax-histMin);
	initialized=true;
};

void ColorSegmenter::updateHistogram(const Mat image, const Mat mask)
{
	float alpha = 0.1; //learning coefficient
	
	Mat yuv;
	cvtColor(image, yuv, CV_BGR2HSV);
	int histSize[] = {32, 32};
	int channels[] = {0, 1};
	float hranges[] = {0, 180};
	float sranges[] = {0, 256};
	const float* ranges[] = {hranges, sranges};
	calcHist(&yuv, 1, channels, mask, histogram, 2, histSize, ranges, true, false);

	
	double histMax = 0;
	double histMin = 0;
	minMaxLoc(histogram, &histMin, &histMax, NULL, NULL);
	
	Mat newNormalizedHistogram = (histogram-histMin)/(histMax-histMin);
	
	if (initialized){
	normalizedHistogram = normalizedHistogram * (1-alpha) 
							+ newNormalizedHistogram * alpha;
	} 
	else {
		normalizedHistogram = newNormalizedHistogram;
		}
}

void ColorSegmenter::backPropHist(const Mat image, Mat* outputImage)
{
	Mat yuv;
	cvtColor(image, yuv, CV_BGR2HSV);
	int histSize[] = {32, 32};
	int channels[] = {0, 1};
	float hranges[] = {0, 180};
	float sranges[] = {0, 256};
	const float* ranges[] = {hranges, sranges};
	calcBackProject(&yuv, 1, channels, histogram, *outputImage, ranges, 1, true);
	/*double histMax = 0;
	double histMin = 0;
	minMaxLoc(*outputImage, &histMin, &histMax, NULL, NULL);
	std::cout << histMin << ", " << histMax << std::endl;*/
	
	return;
}
