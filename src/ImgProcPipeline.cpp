#include "opencv2/imgproc/imgproc.hpp"
#include "ImgProcPipeline.hpp"
#include <iostream>

using namespace cv;

ColorSegmenter::ColorSegmenter(){
    histSize[0] = 64;
    histSize[1] = 64;
    colorspaceCode = CV_BGR2HSV;
    c1range[0]=0; c1range[1]=180; c2range[0]=0; c2range[1]=256;
    channels[0]=0; channels[1]=1;
    initialized=false;
}
  
ColorSegmenter::ColorSegmenter(int code, const int* histogramSize){
	histSize[0] = histogramSize[0];
	histSize[1] = histogramSize[1];
	colorspaceCode=code;
	c1range[0]=0; c1range[1]=256; c2range[0]=0; c2range[1]=256;
	channels[0]=0; channels[1]=1;
	switch (code){
	  case CV_BGR2HSV:
	      c1range[1]=180; break;
	  case CV_BGR2HLS:
	      c1range[1]=180; break;
	  case CV_BGR2YUV:
	      channels[0]=1; channels[1]=2;
	default:
	   break;
	}
	initialized=false;
}

void ColorSegmenter::preprocess(const Mat image, Mat* outputImage){
      cvtColor(image, *outputImage, colorspaceCode);
      GaussianBlur(*outputImage, *outputImage, Size(7,7),0.2);
}

void ColorSegmenter::histFromImage(const Mat image)
{
	Mat cvtImage;
	preprocess(image, &cvtImage);
	
	const float* ranges[] = {c1range, c2range};
	calcHist(&cvtImage, 1, channels, Mat(), histogram, 2, histSize, ranges, true, false);

	double histMax = 0;
	double histMin = 0;
	minMaxLoc(histogram, &histMin, &histMax, NULL, NULL);
	std::cout << "Range = (" << histMin << ", " << histMax <<  ")" << std::endl;
	
	histogram.convertTo(normalizedHistogram,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
	initialized=true;
	
	/*std::cout << "Histogram: " << histogram.depth() << ", " << histogram.channels() << std::endl;
	std::cout << "Normalized histogram: " << normalizedHistogram.depth() << ", " << normalizedHistogram.channels() << std::endl;
	minMaxLoc(normalizedHistogram, &histMin, &histMax, NULL, NULL);
	std::cout << "Range = (" << histMin << ", " << histMax <<  ")" << std::endl;*/
	
	
};

void ColorSegmenter::updateHistogram(const Mat image, const Mat mask)
{
	float alpha = 0.1; //learning coefficient
	
	Mat cvtImage;
	preprocess(image, &cvtImage);
	
	const float* ranges[] = {c1range, c2range};
	calcHist(&cvtImage, 1, channels, mask, histogram, 2, histSize, ranges, true, false);

	
	double histMax = 0;
	double histMin = 0;
	minMaxLoc(histogram, &histMin, &histMax, NULL, NULL);
	
	Mat newNormalizedHistogram;
	histogram.convertTo(newNormalizedHistogram,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
	
	if (initialized){
	normalizedHistogram = normalizedHistogram * (1-alpha) 
							+ newNormalizedHistogram * alpha;
	} 
	else {
		normalizedHistogram = newNormalizedHistogram;
		}
}

void ColorSegmenter::process(const Mat inputImage, Mat* outputImage)
{
	Mat cvtImage;
	preprocess(inputImage, &cvtImage);
	
	const float* ranges[] = {c1range, c2range};
	calcBackProject(&cvtImage, 1, channels, normalizedHistogram, *outputImage, ranges, 255, true);
	
	return;
}
