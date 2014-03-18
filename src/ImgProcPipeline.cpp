#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ImgProcPipeline.hpp"
#include <iostream>

using namespace cv;

ColorHistBackProject::ColorHistBackProject(){
    histSize[0] = 64;
    histSize[1] = 64;
    colorspaceCode = CV_BGR2HSV;
    c1range[0]=0; c1range[1]=180; c2range[0]=0; c2range[1]=256;
    channels[0]=0; channels[1]=1;
    initialized=false;
}
  
ColorHistBackProject::ColorHistBackProject(int code, const int* histogramSize){
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

ColorHistBackProject::ColorHistBackProject(int code, const int* histogramSize, String filename){
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
	Mat img = imread(filename);
	histFromImage(img);
	initialized=true;
}

void ColorHistBackProject::preprocess(const Mat image, Mat* outputImage){
      //GaussianBlur(image, *outputImage, Size(15,15),0);
      //medianBlur(image, *outputImage, 7);
      cvtColor(image, *outputImage, colorspaceCode);
      medianBlur(*outputImage, *outputImage, 7);
}

void ColorHistBackProject::histFromImage(const Mat image){
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
};

void ColorHistBackProject::updateHistogram(const Mat image, const Mat mask){
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

void ColorHistBackProject::process(const Mat inputImage, Mat* outputImage){
	Mat cvtImage;
	preprocess(inputImage, &cvtImage);
	
	const float* ranges[] = {c1range, c2range};
	calcBackProject(&cvtImage, 1, channels, normalizedHistogram, *outputImage, ranges, 255, true);
	
	outputImage->convertTo(*outputImage, CV_32FC1, 1.0/255.0);	
}

void BayesColorHistBackProject::process(const Mat inputImage, Mat* outputImage){
	Mat cvtImage;
	preprocess(inputImage, &cvtImage);
	
	const float* ranges[] = {c1range, c2range};
	calcBackProject(&cvtImage, 1, channels, normalizedHistogram, *outputImage, ranges, 255, true);
	
	Mat imgHist;
	calcHist(&cvtImage, 1, channels, Mat(), imgHist, 2, histSize, ranges, true, false);
	double histMax = 0;
	double histMin = 0;
	minMaxLoc(imgHist, &histMin, &histMax, NULL, NULL);
	imgHist.convertTo(imgHist,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
	imgHist *= 1.0/norm(sum(imgHist));
	float temp = norm(sum(imgHist));
	Mat aprioriColor;
	calcBackProject(&cvtImage, 1, channels, imgHist, aprioriColor, ranges, 255, true);
	
	outputImage->convertTo(*outputImage, CV_32FC1, 1.0/255.0);
	aprioriColor.convertTo(aprioriColor, CV_32FC1, 1.0/255.0);
	
	minMaxLoc(aprioriColor, &histMin, &histMax, NULL, NULL);
	
	*outputImage = *outputImage*0.5/aprioriColor;
	minMaxLoc(*outputImage, &histMin, &histMax, NULL, NULL);

	//outputImage->convertTo(*outputImage,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
	
	//aprioriColor.copyTo(*outputImage);
	
	return;
}

void BayesColorHistBackProject::histFromImage(const Mat image){
	Mat cvtImage;
	preprocess(image, &cvtImage);
	
	const float* ranges[] = {c1range, c2range};
	calcHist(&cvtImage, 1, channels, Mat(), histogram, 2, histSize, ranges, true, false);

	double histMax = 0;
	double histMin = 0;
	minMaxLoc(histogram, &histMin, &histMax, NULL, NULL);
	std::cout << "Range = (" << histMin << ", " << histMax <<  ")" << std::endl;
	
	histogram.convertTo(normalizedHistogram,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
	normalizedHistogram *= 1.0/norm(sum(normalizedHistogram));
	
	initialized=true;
}

SimpleThresholder::SimpleThresholder(){
    thresholdValue = 0.5;
    initialized = true;
}

SimpleThresholder::SimpleThresholder(float threshValue){
    thresholdValue = threshValue;
    initialized = true;
}

void SimpleThresholder::process(const Mat inputImage, Mat* outputImage){
    threshold(inputImage, *outputImage, thresholdValue, 1.0, THRESH_BINARY);
    Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
    erode(*outputImage, *outputImage, element);
    dilate(*outputImage, *outputImage, element);
    element = getStructuringElement(MORPH_ELLIPSE, Size(7,7));
    dilate(*outputImage, *outputImage, element);
    erode(*outputImage, *outputImage, element);
}