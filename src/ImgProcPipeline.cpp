#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "ImgProcPipeline.hpp"
#include <iostream>
#include <cmath>

using namespace cv;

Histogram::Histogram(){};

Histogram::Histogram(int inchannels[2], int histogramSize[2], float channel1range[2], float channel2range[2]){
    channels[0] = inchannels[0];
    channels[1] = inchannels[1];
    histSize[0] = histogramSize[0];
    histSize[1] = histogramSize[1];
    c1range[0] = channel1range[0];
    c1range[1] = channel1range[1];
    c2range[0] = channel2range[0];
    c2range[1] = channel2range[1];
    gmmReady = false;
}

Histogram::Histogram(const Histogram& other){
    channels[0] = other.channels[0];
    channels[1] = other.channels[1];
    histSize[0] = other.histSize[0];
    histSize[1] = other.histSize[1];
    c1range[0] = other.c1range[0];
    c1range[1] = other.c1range[1];
    c2range[0] = other.c2range[0];
    c2range[1] = other.c2range[1];
    other.accumulator.copyTo(accumulator);
    other.normalized.copyTo(normalized);
    gmmReady = other.gmmReady;
    gmm = other.gmm;
}

Histogram& Histogram::operator=(const Histogram& other){
    if (this != &other){
        channels[0] = other.channels[0];
        channels[1] = other.channels[1];
        histSize[0] = other.histSize[0];
        histSize[1] = other.histSize[1];
        c1range[0] = other.c1range[0];
        c1range[1] = other.c1range[1];
        c2range[0] = other.c2range[0];
        c2range[1] = other.c2range[1];
        other.accumulator.copyTo(accumulator);
        other.normalized.copyTo(normalized);
        gmmReady = other.gmmReady;
        gmm = other.gmm;
    }
    return *this;
}

void Histogram::fromImage(Mat image, const Mat mask = Mat()){
    const float* ranges[] = {c1range, c2range};
    calcHist(&image, 1, channels, mask, accumulator, 2, histSize, ranges, true, false);

    double histMax = 0;
    double histMin = 0;
    minMaxLoc(accumulator, &histMin, &histMax, NULL, NULL);
    accumulator.convertTo(normalized,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
}

void Histogram::update(Mat image, double alpha, const Mat mask = Mat()){
    const float* ranges[] = {c1range, c2range};
    Mat temp;
    accumulator *= (1-alpha);
    calcHist(&image, 1, channels, mask, temp, 2, histSize, ranges, true, true);


    double histMax = 0;
    double histMin = 0;
    minMaxLoc(accumulator, &histMin, &histMax, NULL, NULL);
    accumulator.convertTo(normalized,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
}

void Histogram::backPropagate(Mat inputImage, Mat* outputImage){
    const float* ranges[] = {c1range, c2range};
    calcBackProject(&inputImage, 1, channels, normalized, *outputImage, ranges, 1, true);
    outputImage->convertTo(*outputImage, CV_32FC1);
}

void Histogram::makeGMM(int K, int maxIter = 10, double minStepIncrease = 0.01){
    gmm = GaussianMixtureModel(2,K);
    accumulator.convertTo(normalized, CV_64F);
    gmm.fromHistogram(normalized, histSize, c1range, c2range, maxIter, minStepIncrease);
    gmmReady = true;
    
    Mat hist = gmm.lookup;
    double histMax = 0;
    double histMin = 0;
    minMaxLoc(hist, &histMin, &histMax, NULL, NULL);
    hist.convertTo(normalized,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
}

void Histogram::resize(int histogramSize[2]){
    histSize[0] = histogramSize[0];
    histSize[1] = histogramSize[1];
}


GaussianMixtureModel::GaussianMixtureModel(){
    initialized = false;
}

GaussianMixtureModel::GaussianMixtureModel(int dims, int K){
    dimensions=dims;
    components=K;
    for (int i=0; i<K; i++){
        covarianceMatrix.push_back(Mat::eye(dims, dims, CV_64F)*500);
        if (i==0){
            meanVector.push_back(Mat::ones(dims, 1, CV_64F)*100);
        }
        else {
            meanVector.push_back(Mat::ones(dims, 1, CV_64F)*50);
        }
        weight.push_back(1.0/K);
    }
    initialized = false;
}

GaussianMixtureModel::GaussianMixtureModel(const GaussianMixtureModel& other){
    dimensions = other.dimensions;
    components = other.components;
    weight = other.weight;
    initialized = other.initialized;
    covarianceMatrix.clear();
    for(int i=0; i<other.covarianceMatrix.size(); i++){
        Mat temp;
        other.covarianceMatrix[i].copyTo(temp);
        covarianceMatrix.push_back(temp);
    }
    meanVector.clear();
    for(int i=0; i<other.meanVector.size(); i++){
        Mat temp;
        other.meanVector[i].copyTo(temp);
        meanVector.push_back(temp);
    }
    other.componentProbability.copyTo(componentProbability);
    other.lookup.copyTo(lookup);
}

GaussianMixtureModel& GaussianMixtureModel::operator=(const GaussianMixtureModel& other){
    if(this != &other){
        dimensions = other.dimensions;
        components = other.components;
        weight = other.weight;
        initialized = other.initialized;
        covarianceMatrix.clear();
        for(int i=0; i<other.covarianceMatrix.size(); i++){
            Mat temp;
            other.covarianceMatrix[i].copyTo(temp);
            covarianceMatrix.push_back(temp);
        }
        meanVector.clear();
        for(int i=0; i<other.meanVector.size(); i++){
            Mat temp;
            other.meanVector[i].copyTo(temp);
            meanVector.push_back(temp);
        }
        other.componentProbability.copyTo(componentProbability);
        other.lookup.copyTo(lookup);
    }
    
    return *this;
}

/* matrix samples is a N X (M+1) matrix, consisting of N M-dimensional samples. The last row-element is the number of identical samples*/
void GaussianMixtureModel::runExpectationMaximization(const Mat samples, int maxIterations, double minStepIncrease){
    if (true){
        //componentProbability = Mat::ones(samples.rows, components, CV_64F)/(1.0*components);
        initialized = true;

        int k = 0;
        int numEach = samples.rows / components;
        double num = 0;
        Mat tempVec = Mat::zeros(dimensions, 1, CV_64F);
        meanVector.clear();
        for (int i=0; i<samples.rows; i++){
            double element = samples.at<double>(i,dimensions);
            Mat elementx = samples(Range(i,i+1),Range(0,dimensions));
            elementx = elementx.t();
            tempVec += element * elementx;
            num += element;
            if (i!=0 && i%numEach==0){
                meanVector.push_back(tempVec/num);
                num=0;
                tempVec = Mat::zeros(dimensions, 1, CV_64F);
                k++;
            }
        }
        meanVector.push_back(tempVec/num);
    }
    
    double increase = 0;
    double lastLogLikelihood = 0;
    bool start = true;
    double nDataPoints = 0;
    for (int i=0; i<samples.rows; i++){
        nDataPoints+=samples.at<double>(i,dimensions);
    }
    
    std::vector<double> newWeight;
    std::vector<Mat> newCovarianceMatrix;
    std::vector<Mat> newMeanVector;
    Mat newComponentProbability;
    
    for (int step = 0; step<maxIterations; step++){

        newCovarianceMatrix.clear();
        newMeanVector.clear();
        newWeight.clear();

        for (int i = 0; i<components; i++){
            newCovarianceMatrix.push_back(Mat::zeros(dimensions, dimensions, CV_64F));
            newMeanVector.push_back(Mat::zeros(dimensions, 1, CV_64F));
            newWeight.push_back(0.0);
        }
        newComponentProbability = Mat::zeros(samples.rows, components+1, CV_64F);

        for (int i = 0; i<samples.rows; i++){
            Mat elementx = samples(Range(i,i+1),Range(0,dimensions));
            elementx = elementx.t();
            for (int k = 0; k<components; k++){
                double prob = weight[k]*gaussIdx(elementx, k);
                newComponentProbability.at<double>(i,k) = prob;
            }
        }

        //std::cout << "Compprob: "<< newComponentProbability << std::endl;


        for (int i = 0; i<samples.rows; i++){
            double sum = 0;
            const double* tempRow = newComponentProbability.ptr<double>(i);
            for (int k = 0; k<components; k++){
                sum += tempRow[k];
            }
            if (sum>1e-300){
                Mat rowMod = newComponentProbability.row(i);
                rowMod /= sum;
            }
            else {
                for (int k = 0; k<components; k++){
                    newComponentProbability.at<double>(i,k)=1.0/components;
                }
            }
        }

        newComponentProbability.copyTo(componentProbability);

        //std::cout << "Compprob: "<< newComponentProbability << std::endl;


        for (int i = 0; i<samples.rows; i++){
            double element = samples.at<double>(i,dimensions);
            Mat elementx = samples(Range(i,i+1),Range(0,dimensions));
            elementx = elementx.t();
            for (int k = 0; k<components; k++){
                double prob = componentProbability.at<double>(i,k);
                newWeight[k] += element * prob;
                newMeanVector[k] += element * elementx * prob;
            }
        }

        for (int k = 0; k<components; k++){
            newMeanVector[k] /= newWeight[k];
            newWeight[k] /= 1.0 * nDataPoints;
        }
        for (int i = 0; i<samples.rows; i++){
            double element = samples.at<double>(i,dimensions);
            Mat elementx = samples(Range(i,i+1),Range(0,dimensions));
            elementx = elementx.t();
            for (int k = 0; k<components; k++){
                double prob = componentProbability.at<double>(i,k);
                Mat diff = (elementx - newMeanVector[k]);
                newCovarianceMatrix[k] += element * prob * diff*diff.t();
            }
        }

        for (int k = 0; k<components; k++){
            newCovarianceMatrix[k] /= 1.0*nDataPoints * newWeight[k];
        }

        for (int k = 0; k<components; k++){
            weight[k] = newWeight[k];
            newMeanVector[k].copyTo(meanVector[k]);
            newCovarianceMatrix[k].copyTo(covarianceMatrix[k]);
        }

        double logLikelihood = 0.0;
        for (int i=0; i<samples.rows; i++){
            int element = samples.at<int>(i,dimensions);
            Mat elementx = samples.rowRange(0, dimensions);
            elementx = elementx.t();
            double prob = get(elementx);
            logLikelihood += element * prob;
        }
        if (!start){
            increase = logLikelihood/lastLogLikelihood;
            if (increase < (1+minStepIncrease)) {break;}
        }
        start = false;
        lastLogLikelihood = logLikelihood;
    }
}

double GaussianMixtureModel::gauss(const Mat x, Mat covMatrix, Mat meanVec){
    if (x.size()!=Size(1,dimensions) || covMatrix.size()!=Size(dimensions,dimensions) || meanVec.size()!=Size(1,dimensions)){
        return -1.0;
    }
    try{
        Mat temp(x.size(),CV_64F);
        temp = (x-meanVec);
        temp = temp.t();
        temp = temp*covMatrix.inv();
        temp = temp*(x-meanVec)/(-2.0);
        double val = temp.at<double>(0,0);
        double retVal = exp(val);
        double pi = 3.141592653589793238463;
        double det = determinant(covMatrix);
        retVal = retVal/(pow(sqrt(2*pi),dimensions)*sqrt(det));
        return retVal;
    } catch(Exception e){std::cout << e.msg << std::endl; return -1;}
}

double GaussianMixtureModel::gaussIdx(const Mat x, int componentIndex){
    if (componentIndex>=0 && componentIndex<components){
        double retVal = gauss(x, covarianceMatrix[componentIndex], meanVector[componentIndex]);
        return retVal;
    }
    else {return -1;}
}

double GaussianMixtureModel::get(Mat x){
    double retVal = 0;
    for (int k=0; k<components; k++){
        double prob = weight[k]*gaussIdx(x, k);
        if (prob<0) {return -1;}
        retVal += prob;
    }
    return retVal;
}

void GaussianMixtureModel::makeLookup(int histSize[2], float c1range[2], float c2range[2]){
    Mat newLookup(histSize[0], histSize[1], CV_64F);
    float dim1step = (c1range[1]-c1range[0])/(1.0*histSize[0]);
    float dim2step = (c2range[1]-c2range[0])/(1.0*histSize[1]);
    float dim1start = c1range[0]+dim1step/2.0;
    float dim2start = c2range[0]+dim2step/2.0;
    for (int i=0; i<histSize[0]; i++){
        for (int j=0; j<histSize[1]; j++){
            double xarr[] = {dim1start+i*dim1step, dim2start+j*dim2step};
            Mat x(2,1,CV_64F, xarr);
            newLookup.at<double>(i,j)=get(x);
        }
    }
    newLookup=newLookup*dim1step*dim2step;
    newLookup.copyTo(lookup);
    double histMax = 0;
    double histMin = 0;
    minMaxLoc(lookup, &histMin, &histMax, NULL, NULL);
    lookup.convertTo(lookup,CV_64F,1/(histMax-histMin),-histMin/(histMax-histMin));
}

void GaussianMixtureModel::fromHistogram(const Mat histogram, int histSize[2], float c1range[2], float c2range[2], int maxIter=10, double minStepIncrease = 0.01){
    float dim1step = (c1range[1]-c1range[0])/(1.0*histSize[0]);
    float dim2step = (c2range[1]-c2range[0])/(1.0*histSize[1]);
    float dim1start = c1range[0]+dim1step/2.0;
    float dim2start = c2range[0]+dim2step/2.0;
    Mat samples;
    for (int i=0; i<histSize[0]; i++){
        for (int j=0; j<histSize[1]; j++){
            if (histogram.at<double>(i,j)>0){
                double xarr[] = {dim1start+i*dim1step, dim2start+j*dim2step, histogram.at<double>(i,j)};
                Mat x(1,3,CV_64F, xarr);
                samples.push_back(x);
            }
        }
    }
    runExpectationMaximization(samples, maxIter, minStepIncrease);
    makeLookup(histSize,c1range,c2range);
}

ColorHistBackProject::ColorHistBackProject(){
    name = "ColorHistBackProject";
    int histSize[2];
    int channels[2];
    float c1range[2];
    float c2range[2];
    histSize[0] = 64;
    histSize[1] = 64;
    colorspaceCode = CV_BGR2HSV;
    c1range[0]=0; c1range[1]=180; c2range[0]=0; c2range[1]=256;
    channels[0]=0; channels[1]=1;
    
    Histogram histTemp = Histogram(channels, histSize, c1range, c2range);
    objHistogram = histTemp;
    
    initialized=false;
}

ColorHistBackProject::ColorHistBackProject(int code, const int* histogramSize){
    name = "ColorHistBackProject";
    int channels[2];
    float c1range[2];
    float c2range[2];
    int histSize[2];
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
        channels[0]=1; channels[1]=2; break;
    case CV_BGR2Lab:
        channels[0]=1; channels[1]=2;
    default:
        break;
    }

    Histogram histTemp(channels, histSize, c1range, c2range);
    objHistogram = histTemp;
    initialized=false;
}

ColorHistBackProject::ColorHistBackProject(int code, const int* histogramSize, String filename){
    name = "ColorHistBackProject";
    int histSize[2];
    int channels[2];
    float c1range[2];
    float c2range[2];
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

    Histogram histTemp = Histogram(channels, histSize, c1range, c2range);
    objHistogram = histTemp;
    Mat img = imread(filename);

    histFromImage(img);
    initialized=true;
}

void ColorHistBackProject::preprocess(const Mat image, Mat* outputImage){
    //GaussianBlur(image, *outputImage, Size(15,15),0);
    //medianBlur(image, *outputImage, 7);

    bilateralFilter(image, *outputImage, 5, 75, 60);
    cvtColor(*outputImage, *outputImage, colorspaceCode);

    Mat temp;
    Scalar lowRange;
    Scalar highRange;

    switch (colorspaceCode){
    case CV_BGR2HSV:
        lowRange = Scalar(0,25,40);
        highRange = Scalar(255,255,255);  break;
    case CV_BGR2HLS:
        lowRange = Scalar(0,40,10);
        highRange = Scalar(255,220,255); break;
    case CV_BGR2YUV:
        lowRange = Scalar(25,0,0);
        highRange = Scalar(230,255,255);  break;
    case CV_BGR2Lab:
        lowRange = Scalar(25,0,0); highRange = Scalar(230,255,255);  break;
    default:
        break;
    }

    //inRange(*outputImage, lowRange, highRange, histogramMask);

    outputImage->convertTo(*outputImage, CV_32F);

    //medianBlur(*outputImage, *outputImage, 5);
    //blur(*outputImage, *outputImage, Size(5,5));
}

void ColorHistBackProject::histFromImage(const Mat image){
    Mat cvtImage;
    preprocess(image, &cvtImage);

    objHistogram.fromImage(cvtImage, histogramMask);
    initialized=true;
};

void ColorHistBackProject::updateHistogram(const Mat image, const Mat mask){
    Mat cvtImage;
    preprocess(image, &cvtImage);

    Mat andMask;
    bitwise_and(histogramMask,mask, andMask);
    objHistogram.update(image, 0.1, andMask);

}

void ColorHistBackProject::process(const Mat inputImage, Mat* outputImage){
    Mat cvtImage;
    preprocess(inputImage, &cvtImage);
    objHistogram.backPropagate(cvtImage, outputImage);
}




void BayesColorHistBackProject::process(const Mat inputImage, Mat* outputImage){
    name = "BayesColorHistBackProject";
    Mat cvtImage;
    preprocess(inputImage, &cvtImage);

    objHistogram.backPropagate(cvtImage, outputImage);
    /*
    Histogram imgHist = Histogram(objHistogram);
    imgHist.fromImage(cvtImage);

    Mat aprioriColor;
    imgHist.backPropagate(cvtImage, &aprioriColor);

    *outputImage = *outputImage/aprioriColor;

    double histMax = 0;
    double histMin = 0;
    minMaxLoc(*outputImage, &histMin, &histMax, NULL, NULL);

    outputImage->convertTo(*outputImage,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));
    */
    outputImage->convertTo(*outputImage, CV_32F);
    //aprioriColor.copyTo(*outputImage);

    return;
}

void BayesColorHistBackProject::histFromImage(const Mat image){
    Mat cvtImage;
    preprocess(image, &cvtImage);

    objHistogram.fromImage(cvtImage);
    objHistogram.makeGMM(2,4);

    initialized=true;
}


GMMColorHistBackProject::GMMColorHistBackProject() : ColorHistBackProject(){
    name = "GMMColorHistBackProject";
}

GMMColorHistBackProject::GMMColorHistBackProject(int code, const int* histogramSize) : ColorHistBackProject(code, histogramSize){
    name = "GMMColorHistBackProject";
}

GMMColorHistBackProject::GMMColorHistBackProject(int code, const int* histogramSize, String filename) : ColorHistBackProject(code, histogramSize,filename){
    name = "GMMColorHistBackProject";
}

void GMMColorHistBackProject::process(const Mat inputImage, Mat* outputImage){
    Mat cvtImage;
    preprocess(inputImage, &cvtImage);

    objHistogram.backPropagate(cvtImage, outputImage);

    Histogram imgHist = Histogram(objHistogram);
    int size[2] = {16,16};
    imgHist.resize(size);
    imgHist.fromImage(cvtImage);

    medianBlur(*outputImage, *outputImage, 5);
    Mat aprioriColor;
    imgHist.backPropagate(cvtImage, &aprioriColor);
    medianBlur(aprioriColor, aprioriColor, 5);
    *outputImage = *outputImage/aprioriColor;
    double histMax = 0;
    double histMin = 0;
    minMaxLoc(*outputImage, &histMin, &histMax, NULL, NULL);
    outputImage->convertTo(*outputImage,CV_32F,1/(histMax-histMin),-histMin/(histMax-histMin));



    //outputImage->convertTo(*outputImage, CV_32FC1);
    //outputImage->convertTo(*outputImage, CV_64F);
}

void GMMColorHistBackProject::histFromImage(const Mat image){
    Mat cvtImage;
    preprocess(image, &cvtImage);

    objHistogram.fromImage(cvtImage);
    objHistogram.makeGMM(2,4);

    initialized = true;
}



SimpleThresholder::SimpleThresholder(){
    name = "SimpleThresholder";
    thresholdValue = 0.5;
    initialized = true;
}

SimpleThresholder::SimpleThresholder(float threshValue){
    name = "SimpleThresholder";
    thresholdValue = threshValue;
    initialized = true;
}

void SimpleThresholder::process(const Mat inputImage, Mat* outputImage){
    //threshold(*outputImage, *outputImage, thresholdValue, 1.0, THRESH_BINARY);
    
    inputImage.convertTo(*outputImage, CV_8U,255,0);
    
    threshold(*outputImage, *outputImage, 100, 255, THRESH_BINARY);
    //adaptiveThreshold(*outputImage, *outputImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 0.0);
    
    
    outputImage->convertTo(*outputImage, CV_32F, 1/255.0,0.0);
    
    /*Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
    erode(*outputImage, *outputImage, element);
    dilate(*outputImage, *outputImage, element);*/
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7,7));
    dilate(*outputImage, *outputImage, element);
    erode(*outputImage, *outputImage, element);
}




SimpleBlobDetect::SimpleBlobDetect(){
    name = "SimpleBlobDetect";
    initialized = true;
}

void SimpleBlobDetect::process(const Mat inputImage, Mat* outputImage){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat conv(inputImage.size(), CV_8U);
    inputImage.convertTo(conv, CV_8U);
    findContours(conv, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    Mat temp = Mat::zeros(inputImage.size(), CV_8UC3);
    double area = contourArea(contours[0], false);
    double maxarea = area;
    double minarea = area;
    for( int i = 1; i< contours.size(); i++ ){
        area = contourArea(contours[i], false);
        if (area > maxarea){maxarea = area;}
        if (area < minarea){minarea = area;}
    }
    
    
    for( int i = 0; i< contours.size(); i++ ){
        Scalar color = Scalar(0, 0, 255);
        switch (i%6){
        case 0: color = Scalar(0,0,255); break;
        case 1: color = Scalar(0,255,255); break;
        case 2: color = Scalar(0,255,0); break;
        case 3: color = Scalar(255,255,0); break;
        case 4: color = Scalar(255,0,0); break;
        case 5: color = Scalar(255,0,255); break;
        }
        area = contourArea(contours[i], false);
        if ((area-minarea)/(maxarea-minarea)>0.5){
            drawContours(temp, contours, i, color, 2, 8);
        }
    }
    temp.copyTo(*outputImage);
}

OpticalFlow::OpticalFlow(){
    name = "OpticalFlow";
    init = false;
}

void OpticalFlow::process(const Mat inputImage, Mat *outputImage){
    Mat imbw;
    cvtColor(inputImage, imbw, CV_BGR2GRAY);
    blur(imbw, imbw, Size(7,7));
    if (!init){
        init=true;
        imbw.copyTo(old);
        imbw.copyTo(*outputImage);
        return;
    }
    Mat of;
    calcOpticalFlowFarneback(old,imbw, of, 0.5, 3, 15, 3, 7, 1.5,0);
    Mat dxdy[2];
    split(of,dxdy);
    Mat moveMag = abs(dxdy[0])+abs(dxdy[1]);
    moveMag.convertTo(*outputImage,CV_32F,1.0,0);
    imbw.copyTo(old);
}

BGSubtractor::BGSubtractor(){
    name = "BGSubtractor";
    bgsub = BackgroundSubtractorMOG(4,3,0.4);
    init = false;
}

void BGSubtractor::process(const Mat inputImage, Mat *outputImage){
    Mat imbw;
    inputImage.copyTo(imbw);
    //cvtColor(inputImage, imbw, CV_BGR2GRAY);
    inputImage.copyTo(imbw);
    blur(imbw, imbw, Size(7,7));
    bgsub(imbw, *outputImage, 0.01);
}


