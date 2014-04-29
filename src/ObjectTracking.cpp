#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "ObjectTracking.hpp"
#include "boost/smart_ptr.hpp"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include <boost/thread/thread_time.hpp>
#include <cmath>
#include <ctime>
#include <cstdlib>

#ifdef TESTMODE
#define VISUALDEBUG true
#else
#define VISUALDEBUG false
#endif

using namespace cv;
using namespace std;

namespace fs = boost::filesystem;

UpdatableHistogram::UpdatableHistogram(): Histogram(), buffersize(0){}

UpdatableHistogram::UpdatableHistogram(int channels[], int histogramSize[], float channel1range[], float channel2range[], int bufferSize):
    Histogram(channels, histogramSize, channel1range, channel2range),
    buffersize(bufferSize)
{}

void UpdatableHistogram::update(Mat image, double alpha, const Mat mask){
    const float* ranges[] = {c1range, c2range};
    Mat colorHist;
    calcHist(&image, 1, channels, mask, colorHist, 2, histSize, ranges, true, false);

    Mat apriori;
    calcHist(&image, 1, channels, Mat(), apriori, 2, histSize, ranges, true, false);

    apriori.convertTo(apriori, CV_32F);
    colorHist.convertTo(colorHist, CV_32F);

    Mat aposteriori = colorHist/apriori;

    if(buffer.size()>=buffersize){
        buffer.erase(buffer.begin());
    }
    buffer.push_back(aposteriori);

    int full = 1;
    for (int i=0; i<buffer.size()-1; i++){
        double minVal = 0;
        double maxVal = 0;
        minMaxLoc(buffer[i], &minVal, &maxVal);
        if (maxVal>1e-6){
            aposteriori+=buffer[i];
            full++;
        }
    }
    if (full>1){
        aposteriori/=full;
    }

    aposteriori = alpha*offline + (1-alpha)*aposteriori;
}

void UpdatableHistogram::fromImage(const vector<Mat> image, const vector<Mat> mask){
    const float* ranges[] = {c1range, c2range};
    Mat aposteriori;
    Mat colorHist;
    Mat apriori;
    int numImages = min(image.size(), mask.size());
    for (int i=0; i<numImages; i++){
        calcHist(&image[i], 1, channels, mask[i], colorHist, 2, histSize, ranges, true, true);
        calcHist(&image[i], 1, channels, Mat(), apriori, 2, histSize, ranges, true, true);
    }

    apriori.convertTo(apriori, CV_32F);
    colorHist.convertTo(colorHist, CV_32F);
    aposteriori = colorHist/apriori;

    aposteriori.copyTo(offline);
    aposteriori.copyTo(normalized);
    aposteriori.copyTo(accumulator);
    makeGMM(3,20,0.001);
}

TrackedObject::TrackedObject(){
    tracked = false;
}

TrackedObject::TrackedObject(const Mat image, const vector<Point> inContour, bool isContour = false){
    if (inContour.size()<5) {tracked = false; return;}
    tracked = true;
    imageSize = image.size();

    if (isContour){
        contour = inContour;
        points.clear();
    }
    else {
        contour.clear();
        points = inContour;
        if (VISUALDEBUG){
            Mat temp(Mat::zeros(image.size(), CV_8U));
            for (int i=0; i<points.size(); i++){
                temp.at<uchar>(points[i])=255;
            }
            vector<vector<Point> > contours;
            findContours(temp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
            contour = contours[0];
        }
    }
    ellipse = getEllipse();//minAreaRect(inContour);
    actualEllipse = ellipse;
    updateArea();
    timeLost = boost::get_system_time();
    occluded = false;
    estMove = Point2f(0,0);
    kind = -1;
    id = -1;
}


void TrackedObject::update(const Mat image, const vector<Point> inContour, bool isContour = false){
    if (inContour.size()<5) {tracked = false; return;}
    tracked = true;
    imageSize = image.size();
    if (isContour){
        points.clear();
        contour = inContour;
    }
    else {
        contour.clear();
        points = inContour;
        if (VISUALDEBUG){
            Mat temp(Mat::zeros(image.size(), CV_8U));
            for (int i=0; i<points.size(); i++){
                temp.at<uchar>(points[i])=255;
            }
            vector<vector<Point> > contours;
            findContours(temp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE
                         );
            int maxsize = contours[0].size();
            int best = 0;
            for (int i=1; i<contours.size(); i++){
                if (contours[i].size()>maxsize){
                    maxsize = contours[i].size();
                    best = i;
                }
            }
            contour = contours[best];
        }
    }
    RotatedRect newEllipse = getEllipse(); //minAreaRect(inContour);
    estMove = newEllipse.center-actualEllipse.center;
    actualEllipse = newEllipse;
    newEllipse.center += estMove;
    ellipse = newEllipse;
}

void TrackedObject::updateArea(){
    if (points.size()>0){
        area = points.size();
    }
    else{
        area = ellipse.size.area();
    }
}

vector<Point> TrackedObject::pointsFromContour(){
    Mat temp = Mat::zeros(imageSize, CV_8U);
    vector<vector<Point>> conts;
    points.clear();
    conts.push_back(contour);
    drawContours(temp, conts, 0, Scalar(255), CV_FILLED);
    for (int i=0; i<temp.rows; i++){
	for (int j=0; j<temp.cols; j++){
	    if (temp.at<int>(i,j)){
		points.push_back(Point(i,j));
	    }
	}
    }
    return points;
}
  
RotatedRect TrackedObject::useCamShift(const Mat probImage){
    //double size = min(ellipse.size.height, ellipse.size.width);
    //Point tl(ellipse.center.x-size/2, ellipse.center.y-size/2);
    //Rect box(tl, Size(size,size));
    Rect box = ellipse.boundingRect();
    return CamShift(probImage, box, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
}
  
double TrackedObject::compare(boost::shared_ptr<TrackedObject> otherObject){
    if (intersectingOBB(ellipse, otherObject->ellipse)){
        return 0;
    }
    else {
        return distRotatedRect(ellipse, otherObject->ellipse);
    }
}

double TrackedObject::getAreaRatio(double compareArea=-1){
    if (compareArea<=0){
        compareArea = area;
    }
    if(points.size()>0){
        return points.size()/compareArea;
    }
    else {
        return actualEllipse.size.area()/compareArea;
    }
}


double TrackedObject::getArea(){
    if(points.size()>0){
        return points.size();
    }
    else {
        return actualEllipse.size.area();
    }
}

RotatedRect TrackedObject::getEllipse(){
    if (points.size()==0){
        return RotatedRect();
    }
    Point2f centroid(0,0);
    for (int i=0; i<points.size(); i++){
        centroid.x+=points[i].x;
        centroid.y+=points[i].y;
    }
    centroid.x/=points.size();
    centroid.y/=points.size();
    float mxx=0;
    float mxy=0;
    float myy=0;
    for (int i=0; i<points.size(); i++){
        Point2f pt = points[i];
        float dx = pt.x-centroid.x;
        float dy = pt.y-centroid.y;
        mxx+=dx*dx;
        mxy+=dx*dy;
        myy+=dy*dy;
    }
    mxx/=points.size();
    myy/=points.size();
    mxy/=points.size();

    float K = sqrt(pow(mxx+myy,2)-4*(mxx*myy-pow(mxy,2)));
    RotatedRect temp;
    temp.center = centroid;
    Size stemp;
    float l1 = (mxx+myy+K)/2;
    float l2 = (mxx+myy-K)/2;
    stemp.width = 2*sqrt(l1)*1.5;
    stemp.height = 2*sqrt(l2)*1.5;
    temp.size = stemp;
    temp.angle = atan2(mxx-l1, -mxy)*180.0f/3.141592653589f;
    return temp;
}

void TrackedObject::unOcclude(){
    occluded = false;
    for (int i=0; i<occluders.size(); i++){
        for (int j=0; j<occluders[i]->occluding.size(); j++){
            if (occluders[i]->occluding[j].get() == this){
                occluders[i]->occluding.erase(occluders[i]->occluding.begin()+j);
                break;
            }
        }
    }
    occluders.clear();
}




ObjectTracker::ObjectTracker(){    
    frameNumber = 0;
    nextObjectIdx = 0;
}

void ObjectTracker::preprocess(const Mat image, Mat& outputImage, Mat& mask){
    Mat procimg;
    blur(image, procimg, Size(5,5));
    cvtColor(procimg, procimg, CV_BGR2YCrCb);
    Scalar lowRange = Scalar(40,0,0);
    Scalar highRange = Scalar(215,255,255);
    inRange(procimg, lowRange, highRange, mask);
    procimg.convertTo(outputImage, CV_32F);
}

void ObjectTracker::addObjectKind(const vector<Mat> image, const vector<Mat> outMask){
    vector<Mat> procimg;
    vector<Mat> mask;
    int numImg = min(image.size(), outMask.size());
    for(int i=0; i<numImg; i++){
        Mat temp;
        Mat tempmask;
        preprocess(image[i], temp, tempmask);
        bitwise_and(tempmask, outMask[i], tempmask);
        procimg.push_back(temp);
        mask.push_back(tempmask);
    }
    int channels[2] = {1,2};
    float c1range[2] = {0,256};
    float c2range[2] = {0,256};

    int histSize[2] = {64,64};
    UpdatableHistogram objHist(channels, histSize, c1range, c2range, 5);

    objHist.fromImage(procimg, mask);
    objectKinds.push_back(objHist);
    Mat temp;
    resize(objectKinds.back().normalized, temp, Size(300,300),0,0, INTER_NEAREST);
    //objectKinds.back().makeGMM(3,4,0.01);
}

void ObjectTracker::getProbImages(const Mat procimg, const Mat mask, vector<Mat> &outputImages){
    //first, get the general image histogram and use it to get a normalized probability image of the input image
    double histMax = 0;
    double histMin = 0;
    outputImages.clear();
    for (int i=0; i<objectKinds.size(); i++){
        Mat objProb;
        objectKinds[i].backPropagate(procimg, &objProb);
        outputImages.push_back(objProb);
    }
}


void ObjectTracker::process(const Mat inputImage, Mat* outputImage){
    double minimumAreaCutoff = inputImage.size().area()/225.0;
    double closeDistance = 20.0;
    double occludedLow = 0.3;
    double occludedHigh = 0.6;
    Mat drawImg;
    if (VISUALDEBUG){
        inputImage.copyTo(drawImg);
    }

    vector<Mat> probImages;
    vector<Mat> binImages;
    Mat binImg;
    if (VISUALDEBUG){
        Mat temp(Mat::zeros(inputImage.size(), CV_8U));
        temp.copyTo(binImg);
    }
    Mat procimg;
    Mat mask;
    preprocess(inputImage, procimg, mask);
    getProbImages(procimg, mask, probImages);

    vector<vector<Point2i>> blobs;
    vector<int> blobKinds;
    for (int i=0; i<probImages.size(); i++){
        //binarize the probability image
        Mat temp;
        vector<vector<Point2i>> tempBlobs;
        hysteresisThreshold(probImages[i], temp, tempBlobs, 0.15, 0.5);
        objectKinds[i].update(procimg, 0.8, temp);
        for (int j=0; j<tempBlobs.size(); j++){
            blobs.push_back(tempBlobs[j]);
            blobKinds.push_back(i);
        }
        if (VISUALDEBUG){
            binImages.push_back(temp);
            bitwise_or(temp, binImg, binImg);
        }
    }

    for (int k=0; k<blobs.size(); k++){
        double area = blobs[k].size();
        if (area<minimumAreaCutoff){
            blobs.erase(blobs.begin()+k);
            k--;
        }
    }
        /* or use simple 2-means clustering to extract only larger blobs
        if (blobs.size()>4){
            double maxArea = blobs[0].size();
            double minArea = blobs[0].size();
            for (int k=1; k<blobs.size(); k++){
                double area = blobs[k].size();
                if (area>maxArea){maxArea=area;}
                if (area<minArea){minArea=area;}
            }
            double pivot = minArea + 0.4*(maxArea-minArea);
            for (int k=0; k<blobs.size(); k++){
                double area = blobs[k].size();
                if (area<pivot){
                    blobs.erase(blobs.begin()+k);
                    k--;
                }
            }
        }
        */

    int objKeys[objects.size()];
    int tObjIdx = 0;
    for (objMap::iterator it=objects.begin(); it!=objects.end(); ++it){
        it->second->tracked = false;
        objKeys[tObjIdx] = it->first;
        tObjIdx++;
    }
     /*
    for (int j=0; j<objects.size(); j++){
        objects[j]->tracked = false;
    }*/

    int supportPoints[objects.size()][blobs.size()];
    int blobsobject[objects.size()];

    vector<vector<Point2i> > blobsForObjects;
    for (int i=0; i<objects.size(); i++){
        vector<Point2i> temp;
        blobsobject[i] = -1;
        blobsForObjects.push_back(temp);
        for (int j=0; j<blobs.size(); j++){
            supportPoints[i][j] = 0;
        }
    }


    vector<vector<int> > objectsblob;
    for (int i=0; i<blobs.size(); i++){
        vector<int> temp;
        objectsblob.push_back(temp);
        for (int j=0; j<blobs[i].size(); j++){
            Point2i pt = blobs[i][j];
            for (int k=0; k<objects.size(); k++){
                double dist = distEllipse2Point(objects[objKeys[k]]->ellipse, pt);
                if (dist<1.0){
                    supportPoints[k][i]+=1;
                }
            }
        }
    }

    for (int i=0; i<blobs.size(); i++){
        bool singleSupport = true;
        int support = -1;
        for (int j=0; j<objects.size(); j++){
            if (blobsobject[j]== -1 && supportPoints[j][i]>0){
                if(support==-1){
                    support = j;
                }
                else {
                    singleSupport = false;
                    break;
                }
            }
        }
        if (singleSupport && support!=-1){
            blobsobject[support]=i;
            objectsblob[i].push_back(support);
        }
    }


    for (int j=0; j<objects.size(); j++){
        if (blobsobject[j] == -1){
            int mostSupport = 0;
            int best = -1;
            for (int i=0; i<blobs.size(); i++){
                if (supportPoints[j][i]>mostSupport){
                    mostSupport = supportPoints[j][i];
                    best = i;
                }
            }
            if (best!=-1){
                blobsobject[j]=best;
                objectsblob[best].push_back(j);
            }
        }
    }

    vector<int> newBlobs;
    for (int i=0; i<blobs.size(); i++){
        if (objectsblob[i].size()>0){
            for (int j=0; j<blobs[i].size(); j++){
                Point2i pt = blobs[i][j];
                bool claimed = false;
                double distList[objectsblob[i].size()];
                for (int k=0; k<objectsblob[i].size(); k++){
                    int idx = objectsblob[i][k];
                    int key = objKeys[idx];
                    distList[k] = distEllipse2Point(objects[key]->ellipse, pt);
                    if (distList[k]<1.0){
                        claimed = true;
                        blobsForObjects[idx].push_back(pt);
                    }
                }
                if (!claimed && objectsblob[i].size()>0){
                    int best = objectsblob[i][0];
                    double closest = distList[0];
                    for (int k=1; k<objectsblob[i].size(); k++){
                        int idx = objectsblob[i][k];
                        if (closest>distList[k]){
                            closest = distList[k];
                            best = idx;
                        }
                    }
                    blobsForObjects[best].push_back(pt);
                }
            }
        }
        else {
            newBlobs.push_back(i);
        }
    }

    for (int i=0; i<objects.size(); i++){
        if (blobsobject[i]!=-1){
            objects[objKeys[i]]->update(inputImage, blobsForObjects[i]);
        }
    }

    for (int i=0; i<newBlobs.size(); i++){
        boost::shared_ptr<TrackedObject> temp(new TrackedObject(procimg, blobs[newBlobs[i]]));
        temp->kind = blobKinds[newBlobs[i]];
        int id = nextObjectIdx++;
        temp->id = id;
        int ctmp = (temp->id*21)%51 *10;
        Scalar color(ctmp>255?0:255-ctmp, ctmp>255?512-ctmp:ctmp, ctmp>255?ctmp-255:0);
        temp->color = color;
        objects[id] = temp;
    }

    /* cool multichannel access
    if (objects.size()>0){
        for (int i=0; i<drawImg.size().width; i++){
            for (int j=0; j<drawImg.size().height; j++){
                Point2i pt(i,j);
                if (distEllipse2Point(objects[0]->ellipse, pt)<1.0){
                    drawImg.at<Vec3b>(pt)=Vec3b(0,0,255);
                }
            }
        }
    }*/


    if (VISUALDEBUG){
        for(objMap::iterator it=objects.begin(); it!=objects.end(); ++it){
            vector<vector<Point> > contours;
            boost::shared_ptr<TrackedObject> obj = it->second;
            if (obj->contour.size()>0){
                contours.push_back(obj->contour);
                drawContours(drawImg, contours, 0, obj->color, 2);
                ellipse(drawImg, obj->ellipse, obj->color, 1);
                string id = to_string(obj->id);
                Point shifted = obj->ellipse.center;
                shifted.x += -4;
                shifted.y += 4;
                putText(drawImg, id, shifted, FONT_HERSHEY_SIMPLEX, 0.7, obj->color, 2);
            }
        }
    }
    /*
    for (int i=0; i<objects.size(); i++){
        boost::shared_ptr<TrackedObject> obj = objects[i];
        Size temp = obj->ellipse.size;
        if (temp.width>0 && temp.height>0 && temp.area()>100){
            if (!obj->occluded){
                int ctmp = obj->id;
                Scalar color(ctmp>255?0:255-ctmp, ctmp>255?512-ctmp:ctmp, ctmp>255?ctmp-255:0);
                ellipse(drawImg, obj->ellipse, color, 3);
                for (int j=0; j<obj->occluding.size()>0; j++){
                    RotatedRect tempEl = obj->ellipse;
                    tempEl.size.width -= 4*(j+1);
                    tempEl.size.height -= 4*(j+1);
                    int tmpctmp = obj->occluding[j]->id;
                    Scalar tmpcolor(tmpctmp>255?0:255-tmpctmp, tmpctmp>255?512-tmpctmp:tmpctmp, tmpctmp>255?tmpctmp-255:0);
                    ellipse(drawImg, tempEl, tmpcolor, 2);
                }
            }
            else {
                if (obj->tracked){
                    int ctmp = obj->id;
                    Scalar color(ctmp>255?0:255-ctmp, ctmp>255?512-ctmp:ctmp, ctmp>255?ctmp-255:0);
                    ellipse(drawImg, obj->ellipse, color, 1);
                } else {
                    obj->tracked = true;
                }
            }
        }
    }
    */


   vector<int> deleteKeys;

   for(objMap::iterator it=objects.begin(); it!=objects.end(); ++it){
        boost::shared_ptr<TrackedObject> obj = it->second;
        boost::system_time timenow = boost::get_system_time();
        if (obj->tracked){
            obj->timeLost = timenow;
        }
        else {
            boost::posix_time::time_duration duration = timenow-obj->timeLost;
            if (duration.total_milliseconds() > 300){
                deleteKeys.push_back(it->first);
            }
        }
    }

    for (int i=0; i<deleteKeys.size(); i++){
        objects.erase(deleteKeys[i]);
    }

    drawImg.copyTo(*outputImage);
    
    frameNumber++;
}


//separating axis theorem implementation
bool intersectingOBB(RotatedRect obb1, RotatedRect obb2){
    double radAng1 = -obb1.angle/180.0*3.1415927;
    double radAng2 = -obb2.angle/180.0*3.1415927;
    Mat rot1 = (Mat_<float>(2,2) << cos(radAng1), -sin(radAng1), sin(radAng1), cos(radAng1));
    Mat rot2 = (Mat_<float>(2,2) << cos(radAng2), -sin(radAng2), sin(radAng2), cos(radAng2));
    
    Point2f vertices1[4];
    obb1.points(vertices1);
    Point2f vertices2[4];
    obb2.points(vertices2);
    
    Mat points1(2,4,CV_32F);
    Mat points2(2,4,CV_32F);
    
    for (int i=0; i<4; i++){
	points1.at<float>(0,i)=vertices1[i].x;
	points1.at<float>(1,i)=vertices1[i].y;
	points2.at<float>(0,i)=vertices2[i].x;
	points2.at<float>(1,i)=vertices2[i].y;
    }
    
    Mat pt1rot = rot1*points1;
    Mat pt2rot = rot1*points2;
    double minx1 = 0;
    double maxx1 = 0;
    double minx2 = 0;
    double maxx2 = 0;
    double miny1 = 0;
    double maxy1 = 0;
    double miny2 = 0;
    double maxy2 = 0;
    minMaxLoc(pt1rot.row(0), &minx1, &maxx1);
    minMaxLoc(pt2rot.row(0), &minx2, &maxx2);
    minMaxLoc(pt1rot.row(1), &miny1, &maxy1);
    minMaxLoc(pt2rot.row(1), &miny2, &maxy2);
    
    if (minx1>maxx2 || minx2>maxx1 || miny1>maxy2 || miny2>maxy1){
	return false;
    }
    
    pt1rot = rot2*points1;
    pt2rot = rot2*points2;
    minMaxLoc(pt1rot.row(0), &minx1, &maxx1);
    minMaxLoc(pt2rot.row(0), &minx2, &maxx2);
    minMaxLoc(pt1rot.row(1), &miny1, &maxy1);
    minMaxLoc(pt2rot.row(1), &miny2, &maxy2);
    
    if (minx1>maxx2 || minx2>maxx1 || miny1>maxy2 || miny2>maxy1){
	return false;
    }
    
    return true;
}

double distEllipse2Point(RotatedRect ellipse, Point2f pt){
    Point2f ptshift = pt-ellipse.center;
    Point2f ptRot(0,0);
    double ang = - ellipse.angle / 180.0 *  3.141592653589;
    ptRot.x = (cos(ang)*ptshift.x - sin(ang)*ptshift.y)/(ellipse.size.width/2.0);
    ptRot.y = (cos(ang)*ptshift.y + sin(ang)*ptshift.x)/(ellipse.size.height/2.0);

    double distsq = pow(ptRot.x,2)+pow(ptRot.y,2);
    return distsq;
}

void hysteresisThreshold(const cv::Mat inputImg, cv::Mat& binary, std::vector < std::vector<cv::Point2i> > &blobs, double lowThresh, double hiThresh){
    int label_count = 2;

    Mat probImg;
    inputImg.copyTo(probImg);
    for(int y=0; y < probImg.rows; y++) {
        const float *row = probImg.ptr<float>(y);
        for(int x=0; x < probImg.cols; x++) {
            if(row[x] > 1 || row [x] < hiThresh) {
                continue;
            }

            float ptVal = row[x];
            cv::Rect rect;

            int flags = 4 + FLOODFILL_FIXED_RANGE;
            floodFill(probImg, cv::Point(x,y), label_count, &rect, ptVal-lowThresh, 1-ptVal, flags);

            label_count++;
        }
    }

    blobs.clear();
    for (int i=0; i<label_count-2; i++){
        vector<Point2i> temp;
        blobs.push_back(temp);
    }

    Mat temp(Mat::zeros(probImg.size(), CV_8UC1));

    for(int y=0; y < probImg.rows; y++) {
        const float *row = probImg.ptr<float>(y);
        for(int x=0; x < probImg.cols; x++) {
            if(row[x] < 2) {
                continue;
            }

            int ptVal = round(row[x]);
            Point2i pt(x,y);
            blobs[ptVal-2].push_back(pt);
            temp.at<unsigned char>(pt)=255;
        }
    }

    temp.copyTo(binary);

}

double distLine2Point(Point2d pt1, Point2d pt2, Point2d pt3){
    double alpha = -((pt1.x-pt3.x)*(pt2.x-pt1.x)+(pt1.y-pt3.y)*(pt2.y-pt1.y))/(pow(pt2.x-pt1.x,2)+pow(pt2.y-pt1.y,2));
    if (alpha<0){
        return norm(pt1-pt3);
    }
    if (alpha>1){
        return norm(pt2-pt3);
    }
    return norm(pt1+alpha*(pt2-pt1)-pt3);
}

double distRotatedRect(RotatedRect r1, RotatedRect r2){
    Point2f points1[4];
    Point2f points2[4];
    r1.points(points1);
    r2.points(points2);

    int closest1 = 0;
    int closest2 = 0;
    double mindist = norm(points1[0]-points2[0]);

    for (int i=0; i<4; i++){
        Point2f pt1 = points1[i];
        Point2f pt2 = points1[(i+1)%4];
        for (int j=0; j<4; j++){
            Point2f pt3 = points2[j];
            Point2f pt4 = points2[(j+1)%4];
            double dist = distLine2Point(pt1, pt2, pt3);
            if (dist<mindist) {mindist = dist;}
            dist = distLine2Point(pt3, pt4, pt1);
            if (dist<mindist) {mindist = dist;}
        }
    }

    return mindist;
}


void occludeBy(boost::shared_ptr<TrackedObject> underObject, boost::shared_ptr<TrackedObject> overObject){
    underObject->occluded = true;
    bool add = true;
    for (int i=0; i<overObject->occluding.size(); i++){
        if (overObject->occluding[i]==underObject) {add = false; return;}
    }
    if (add){
        overObject->occluding.push_back(underObject);
    }
    add = true;
    for (int i=0; i<underObject->occluders.size(); i++){
        if (underObject->occluders[i]==overObject) {add = false; return;}
    }
    if (add){
        underObject->occluders.push_back(overObject);
    }

    for (int i=0; i<underObject->occluding.size(); i++){
        occludeBy(underObject->occluding[i], overObject);
    }
}



