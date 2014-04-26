#include "NAOObjectGesture.h"
#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

#include <alvalue/alvalue.h>
#include <alcommon/alproxy.h>
#include <alcommon/albroker.h>
#include <alproxies/almemoryproxy.h>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/ref.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/filesystem.hpp>

#include <boost/thread/pthread/condition_variable.hpp>
#include <boost/thread/pthread/mutex.hpp> //remove

#include <qi/log.hpp>
#include "include/ObjectTracking.hpp"

using namespace boost::filesystem;


struct NAOObjectGesture::Impl{
    NAOObjectGesture &module;
    boost::shared_ptr<AL::ALMemoryProxy> memoryProxy;
    boost::shared_ptr<ObjectTracker> objectTracker;
    vector<int> imgTimestamp;
    vector<std::string> eventNames;
    vector<int> objectIds;
    boost::mutex objTrackerLock;

    boost::mutex fileLock;

    boost::posix_time::time_duration samplingPeriod;
    boost::thread *t;
    bool stopThread;
    boost::mutex stopThreadLock;
    boost::condition_variable condVar;


    Impl(NAOObjectGesture& mod)
        : module(mod), t(NULL), samplingPeriod(boost::posix_time::milliseconds(34))
    {
        try{
            objectTracker = boost::shared_ptr<ObjectTracker>(new ObjectTracker());
            memoryProxy = boost::shared_ptr<AL::ALMemoryProxy>(new AL::ALMemoryProxy(module.getParentBroker()));
        } catch (std::exception &e){
            qiLogError("NAOObjectGesture") << "Failed to initialize NAOObjectGesture class: " << e.what() << std::endl;
        }
        if (!memoryProxy){
            qiLogError("NAOObjectGesture") << "Failed to get a proxy to ALMemory" << std::endl;
            throw std::runtime_error("Failed to get a proxy to ALMemory");
        }
    }

    ~Impl(){
        stopThreadLock.lock();
        stopThread = true;
        stopThreadLock.unlock();
        if (t){
            t->join();
        }
    }

    void operator()(){
        boost::mutex::scoped_lock scopeFileLock(fileLock);
        bool stopThreadCopy;
        boost::posix_time::ptime time_t_epoch(boost::gregorian::date(1970,1,1));
        stopThreadLock.lock();
        stopThreadCopy = stopThread;
        stopThreadLock.unlock();
        boost::system_time tickTime = boost::get_system_time();
        while(!stopThreadCopy){
            //do things
            objTrackerLock.lock();
            //get image here

            boost::posix_time::time_duration lastImgTime = boost::get_system_time() - time_t_epoch;
            long timesec = lastImgTime.total_seconds();
            boost::posix_time::time_duration millis = lastImgTime - boost::posix_time::seconds(timesec);
            long timemillis = millis.total_milliseconds();
            imgTimestamp.clear();
            imgTimestamp.push_back(timesec);
            imgTimestamp.push_back(timemillis);
            //this is time since epoch

            int numObj = objectTracker->objects.size();
            for (int j=0; j<objectIds.size(); j++){
                AL::ALValue objData;
                for (int i=0; i<numObj; i++){
                    if (objectTracker->objects[i]->id == objectIds[j]){
                        boost::shared_ptr<TrackedObject> obj = objectTracker->objects[i];
                        objData.arrayPush(obj->id);
                        AL::ALValue alpt;
                        alpt.arrayPush(obj->ellipse.center.x);
                        alpt.arrayPush(obj->ellipse.center.y);
                        objData.arrayPush(alpt);
                        objData.arrayPush(obj->area);
                        break;
                    }
                }
                if (objData.getSize()>0){
                    memoryProxy->raiseMicroEvent(eventNames[j], objData);
                }
                else {
                    memoryProxy->removeMicroEvent(eventNames[j]);
                    eventNames.erase(eventNames.begin()+j);
                    objectIds.erase(objectIds.begin()+j);
                    j--;
                }
            }
            objTrackerLock.unlock();


            tickTime += samplingPeriod;
            condVar.timed_wait(scopeFileLock, tickTime, boost::lambda::var(stopThread)); //unlock scopeFileLock while waiting111
            stopThreadLock.lock();
            stopThreadCopy = stopThread;
            stopThreadLock.unlock();
        }
    }
};





NAOObjectGesture::NAOObjectGesture(boost::shared_ptr<AL::ALBroker> pBroker, const std::string& pName) : ALModule(pBroker, pName){
    setModuleDescription("Object tracker and gesture recognition module");

}

NAOObjectGesture::~NAOObjectGesture(){}

void NAOObjectGesture::init(){
    try{
        impl = boost::shared_ptr<Impl>(new Impl(*this));
        AL::ALModule::init();
    } catch (std::exception &e){
        qiLogError("NAOObjectGesture") << "Failed to initialize NAOObjectGesture class: " << e.what() << std::endl;
        exit();
    }
}

void NAOObjectGesture::exit(){
    AL::ALModule::exit();
}

void NAOObjectGesture::startTracker(const int &milli){
    stopTracker();
    qiLogVerbose("NAOObjectGesture") << "Starting ObjectTracker with T = " << milli <<"ms" << std::endl;
    try{
        impl->samplingPeriod = boost::posix_time::milliseconds(milli);
        impl->stopThreadLock.lock();
        impl->stopThread=false;
        impl->stopThreadLock.unlock();
        impl->t = new boost::thread(boost::ref(*impl));

    } catch (std::exception &e){
        qiLogError("NAOObjectGesture") << "Failed to start ObjectTracker: " << e.what() << std::endl;
        exit();
    }
}

void NAOObjectGesture::stopTracker(){
    qiLogVerbose("NAOObjectGesture") << "Stopping ObjectTracker" << std::endl;
    try{
        impl->stopThreadLock.lock();
        impl->stopThread=true;
        impl->stopThreadLock.unlock();
        impl->condVar.notify_one();
        if (impl->t){
            impl->t->join();
        }
    } catch (std::exception &e){
        qiLogError("NAOObjectGesture") << "Failed to stop ObjectTracker: " << e.what() << std::endl;
        exit();
    }
}

void NAOObjectGesture::loadDataset(const std::string dataFolder){
    path rootDir(dataFolder);
    qiLogVerbose("NAOObjectGesture") << "Attempting to load dataset in " << dataFolder << std::endl;
    if (!exists(rootDir) || !is_directory(rootDir)){
        qiLogError("NAOObjectGesture") << "Failed to load dataset: Bad directory "<< dataFolder << std::endl;
    }
    path dataDir = rootDir / "Dataset";
    path gTruthDir = rootDir / "GroundTruth";
    vector<Mat> images;
    vector<Mat> masks;
    if (exists(dataDir) && exists(gTruthDir) && is_directory(dataDir) && is_directory(gTruthDir)){
        try{
            directory_iterator end_itr;
            for(directory_iterator itr(dataDir); itr!=end_itr; ++itr){
                path filename = itr->path().stem();
                for(directory_iterator itr2(gTruthDir); itr2!=end_itr; ++itr2){
                    path gTruthName = itr2->path().stem();
                    if(filename==gTruthName){
                        string impath = itr->path().string();
                        string maskpath = itr2->path().string();
                        Mat img(imread(impath));
                        Mat mask(imread(maskpath,0));
                        images.push_back(img);
                        masks.push_back(mask);
                    }
                }
            }
            impl->objTrackerLock.lock();
            impl->objectTracker->addObjectKind(images, masks);
            impl->objTrackerLock.unlock();
            qiLogVerbose("NAOObjectGesture") << "Loaded " << images.size() << " images" << std::endl;
        } catch (std::exception &e){
            qiLogError("NAOObjectGesture") << "Failed to load dataset images: " << e.what() << std::endl;
            exit();
        }
    } else {
        qiLogError("NAOObjectGesture") << "Failed to load dataset: Subdirectories missing"<< std::endl;
    }
}

/**
*   dataCode is any combination of:
*   1: object id
*   2: object centroid point
*   4: object area
*
*   Data is returned in that exact order (id, then centroid, then area etc.)
*   Returned array always starts with timestamp
*/
AL::ALValue NAOObjectGesture::getObjectList(const int dataCode){
    try {
        AL::ALValue retval;
        AL::ALValue timestamp(impl->imgTimestamp);
        retval.arrayPush(timestamp);
        impl->objTrackerLock.lock();
        int numObj = impl->objectTracker->objects.size();
        for (int i=0; i<numObj; i++){
            AL::ALValue objData;
            boost::shared_ptr<TrackedObject> obj = impl->objectTracker->objects[i];
            if (dataCode & 1){
                objData.arrayPush(obj->id);
            }
            if (dataCode & 2){
                AL::ALValue alpt;
                alpt.arrayPush(obj->ellipse.center.x);
                alpt.arrayPush(obj->ellipse.center.y);
                objData.arrayPush(alpt);
            }
            if (dataCode & 4){
                objData.arrayPush(obj->area);
            }
            retval.arrayPush(objData);
        }
        impl->objTrackerLock.unlock();
        return retval;
    } catch (std::exception &e){
        qiLogError("NAOObjectGesture") << "getObjectList failed: " << e.what() << std::endl;
        exit();
    }
}

AL::ALValue NAOObjectGesture::getObjectData(vector<int> objectIds, const int dataCode){
    try{
        AL::ALValue retval;
        AL::ALValue timestamp(impl->imgTimestamp);
        retval.arrayPush(timestamp);
        retval.arraySetSize(objectIds.size());
        impl->objTrackerLock.lock();
        int numObj = impl->objectTracker->objects.size();
        for (int j=0; j<objectIds.size(); j++){
            AL::ALValue objData;
            for (int i=0; i<numObj; i++){
                if (impl->objectTracker->objects[i]->id == objectIds[j]){
                    boost::shared_ptr<TrackedObject> obj = impl->objectTracker->objects[i];
                    if (dataCode & 1){
                        objData.arrayPush(obj->id);
                    }
                    if (dataCode & 2){
                        AL::ALValue alpt;
                        alpt.arrayPush(obj->ellipse.center.x);
                        alpt.arrayPush(obj->ellipse.center.y);
                        objData.arrayPush(alpt);
                    }
                    if (dataCode & 4){
                        objData.arrayPush(obj->area);
                    }
                    break;
                }
            }
            retval.arrayPush(objData);
        }
        impl->objTrackerLock.unlock();
        return retval;
    } catch (std::exception &e){
        qiLogError("NAOObjectGesture") << "getObjectPositions failed: " << e.what() << std::endl;
        exit();
    }
}

bool NAOObjectGesture::trackObject(const std::string name, const int objId){
    impl->objTrackerLock.lock();
    impl->eventNames.push_back(name);
    impl->objectIds.push_back(objId);
    impl->objTrackerLock.unlock();
}
