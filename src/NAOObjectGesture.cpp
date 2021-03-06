#include "NAOObjectGesture.h"
#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>


#include <alcommon/alproxy.h>
#include <alcommon/albroker.h>
#include <alproxies/almemoryproxy.h>
#include <alproxies/alvideodeviceproxy.h>
#include <alproxies/almotionproxy.h>
#include <alvision/alimage.h>
#include <alvision/alvisiondefinitions.h>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/ref.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread/pthread/condition_variable.hpp>
#include <boost/thread/pthread/mutex.hpp>

#include <qi/log.hpp>
#include "include/ObjectTracking.hpp"
#include "GestureRecognition.hpp"

#define RESOLUTION AL::kQVGA
#define COLORSPACE AL::kBGRColorSpace


using namespace boost::filesystem;

struct NAOObjectGesture::Impl{
    NAOObjectGesture &module;
    boost::shared_ptr<AL::ALMemoryProxy> memoryProxy;
    vector<int> imgTimestamp;
    vector<NAOEvent> events;

    boost::shared_ptr<AL::ALVideoDeviceProxy> camProxy;
    std::string camProxyName;
    Size imsize;
    int camIdx;
    int FPS;

    boost::shared_ptr<AL::ALMotionProxy> motionProxy;
    int focusObjectId;

    boost::shared_ptr<ObjectTracker> objectTracker;
    boost::mutex objTrackerLock;
    vector<Gesture> gestures;

    boost::mutex fileLock;

    boost::posix_time::time_duration samplingPeriod;
    boost::thread *t;
    bool stopThread;
    boost::mutex stopThreadLock;
    boost::condition_variable condVar;


    Impl(NAOObjectGesture& mod)
        : module(mod), t(NULL), FPS(20), samplingPeriod(boost::posix_time::milliseconds(50)), focusObjectId(0)
    {
        try{
            objectTracker = boost::shared_ptr<ObjectTracker>(new ObjectTracker());
            memoryProxy = boost::shared_ptr<AL::ALMemoryProxy>(new AL::ALMemoryProxy(module.getParentBroker()));
            camProxy = boost::shared_ptr<AL::ALVideoDeviceProxy>(new AL::ALVideoDeviceProxy(module.getParentBroker()));
            motionProxy = boost::shared_ptr<AL::ALMotionProxy>(new AL::ALMotionProxy(module.getParentBroker()));
        } catch (std::exception &e){
            qiLogError("NAOObjectGesture") << "Failed to initialize NAOObjectGesture class: " << e.what() << std::endl;
        }
        if (!memoryProxy){
            qiLogError("NAOObjectGesture") << "Failed to get a proxy to ALMemory" << std::endl;
            throw std::runtime_error("Failed to get a proxy to ALMemory");
        }
        if (!motionProxy){
            qiLogError("NAOObjectGesture") << "Failed to get a proxy to ALMotion" << std::endl;
            throw std::runtime_error("Failed to get a proxy to ALMotion");
        }
        if (!camProxy){
            qiLogError("NAOObjectGesture") << "Failed to get a proxy to ALVideoDevice" << std::endl;
            throw std::runtime_error("Failed to get a proxy to ALVideoDevice");
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
        camProxyName = camProxy->subscribeCamera("NAOObjectGesture", camIdx, RESOLUTION, COLORSPACE, FPS);/*
        camProxy->setParam(3, 40);
        camProxy->setParam(11, 1);
        camProxy->setParam(22, 2);
        camProxy->setParam(12, 0);
        camProxy->setParam(33, -36);*/
        switch (RESOLUTION){
            case AL::kQQVGA: imsize = Size(160,120); break;
            case AL::kQVGA: imsize = Size(320,240); break;
            case AL::kVGA: imsize = Size(640,480); break;
        }
        boost::system_time tickTime = boost::get_system_time();
        int ticks = 0;
        boost::posix_time::time_duration thousandFrameTime(boost::posix_time::seconds(0));

        while(!stopThreadCopy){
            ticks+=1;
            boost::system_time now = boost::get_system_time();
            objTrackerLock.lock();
            Mat inputImage;
            try{
                const AL::ALImage* img = (AL::ALImage*)camProxy->getImageLocal(camProxyName);
                Mat imgHeader = Mat(imsize, CV_8UC3, (void*)img->getData());
                imgHeader.copyTo(inputImage);
                camProxy->releaseImage(camProxyName);
            }
            catch (std::exception& e){
                qiLogError("NAOObjectGesture") << "Error acquiring image: " << e.what() << std::endl;
                stopThreadLock.lock();
                stopThread = true;
                stopThreadLock.unlock();
            }

            Mat disregard;
            objectTracker->process(inputImage, &disregard);


            int tFocusObject = focusObjectId;
            if (focusObjectId<0 && (-focusObjectId)<= objectTracker->objectKinds.size()){
                tFocusObject = objectTracker->largestObjOfKind[(-focusObjectId)-1];
            }
            else{
                if (focusObjectId<0){
                    focusObjectId=0;
                    tFocusObject = 0;
                }
            }

            //this works because largestObjOfKind[i] = 0 when no objects of kind present
            if (tFocusObject!=0){
                motionProxy->setStiffnesses("Head", 0.8);
                objMap::iterator it = objectTracker->objects.find(tFocusObject);
                if (it != objectTracker->objects.end()){
                    AL::ALValue newAngles = pt2headAngles(it->second->ellipse.center);
                    AL::ALValue currentAngles = motionProxy->getAngles("Head", true);
                    bool moveNow = false;
                    for (int i=0; i<newAngles.getSize(); i++){
                        float a1 = newAngles[i];
                        float a2 = currentAngles[i];
                        if (abs(a1-a2)>0.08){
                            moveNow = true;
                        }
                    }
                    if(moveNow){
                        motionProxy->setAngles("Head", newAngles, 0.08);
                    }
                }
                else {
                    qiLogInfo("NAOObjectGesture") << "Lost focus on object " << focusObjectId << std::endl;
                    focusObjectId = 0;
                }
            }

            /* manual timestamp from posix_time */

            boost::posix_time::time_duration lastImgTime = boost::get_system_time() - time_t_epoch;
            long timesec = lastImgTime.total_seconds();
            boost::posix_time::time_duration millis = lastImgTime - boost::posix_time::seconds(timesec);
            long timemillis = millis.total_milliseconds();
            imgTimestamp.clear();
            imgTimestamp.push_back(timesec);
            imgTimestamp.push_back(timemillis);
            //this is time since epoch in compatible values

            for (int j=0; j<events.size(); j++){
                int id = events[j].objectId;
                bool trackingLargest = false;
                if ((-id) > objectTracker->objectKinds.size()){
                    //if tracking nonexistent kind (simplified)
                    events[j].log(gestures);
                    memoryProxy->removeMicroEvent(events[j].name);
                    events.erase(events.begin()+j);
                    j--;
                    continue;
                }
                if (id<0 && (-id) <= objectTracker->objectKinds.size()){
                    id = objectTracker->largestObjOfKind[(-events[j].objectId)-1];
                    trackingLargest = true;
                }
                objMap::iterator it = objectTracker->objects.find(id);
                if (it != objectTracker->objects.end()){
                    boost::shared_ptr<TrackedObject> obj = it->second;
                    AL::ALValue objData = getObjDataInternal(id,15);
                    events[j].notify(memoryProxy, objData, gestures);
                }
                else {
                    if (!trackingLargest){
                        events[j].deadNotify(memoryProxy, gestures);
                        memoryProxy->removeMicroEvent(events[j].name);
                        events.erase(events.begin()+j);
                        j--;
                        continue;
                    }
                    else {
                        events[j].deadNotify(memoryProxy, gestures);
                        events[j].trajectory.cutoff(-1);
                    }
                }
            }

            objTrackerLock.unlock();


            tickTime += samplingPeriod;
            condVar.timed_wait(scopeFileLock, tickTime, boost::lambda::var(stopThread)); //unlock scopeFileLock while waiting111
            stopThreadLock.lock();
            stopThreadCopy = stopThread;
            stopThreadLock.unlock();

            //framerate display, purely diagnostic
            boost::posix_time::time_duration oneLoop = boost::get_system_time() - now;
            thousandFrameTime += oneLoop;
            if (ticks==100){
                ticks=0;
                qiLogVerbose("NAOObjectGesture") << "Tracking working at " << 100000.0f/thousandFrameTime.total_milliseconds() << " FPS" << std::endl;
                thousandFrameTime = boost::posix_time::milliseconds(0);
            }

        }
        camProxy->unsubscribe(camProxyName);
    }

    vector<float> pt2headAngles(Point2i pt){
        float normx = 1.0f*pt.x/imsize.width;
        float normy = 1.0f*pt.y/imsize.height;
        std::vector<float> normpos;
        normpos.push_back(normx);
        normpos.push_back(normy);
        std::vector<float> angpos;
        angpos = camProxy->getAngularPositionFromImagePosition(camIdx, normpos);
        std::vector<float> headAngles;
        headAngles = motionProxy->getAngles("Head", true);
        headAngles[0]+=angpos[0];
        headAngles[1]+=angpos[1];
        return headAngles;
    }

    AL::ALValue getObjData(int objId, int dataCode){
        AL::ALValue objData;
        objTrackerLock.lock();
        objData = getObjDataInternal(objId, dataCode);
        objTrackerLock.unlock();
        return objData;
    }

    AL::ALValue getObjDataInternal(int objId, int dataCode){
        AL::ALValue objData;
        objMap::iterator it = objectTracker->objects.find(objId);
        if (it != objectTracker->objects.end()){
            objData.arrayPush(it->first);
            boost::shared_ptr<TrackedObject> obj = it->second;
            if (dataCode & 1){
                AL::ALValue timestamp(imgTimestamp);
                objData.arrayPush(timestamp);
            }
            if (dataCode & 2){
                objData.arrayPush(obj->kind);
            }
            if (dataCode & 4){
                AL::ALValue alpt = pt2headAngles(obj->ellipse.center);
                objData.arrayPush(alpt);
            }
            if (dataCode & 8){
                objData.arrayPush(obj->area);
            }
            if (dataCode & 16){
                AL::ALValue gesturesRecognized;
                for (int i=0; i<gestures.size(); i++){
                    vector<int> gr = gestures[i].existsIn(obj->traj, false);
                    if (gr.size()>0){
                        gesturesRecognized.arrayPush(gestures[i].name);
                    }
                }
                objData.arrayPush(gesturesRecognized);
            }
        }
        return objData;
    }

    bool removeEvent(std::string name){
        qiLogInfo("NAOObjectGesture") << "Attempting to remove event " << name << std::endl;
        objTrackerLock.lock();
        bool nameFound = false;
        int nameIdx = -1;
        for (int i=0; i<events.size(); i++){
            if (name.compare(events[i].name)==0){
                nameFound = true;
                nameIdx=i;
                break;
            }
        }
        if (!nameFound){
            qiLogError("NAOObjectGesture") << "Attempted to remove nonexistent event." << std::endl;
            objTrackerLock.unlock();
            return false;
        }
        else {
            events[nameIdx].deadNotify(memoryProxy, gestures);
            memoryProxy->removeMicroEvent(events[nameIdx].name);
            events.erase(events.begin()+nameIdx);
            qiLogInfo("NAOObjectGesture") << "Removed event " << name << std::endl;
        }
        objTrackerLock.unlock();
        return true;
    }
};


NAOObjectGesture::NAOObjectGesture(boost::shared_ptr<AL::ALBroker> pBroker, const std::string& pName) : ALModule(pBroker, pName){
    setModuleDescription("Object tracker and gesture recognition module");

    functionName("startTracker", getName(), "Start tracking all initialized kinds of objects.");
    addParam("fps", "Video stream framerate");
    addParam("camIdx", "Index of the camera in the video system. 0 - top camera, 1 - bottom camera");
    BIND_METHOD(NAOObjectGesture::startTracker);

    functionName("stopTracker", getName(), "Stop object tracker without deleting object kinds.");
    BIND_METHOD(NAOObjectGesture::stopTracker);

    functionName("loadDataset", getName(), "Load image dataset from folder.");
    addParam("dataFolder", "Root folder containing Dataset and GroundTruth folders.");
    BIND_METHOD(NAOObjectGesture::loadDataset);

    functionName("removeObjectKind", getName(), "Remove object kind specified by id");
    addParam("kindId", "Id of object kind to remove");
    BIND_METHOD(NAOObjectGesture::removeObjectKind);

    functionName("getObjectList", getName(), "Get specified data for all tracked objects");
    addParam("dataCode", "Requested data identifier");
    setReturn("objectData", "ALValue array containing timestamp at position 0 and object data at positions 1 to N");
    BIND_METHOD(NAOObjectGesture::getObjectList);

    functionName("getObjectData", getName(), "Get specified data for list of object identifiers");
    addParam("objectIds", "ALValue integer vector of object identifiers");
    addParam("dataCode", "Requested data identifier");
    setReturn("objectData", "ALValue array containing timestamp at position 0 and object data at positions 1 to N. Empty array means object is untracked.");
    BIND_METHOD(NAOObjectGesture::getObjectData);

    functionName("trackObject", getName(), "Raise microevent after each object detection");
    addParam("name", "Microevent name");
    addParam("objId", "Identifier of object to be tracked");
    setReturn("eventAdded", "Boolean value. Returns true if event was added or already exists, false otherwise");
    BIND_METHOD(NAOObjectGesture::trackObject);

    functionName("clearEventTraj", getName(), "Clear the trajectory of object tracked under event");
    addParam("name", "Microevent name");
    BIND_METHOD(NAOObjectGesture::clearEventTraj);

    functionName("getEventList", getName(), "Get list of all events this module raises with corresponding object ids");
    setReturn("eventList", "List of all events. Each event is in [name, objectid] format");
    BIND_METHOD(NAOObjectGesture::getEventList);

    functionName("removeEvent", getName(), "Removes event from list and notifies subscribers");
    addParam("name", "Microevent name");
    setReturn("eventRemoved", "Returns true if event successfully removed, false otherwise");
    BIND_METHOD(NAOObjectGesture::removeEvent);

    functionName("addGesture", getName(), "Add gesture to recognize");
    addParam("name", "Name of gesture");
    addParam("dirList", "Vector of integer directions in range 0-7");
    BIND_METHOD(NAOObjectGesture::addGesture);

    functionName("removeGesture", getName(), "Remove gesture from list");
    addParam("name", "Name of gesture");
    BIND_METHOD(NAOObjectGesture::removeGesture);

    functionName("getGestureList", getName(), "Get list of gesture names");
    setReturn("gestureNames", "ALValue array containing gesture names");
    BIND_METHOD(NAOObjectGesture::getGestureList);

    functionName("focusObject", getName(), "Focus NAO on object and track it using head movements");
    addParam("objId", "Identifier of object to be tracked");
    setReturn("eventAdded", "Boolean value. Returns true if object is being tracked, false otherwise");
    BIND_METHOD(NAOObjectGesture::focusObject);


    functionName("stopFocus", getName(), "Stop tracking objects with head. Note: doesn't return head to neutral position");
    BIND_METHOD(NAOObjectGesture::stopFocus);

}

NAOObjectGesture::~NAOObjectGesture(){}

void NAOObjectGesture::init(){
    try{
        qi::log::init();
        atexit(qi::log::destroy);
        impl = boost::shared_ptr<Impl>(new Impl(*this));
        AL::ALModule::init();
    } catch (std::exception &e){
        qiLogError("NAOObjectGesture") << "Failed to initialize NAOObjectGesture class: " << e.what() << std::endl;
        exit();
    }
    qiLogInfo("NAOObjectGesture") << "Successfully initialized NAOObjectGesture module" << std::endl;
}

void NAOObjectGesture::exit(){
    AL::ALModule::exit();
}

void NAOObjectGesture::startTracker(const int &FPS, const int &camIdx){
    stopTracker();
    qiLogInfo("NAOObjectGesture") << "Starting ObjectTracker with image acquisition at " << FPS << " FPS" << std::endl;
    try{
        impl->FPS = FPS;
        impl->camIdx = camIdx;
        impl->samplingPeriod = boost::posix_time::milliseconds(1000/FPS);
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
    qiLogInfo("NAOObjectGesture") << "Stopping ObjectTracker" << std::endl;
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

void NAOObjectGesture::loadDataset(const std::string& dataFolder){
    path rootDir(dataFolder);
    qiLogInfo("NAOObjectGesture") << "Attempting to load dataset in " << dataFolder << std::endl;
    if (!exists(rootDir) || !is_directory(rootDir)){
        qiLogError("NAOObjectGesture") << "Failed to load dataset: Bad directory "<< dataFolder << std::endl;
    }
    impl->objTrackerLock.lock();
    bool cond = impl->objectTracker->addObjectKind(rootDir.string());
    impl->objTrackerLock.unlock();
    if (cond){
        qiLogInfo("NAOObjectGesture") << "Loaded histogram from .png in " << dataFolder << std::endl;
        return;
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
            bool cond = impl->objectTracker->addObjectKind(images, masks, rootDir.string());
            impl->objTrackerLock.unlock();
            if (cond){
                qiLogInfo("NAOObjectGesture") << "Loaded " << images.size() << " images" << std::endl;
            }
            else {
                qiLogError("NAOObjectGesture") << "Failed to load dataset" << std::endl;
            }
        } catch (std::exception &e){
            qiLogError("NAOObjectGesture") << "Failed to load dataset images: " << e.what() << std::endl;
            exit();
        }
    } else {
        qiLogError("NAOObjectGesture") << "Failed to load dataset: Subdirectories missing"<< std::endl;
    }
}

void NAOObjectGesture::removeObjectKind(const int& id){
    impl->objTrackerLock.lock();
    if (id>=impl->objectTracker->objectKinds.size()){
        qiLogError("NAOObjectGesture") << "Attempted to erase nonexistent object kind."<< std::endl;
    } else {
        impl->objectTracker->objectKinds.erase(impl->objectTracker->objectKinds.begin()+id);
        qiLogInfo("NAOObjectGesture") << "Removed object kind " << id << std::endl;
    }
    impl->objTrackerLock.unlock();
}

/**
*   dataCode is any combination of:
*   1: object timestamp
*   2: object kind
*   4: object centroid point
*   8: object area
*   16: list of recognized gestures
*
*   Data is returned in that exact order (timestamp, then centroid, then area etc.)
*   Returned array always starts with object id
*/
AL::ALValue NAOObjectGesture::getObjectList(const int &dataCode){
    try {
        AL::ALValue retval;
        impl->objTrackerLock.lock();
        AL::ALValue timestamp(impl->imgTimestamp);
        for (objMap::iterator it=impl->objectTracker->objects.begin(); it!=impl->objectTracker->objects.end(); ++it){
            AL::ALValue objData;
            objData.arrayPush(it->first);
            boost::shared_ptr<TrackedObject> obj = it->second;
            if (dataCode & 1){
                objData.arrayPush(timestamp);
            }
            if (dataCode & 2){
                objData.arrayPush(obj->kind);
            }
            if (dataCode & 4){
                AL::ALValue alpt = impl->pt2headAngles(obj->ellipse.center);
                objData.arrayPush(alpt);
            }
            if (dataCode & 8){
                objData.arrayPush(obj->area);
            }
            if (dataCode & 16){
                AL::ALValue gesturesRecognized;
                for (int i=0; i<impl->gestures.size(); i++){
                    vector<int> gr = impl->gestures[i].existsIn(obj->traj, false);
                    if (gr.size()>0){
                        gesturesRecognized.arrayPush(impl->gestures[i].name);
                    }
                }
                objData.arrayPush(gesturesRecognized);
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

AL::ALValue NAOObjectGesture::getObjectData(const AL::ALValue& objectIds, const int &dataCode){
    try{
        AL::ALValue retval;
        for (int j=0; j<objectIds.getSize(); j++){
            int objId = objectIds[j];
            AL::ALValue objData = impl->getObjData(objId, dataCode);
            retval.arrayPush(objData);
        };
        return retval;
    } catch (std::exception &e){
        qiLogError("NAOObjectGesture") << "getObjectPositions failed: " << e.what() << std::endl;
        exit();
    }
}

bool NAOObjectGesture::trackObject(const string &name, const int &objId){
    impl->objTrackerLock.lock();
    for (int i=0; i<impl->events.size(); i++){
        if (name.compare(impl->events[i].name)==0){
            if (impl->events[i].objectId==objId){
                qiLogInfo("NAOObjectGesture") << "Attempted to create tracking event identical to an existing one." << std::endl;
                impl->objTrackerLock.unlock();
                return true;
            }
            else {
                qiLogError("NAOObjectGesture") << "Duplicate tracking event name, event not created" << std::endl;
                impl->objTrackerLock.unlock();
                return false;
            }
        }
    }
    if (objId>0){
        objMap::iterator it = impl->objectTracker->objects.find(objId);
        if (it == impl->objectTracker->objects.end()){
            qiLogError("NAOObjectGesture") << "Attempted to track nonexistent object, event not created" << std::endl;
            impl->objTrackerLock.unlock();
            return false;
        }
    } else {
        if ( (-objId) > impl->objectTracker->objectKinds.size() && objId!=0){
            qiLogError("NAOObjectGesture") << "Attempted to track largest object of nonexistent kind." << std::endl;
            impl->objTrackerLock.unlock();
            return false;
        }
    }
    NAOEvent tEvent(name, objId, {0.3, 0.0},{1.0, -0.7});
    impl->events.push_back(tEvent);
    impl->objTrackerLock.unlock();
    qiLogInfo("NAOObjectGesture") << "Now tracking object " << objId << " using event " << name << std::endl;
    return true;
}

void NAOObjectGesture::clearEventTraj(const string &name){
    impl->objTrackerLock.lock();
    for (int i=0; i<impl->events.size(); i++){
        if (name.compare(impl->events[i].name)==0){
            impl->events[i].trajectory.cutoff(-1);
            qiLogVerbose("NAOObjectGesture") << "Trajectory assigned to event " << name << " cleared" << std::endl;
            impl->objTrackerLock.unlock();
            return;
        }
    }
    qiLogError("NAOObjectGesture") << "Attempted to clear trajectory assigned to nonexistent event" << std::endl;
    impl->objTrackerLock.unlock();
}

AL::ALValue NAOObjectGesture::getEventList(){
    impl->objTrackerLock.lock();
    AL::ALValue retval;
    for (int i=0; i< impl->events.size(); i++){
        AL::ALValue elm;
        elm.arrayPush(impl->events[i].name);
        elm.arrayPush(impl->events[i].objectId);
        retval.arrayPush(elm);
    }
    impl->objTrackerLock.lock();
    return retval;
}

bool NAOObjectGesture::removeEvent(const std::string& name){
    bool ret = impl->removeEvent(name);
    return ret;
}

void NAOObjectGesture::addGesture(const string &name, const AL::ALValue& dirList){
    impl->objTrackerLock.lock();
    for (int i=0; i<impl->gestures.size(); i++){
        if (name.compare(impl->gestures[i].name)==0){
            qiLogError("NAOObjectGesture") << "Attempted to create gesture with duplicate name." << std::endl;
            impl->objTrackerLock.unlock();
            return;
        }
    }
    if (!dirList.isArray()){
        qiLogError("NAOObjectGesture") << "Gesture direction list is not array." << std::endl;
        impl->objTrackerLock.unlock();
        return;
    }
    vector<int> directionList;
    for (int i=0; i<dirList.getSize(); i++){
        if (!dirList[i].isInt()){
            qiLogError("NAOObjectGesture") << "Gesture direction list contains invalid element" << std::endl;
            impl->objTrackerLock.unlock();
            return;
        }
        int el = dirList[i];
        if (el<0 || el >7){
            qiLogError("NAOObjectGesture") << "Gesture direction list contains invalid element" << std::endl;
            impl->objTrackerLock.unlock();
            return;
        }
        directionList.push_back(el);
    }
    Gesture temp(name, directionList);
    impl->gestures.push_back(temp);
    qiLogInfo("NAOObjectGesture") << "Added gesture " << name << std::endl;
    impl->objTrackerLock.unlock();
}

void NAOObjectGesture::removeGesture(const string &name){
    impl->objTrackerLock.lock();
    for (int i=0; i<impl->gestures.size(); i++){
        if (name.compare(impl->gestures[i].name)==0){
            impl->gestures.erase(impl->gestures.begin()+i);
            impl->objTrackerLock.unlock();
            qiLogInfo("NAOObjectGesture") << "Removed gesture " << name << std::endl;
            return;
        }
    }
    qiLogError("NAOObjectGesture") << "Gesture " << name << " does not exist." << std::endl;
    impl->objTrackerLock.unlock();
}

AL::ALValue NAOObjectGesture::getGestureList(){
    AL::ALValue retval;
    impl->objTrackerLock.lock();
    for (int i=0; i<impl->gestures.size(); i++){
        retval.arrayPush(impl->gestures[i].name);
    }
    impl->objTrackerLock.unlock();
    return retval;
}

bool NAOObjectGesture::focusObject(const int &objId){
    impl->objTrackerLock.lock();
    if (objId == impl->focusObjectId){
        qiLogInfo("NAOObjectGesture") << "Attempted to refocus on already focused object." << std::endl;
        impl->objTrackerLock.unlock();
        return true;
    }
    if (objId>0){
        objMap::iterator it = impl->objectTracker->objects.find(objId);
        if (it == impl->objectTracker->objects.end()){
            qiLogError("NAOObjectGesture") << "Attempted to focus on nonexistent object." << std::endl;
            impl->objTrackerLock.unlock();
            return false;
        }
    } else {
        if ( (-objId) > impl->objectTracker->objectKinds.size() ){
            qiLogError("NAOObjectGesture") << "Attempted to focus on largest object of nonexistent kind." << std::endl;
            impl->objTrackerLock.unlock();
            return false;
        }
    }
    impl->focusObjectId = objId;
    impl->objTrackerLock.unlock();
    qiLogInfo("NAOObjectGesture") << "Now focused on object " << objId << ". Tracking with NAO head." << std::endl;
    return true;
}

void NAOObjectGesture::stopFocus(){
    impl->objTrackerLock.lock();
    impl->focusObjectId = 0;
    impl->objTrackerLock.unlock();
}

NAOEvent::NAOEvent(string tName, int tObjectId): name(tName), objectId(tObjectId), trajectory(Trajectory()){}

NAOEvent::NAOEvent(string tName, int tObjectId, vector<float> num, vector<float> den) : name(tName), objectId(tObjectId), trajectory(Trajectory(num, den)){}

NAOEvent::~NAOEvent()
{}

void NAOEvent::notify(boost::shared_ptr<AL::ALMemoryProxy> memoryProxy, AL::ALValue value, vector<Gesture> gestures)
{
    AL::ALValue gesturesRecognized;
    for (int i=0; i<gestures.size(); i++){
        vector<int> gr = gestures[i].existsIn(trajectory, false);
        if (gr.size()>0){
            gesturesRecognized.arrayPush(gestures[i].name);
        }
    }
    value.arrayPush(gesturesRecognized);
    memoryProxy->raiseMicroEvent(name, value);
    //this is an extremely inelegant way to do this
    cv::Point2f newpt(0.0,0.0);
    newpt.x = value[3][0];
    newpt.y = value[3][1];
    /*boost::posix_time::ptime time_t_epoch(boost::gregorian::date(1970,1,1));
    boost::posix_time::ptime now(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration sinceEpoch = now-time_t_epoch;
    trajectory.append(newpt, sinceEpoch.total_milliseconds());*/
    long secs = (int)value[1][0];
    long ms = (int)value[1][1];
    long long timestamp = 1000*secs + ms;
    trajectory.append(newpt, timestamp);
}

void NAOEvent::deadNotify(boost::shared_ptr<AL::ALMemoryProxy> memoryProxy, vector<Gesture> gestures)
{
    AL::ALValue lastData;
    lastData.arrayPush(0);
    AL::ALValue gesturesRecognized;
    for (int i=0; i<gestures.size(); i++){
        vector<int> gr = gestures[i].existsIn(trajectory, true);
        if (gr.size()>0){
            gesturesRecognized.arrayPush(gestures[i].name);
        }
    }
    lastData.arrayPush(gesturesRecognized);
    memoryProxy->raiseMicroEvent(name, lastData);
    log(gestures);
}

void NAOEvent::log(vector<Gesture> gestures)
{
    boost::filesystem3::path tpath("/home/nao/LogTrajectory");
    std::string tname = name;
    time_t rawtime;
    struct tm * timeinfo;
    char buffer [80];
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime(buffer, 80, "_%F_%H-%M-%S", timeinfo);
    tname.append(buffer);
    tname.append(".csv");
    tpath/=tname;
    trajectory.logTo(tpath, gestures);
}

void NAOEvent::log()
{
    vector<Gesture> temp;
    log(temp);
}

