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
#include <boost/thread/pthread/mutex.hpp> //remove

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
    vector<std::string> eventNames;
    vector<int> eventObjectIds;
    vector<Trajectory> eventTrajectories;

    boost::shared_ptr<AL::ALVideoDeviceProxy> camProxy;
    std::string camProxyName;
    Size imsize;
    int camIdx;
    int FPS;

    boost::shared_ptr<AL::ALMotionProxy> motionProxy;
    int focusObjectId;

    boost::shared_ptr<ObjectTracker> objectTracker;
    boost::mutex objTrackerLock;
    vector<std::string> gestureNames;
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
        if (!camProxy){
            qiLogError("NAOObjectGesture") << "Failed to get a proxy to ALVideoDevice" << std::endl;
            throw std::runtime_error("Failed to get a proxy to ALVideoDevice");
        }
        if (!motionProxy){
            qiLogError("NAOObjectGesture") << "Failed to get a proxy to ALMotion" << std::endl;
            throw std::runtime_error("Failed to get a proxy to ALMotion");
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
        camProxyName = camProxy->subscribeCamera("NAOObjectGesture", camIdx, RESOLUTION, COLORSPACE, FPS);
        switch (RESOLUTION){
            case AL::kQQVGA: imsize = Size(160,120); break;
            case AL::kQVGA: imsize = Size(320,240); break;
            case AL::kVGA: imsize = Size(640,480); break;
        }
        boost::system_time tickTime = boost::get_system_time();
        int ticks = 0;
        boost::posix_time::time_duration thousandFrameTime(boost::posix_time::seconds(0));
        while(!stopThreadCopy){
            //do things
            ticks+=1;
            boost::system_time now = boost::get_system_time();
            objTrackerLock.lock();
            //get image here
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
                    motionProxy->setAngles("Head", newAngles, 0.1);
                }
                else {
                    qiLogInfo("NAOObjectGesture") << "Lost focus on object " << focusObjectId << std::endl;
                    motionProxy->setStiffnesses("Head", 0.0);
                    focusObjectId = 0;
                }
            } else {
                motionProxy->setStiffnesses("Head", 0.0);
            }

            /* manual timestamp from posix_time */
            boost::posix_time::time_duration lastImgTime = boost::get_system_time() - time_t_epoch;
            long timesec = lastImgTime.total_seconds();
            boost::posix_time::time_duration millis = lastImgTime - boost::posix_time::seconds(timesec);
            long timemillis = millis.total_milliseconds();
            imgTimestamp.clear();
            imgTimestamp.push_back(timesec);
            imgTimestamp.push_back(timemillis);
            //this is time since epoch

            int numObj = objectTracker->objects.size();

            for (int j=0; j<eventObjectIds.size(); j++){
                int id = eventObjectIds[j];
                if (id<0 && (-id)<= objectTracker->objectKinds.size()){
                    id = objectTracker->largestObjOfKind[(-eventObjectIds[j])-1];
                }
                else {
                    if (id<0 && (-id) > objectTracker->objectKinds.size()){
                        memoryProxy->removeMicroEvent(eventNames[j]);
                        eventNames.erase(eventNames.begin()+j);
                        eventObjectIds.erase(eventObjectIds.begin()+j);
                        eventTrajectories.erase(eventTrajectories.begin()+j);
                        j--;
                        continue;
                    }
                }
                AL::ALValue objData = getObjDataInternal(id,31);

                if (objData.getSize() == 0 && eventTrajectories[j].points.size()>0){
                    AL::ALValue lastData;
                    lastData.arrayPush(0);
                    AL::ALValue gesturesRecognized;
                    for (int i=0; i<gestures.size(); i++){
                        vector<int> gr = gestures[i].existsIn(eventTrajectories[j], true);
                        if (gr.size()>0){
                            gesturesRecognized.arrayPush(gestureNames[i]);
                        }
                    }
                    lastData.arrayPush(gesturesRecognized);
                    eventTrajectories[j] = Trajectory();
                    memoryProxy->raiseMicroEvent(eventNames[j], lastData);

                    if (eventObjectIds[j]>0){
                        qiLogInfo("NAOObjectGesture") << "Object " << eventObjectIds[j] << " permanently lost. Notifying subscribers and deleting event " << eventNames[j] << std::endl;
                        memoryProxy->removeMicroEvent(eventNames[j]);
                        eventNames.erase(eventNames.begin()+j);
                        eventObjectIds.erase(eventObjectIds.begin()+j);
                        eventTrajectories.erase(eventTrajectories.begin()+j);
                        j--;
                        continue;
                    }
                }
                else {
                    if (id!=0){
                        memoryProxy->raiseMicroEvent(eventNames[j], objData);
                        //cv::Point2f newpt(0.0,0.0);
                        //newpt.x = objData[3][0];
                        //newpt.y = objData[3][1];
                        boost::shared_ptr<TrackedObject> obj = objectTracker->objects[id];
                        obj->updateTrajectory(obj->ellipse.center,1.0);

                        eventTrajectories[j].append(objectTracker->objects[id]->ellipse.center, 1);
                    }
                    //objectTracker->objects[id]->updateTrajectory(newpt, 1);
                }
            }

            objTrackerLock.unlock();


            tickTime += samplingPeriod;
            condVar.timed_wait(scopeFileLock, tickTime, boost::lambda::var(stopThread)); //unlock scopeFileLock while waiting111
            stopThreadLock.lock();
            stopThreadCopy = stopThread;
            stopThreadLock.unlock();
            boost::posix_time::time_duration oneLoop = boost::get_system_time() - now;
            thousandFrameTime += oneLoop;
            if (ticks==100){
                ticks=0;
                qiLogInfo("NAOObjectGesture") << "Tracking working at " << 100000.0f/thousandFrameTime.total_milliseconds() << " FPS" << std::endl;
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
                        gesturesRecognized.arrayPush(gestureNames[i]);
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
        for (int i=0; i<eventNames.size(); i++){
            if (name.compare(eventNames[i])==0){
                nameFound = true;
                i=nameIdx;
                break;
            }
        }
        if (!nameFound){
            qiLogError("NAOObjectGesture") << "Attempted to remove nonexistent event." << std::endl;
            objTrackerLock.unlock();
            return false;
        }
        else {

            AL::ALValue lastData;
            lastData.arrayPush(0);
            AL::ALValue gesturesRecognized;
            for (int i=0; i<gestures.size(); i++){
                vector<int> gr = gestures[i].existsIn(eventTrajectories[nameIdx], true);
                if (gr.size()>0){
                    gesturesRecognized.arrayPush(gestureNames[i]);
                }
            }
            lastData.arrayPush(gesturesRecognized);
            memoryProxy->raiseMicroEvent(eventNames[nameIdx], lastData);
            memoryProxy->removeMicroEvent(eventNames[nameIdx]);
            eventNames.erase(eventNames.begin()+nameIdx);
            eventObjectIds.erase(eventObjectIds.begin()+nameIdx);
            eventTrajectories.erase(eventTrajectories.begin()+nameIdx);
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
            qiLogInfo("NAOObjectGesture") << "Loaded " << images.size() << " images" << std::endl;
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
                        gesturesRecognized.arrayPush(impl->gestureNames[i]);
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
    for (int i=0; i<impl->eventNames.size(); i++){
        if (name.compare(impl->eventNames[i])==0){
            if (impl->eventObjectIds[i]==objId){
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
    impl->eventNames.push_back(name);
    impl->eventObjectIds.push_back(objId);
    Trajectory tempTraj = Trajectory();
    impl->eventTrajectories.push_back(tempTraj);
    impl->objTrackerLock.unlock();
    qiLogInfo("NAOObjectGesture") << "Now tracking object " << objId << " using event " << name << std::endl;
    return true;
}

AL::ALValue NAOObjectGesture::getEventList(){
    impl->objTrackerLock.lock();
    AL::ALValue retval;
    for (int i=0; i< impl->eventNames.size(); i++){
        AL::ALValue elm;
        elm.arrayPush(impl->eventNames[i]);
        elm.arrayPush(impl->eventObjectIds[i]);
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
    for (int i=0; i<impl->gestureNames.size(); i++){
        if (name.compare(impl->gestureNames[i])==0){
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
    Gesture temp(directionList);
    impl->gestures.push_back(temp);
    impl->gestureNames.push_back(name);
    qiLogInfo("NAOObjectGesture") << "Added gesture " << name << std::endl;
    impl->objTrackerLock.unlock();
}

void NAOObjectGesture::removeGesture(const string &name){
    impl->objTrackerLock.lock();
    for (int i=0; i<impl->gestureNames.size(); i++){
        if (name.compare(impl->gestureNames[i])==0){
            impl->gestures.erase(impl->gestures.begin()+i);
            impl->gestureNames.erase(impl->gestureNames.begin()+i);
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
    for (int i=0; i<impl->gestureNames.size(); i++){
        retval.arrayPush(impl->gestureNames[i]);
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
