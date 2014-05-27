#ifndef NAOOBJECTGESTURE
#define NAOOBJECTGESTURE

#include <alvalue/alvalue.h>
#include <alcommon/almodule.h>
#include <alproxies/almemoryproxy.h>
#include "boost/smart_ptr.hpp"
#include "boost/filesystem.hpp"
#include <string>
#include "ObjectTracking.hpp"
#include "GestureRecognition.hpp"

namespace AL{class ALBroker; class ALModule;}

class NAOEvent{
public:
    std::string name;
    int objectId;
    Trajectory trajectory;
    NAOEvent(std::string tName, int tObjectId);
    NAOEvent(std::string tName, int tObjectId, std::vector<float> num, std::vector<float> den);
    ~NAOEvent();
    void notify(boost::shared_ptr<AL::ALMemoryProxy> memoryProxy, AL::ALValue value,  std::vector<Gesture> gestures);
    void deadNotify(boost::shared_ptr<AL::ALMemoryProxy> memoryProxy, std::vector<Gesture> gestures);
    void log(std::vector<Gesture> gestures);
    void log();
};

class NAOObjectGesture : public AL::ALModule {
public:
    NAOObjectGesture(boost::shared_ptr<AL::ALBroker> pBroker, const std::string& pName);
    ~NAOObjectGesture();
    virtual void init();
    void exit();

    void startTracker(const int &milli, const int &camIdx);
    void stopTracker();

    void loadDataset(const std::string& dataFolder);
    AL::ALValue getObjectList(const int& dataCode);
    AL::ALValue getObjectData(const AL::ALValue &objectIds, const int& dataCode);
    bool trackObject(const std::string& name, const int& objId);
    bool focusObject(const int &objId);
    void stopFocus();

    void addGesture(const std::string &name, const AL::ALValue &dirList);
    void removeGesture(const std::string &name);

    AL::ALValue getGestureList();
    AL::ALValue getEventList();
    bool removeEvent(const std::string &name);
    void removeObjectKind(const int &id);
    void clearEventTraj(const std::string &name);
private:
    struct Impl;
    boost::shared_ptr<Impl> impl;
};

#endif
