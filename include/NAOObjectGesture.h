#ifndef NAOOBJECTGESTURE
#define NAOOBJECTGESTURE

#include <alcommon/almodule.h>
#include "boost/smart_ptr.hpp"
#include <string>


namespace AL{class ALBroker;}

class NAOObjectGesture : public AL::ALModule {
public:
    NAOObjectGesture(boost::shared_ptr<AL::ALBroker> pBroker, const std::string& pName);
    ~NAOObjectGesture();
    virtual void init();
    void exit();

    void startTracker(const int &milli);
    void stopTracker();

    void loadDataset(const std::string dataFolder);
    AL::ALValue getObjectData(std::vector<int> objectIds, const int dataCode);
    AL::ALValue getObjectList(const int dataCode);
    bool trackObject(const std::string name, const int objId);
private:
    struct Impl;
    boost::shared_ptr<Impl> impl;
};

#endif
