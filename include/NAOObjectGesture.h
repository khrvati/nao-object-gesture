#ifndef NAOOBJECTGESTURE
#define NAOOBJECTGESTURE

#include <alcommon/almodule.h>
#include <alcommon/albrokermanager.h>
#include <alcommon/altoolsmain.h>
#include "boost/smart_ptr.hpp"
#include <string>


namespace AL{class ALBroker;}

class NAOObjectGesture : public AL::ALModule {
public:
    NAOObjectGesture(boost::shared_ptr<AL::ALBroker> pBroker, const std::string& pName);
    ~NAOObjectGesture();
    virtual void init();

private:
    struct Impl;
    boost::shared_ptr<Impl> impl;
};

#endif
