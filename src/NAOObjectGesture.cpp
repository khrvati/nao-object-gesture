#include "NAOObjectGesture.h"
#include <iostream>
#include <fstream>
#include <alvalue/alvalue.h>
#include <alcommon/alproxy.h>
#include <alcommon/albroker.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>

#include <boost/thread/pthread/condition_variable.hpp>
#include <boost/thread/pthread/mutex.hpp> //remove

#include <boost/ref.hpp>
#include <boost/lambda/lambda.hpp>
#include <qi/log.hpp>
#include "include/ObjectTracking.hpp"


struct NAOObjectGesture::Impl{
    boost::shared_ptr<ObjectTracker> objectTracker;
    NAOObjectGesture &module;
    boost::mutex fileLock;

    boost::posix_time::time_duration samplingPeriod;
    boost::thread *t;
    bool stopThread;
    boost::mutex stopThreadLock;
    boost::condition_variable condVar;


    Impl(NAOObjectGesture& mod)
        : module(mod), t(NULL), samplingPeriod(boost::posix_time::milliseconds(34))
    {
        objectTracker = boost::shared_ptr<ObjectTracker>(new ObjectTracker());
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

            boost::posix_time::time_duration duration = boost::get_system_time() - time_t_epoch;
            long long timestampNanos = duration.total_nanoseconds();
            //this is time since epoch

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

}
