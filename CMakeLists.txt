cmake_minimum_required(VERSION 2.8)
project(objectGesture)
set(CMAKE_CXX_FLAGS "-std=c++0x")
include_directories("include")

find_package(qibuild)
find_package(OpenCV REQUIRED)
find_package(BOOST)

option(TRACKER_IS_REMOTE
  "module is compiled as a remote module (ON or OFF)"
  OFF)


option(TESTMODE
    "Program is compiled for testing with visual output to screen "
    OFF)
if(TESTMODE)
    ADD_DEFINITIONS("-DTESTMODE")
endif()

option(USENAOCAMERA
    "Connect to NAO's camera over the network."
    OFF)

qi_create_lib(ImgProcPipeline STATIC SRC include/ImgProcPipeline.hpp src/ImgProcPipeline.cpp)
qi_use_lib(ImgProcPipeline BOOST BOOST_FILESYSTEM OPENCV2_CORE OPENCV2_HIGHGUI OPENCV2_IMGPROC OPENCV2_VIDEO)
qi_stage_lib(ImgProcPipeline)

qi_create_lib(ObjectTracking STATIC SRC include/ObjectTracking.hpp src/ObjectTracking.cpp)
qi_use_lib(ObjectTracking BOOST BOOST_FILESYSTEM OPENCV2_CORE OPENCV2_HIGHGUI OPENCV2_IMGPROC OPENCV2_VIDEO ImgProcPipeline)
qi_stage_lib(ObjectTracking)

qi_create_lib(ModuleImpl STATIC include/NAOObjectGesture.h src/NAOObjectGesture.cpp)
qi_use_lib(ModuleImpl ALCOMMON ALVISION ALPROXIES ALERROR BOOST BOOST_THREAD BOOST_FILESYSTEM BOOST_DATE_TIME OPENCV2_CORE OPENCV2_HIGHGUI ImgProcPipeline ObjectTracking)
qi_stage_lib(ModuleImpl)

if(TESTMODE)
    qi_create_lib(DisplayWindow STATIC SRC include/DisplayWindow.hpp src/DisplayWindow.cpp)
    qi_use_lib(DisplayWindow BOOST BOOST_FILESYSTEM OPENCV2_CORE OPENCV2_HIGHGUI ImgProcPipeline ObjectTracking)
    qi_stage_lib(DisplayWindow)

    qi_create_lib(ImageAcquisition STATIC SRC include/ImageAcquisition.h src/ImageAcquisition.cpp)
    qi_use_lib(ImageAcquisition OPENCV2_CORE OPENCV2_HIGHGUI ALCOMMON ALVISION ALPROXIES)
    qi_stage_lib(ImageAcquisition)

    if(USENAOCAMERA)
        qi_create_bin(nao-object-gesture main_remoteNAO.cpp)
        qi_use_lib(nao-object-gesture ImgProcPipeline ObjectTracking DisplayWindow ImageAcquisition ALCOMMON ALVISION ALPROXIES ALERROR)
    else()
        qi_create_bin(nao-object-gesture src/main.cpp)
        qi_use_lib(nao-object-gesture ImgProcPipeline ObjectTracking DisplayWindow ImageAcquisition ALCOMMON ALVISION ALPROXIES ALERROR)
    endif()
else()
    if(TRACKER_IS_REMOTE)
        add_definitions(" -DTRACKER_IS_REMOTE ")
        qi_create_bin(nao-object-gesture SHARED src/naoqi_module_loader.cpp SUBFOLDER naoqi)
        qi_use_lib(nao-object-gesture ImgProcPipeline ObjectTracking ModuleImpl ALCOMMON ALVISION ALPROXIES ALERROR)

    else()
        qi_create_lib(nao-object-gesture SHARED src/naoqi_module_loader.cpp SUBFOLDER naoqi)
        qi_use_lib(nao-object-gesture ImgProcPipeline ObjectTracking ModuleImpl BOOST ALCOMMON ALVISION ALPROXIES ALERROR)
    endif()
endif()

