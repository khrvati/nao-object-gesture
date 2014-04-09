add_library(ImgProcPipeline STATIC ImgProcPipeline.cpp)
target_link_libraries(ImgProcPipeline ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_library(ObjectTracking STATIC ObjectTracking.cpp)
target_link_libraries(ObjectTracking ${Boost_LIBRARIES} ${OpenCV_LIBS} ImgProcPipeline)

add_library(DisplayWindow STATIC DisplayWindow.cpp)
target_link_libraries(DisplayWindow ${Boost_LIBRARIES} ${OpenCV_LIBS} ImgProcPipeline ObjectTracking)

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${Boost_LIBRARIES} DisplayWindow ImgProcPipeline ObjectTracking)
