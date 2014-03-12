add_library(ColorSegmenter STATIC ColorSegmenter.cpp)
add_executable(main main.cpp)

target_link_libraries(ColorSegmenter ${OpenCV_LIBS})
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main ColorSegmenter)
