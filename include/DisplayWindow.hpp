#ifndef DISPLAYWINDOW
#define DISPLAYWINDOW

#include "opencv2/highgui/highgui.hpp"
#include "ImgProcPipeline.hpp"
#include <iostream>
#include <chrono>
#include </opt/ros/fuerte/include/opencv2/core/core.hpp>

using namespace cv;


class DisplayWindow{
  String windowName;
  Mat lastDispImg;
  Point dragStartL, dragStartR, currentPos;
  bool leftDrag, rightDrag, dragging;
  int mode;
  std::vector<std::vector<int>> pipelineVector;
  std::chrono::time_point<std::chrono::system_clock> clickTime;
  std::vector<ProcessingElement*> processingElements;
  static void staticMouseCallback(int event, int x, int y, int flags, void* param);
  void mouseCallback(int event, int x, int y, int flags, void* param);
  virtual void onLeftClick();
  virtual void onRightClick();
  virtual void onDragStop();
  
  public:
      DisplayWindow(String name);
      DisplayWindow(String name, std::vector<ProcessingElement*> prcElm, std::vector<std::vector<int>> pipelineVec);
      virtual void display(const Mat image);
      virtual void onKeyPress(int key);
};













#endif
