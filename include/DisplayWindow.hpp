#ifndef DISPLAYWINDOW
#define DISPLAYWINDOW

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/ref.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread/pthread/condition_variable.hpp>
#include <boost/thread/pthread/mutex.hpp>
#include "boost/filesystem/fstream.hpp"

#include "ImgProcPipeline.hpp"
#include "ImageAcquisition.h"
#include <iostream>
#include <chrono>

using namespace cv;


/*! User interface class that implements an image processing pipeline.
  *
  */
class DisplayWindow{
    /*! Name of window to display images in*/
    String windowName;

    /*! Temporary container for the latest received unprocessed image*/
    Mat dispImg;
    /*! Mutex used for threadsafe access to the dispImg variable*/
    boost::mutex imLock;

    /*! A directory containing images to process on demand.*/
    std::string dirname;
    /*! Directory iterator pointing to the next image file to load*/
    boost::filesystem::directory_iterator itr;
    /*! Path to the next image file to load*/
    boost::filesystem::path filename;

    /** Internal user interface variables*/
    Point dragStartL, dragStartR, currentPos;
    bool leftDrag, rightDrag, dragging;
    std::chrono::time_point<std::chrono::system_clock> clickTime;

    /*! Processing mode, corresponds to an image pipeline stored in pipelineVector*/
    int mode;
    /*! Ordered lists of processing element indices, each of which defines a single image processing pipeline*/
    std::vector<std::vector<int > > pipelineVector;
    /*! All processing elements available for use*/
    std::vector<ProcessingElement*> processingElements;

    /*! A static mouse callback function.
      * This function simply calls the non-static version of the callback function.
      * \param event Id of the event that triggered the callback
      * \param x x Coordinate of the cursor at the time the event occured
      * \param y y Coordinate of the cursor at the time the event occured
      * \param flags OpenCV flags
      * \param param Pointer to user data
      */
    static void staticMouseCallback(int event, int x, int y, int flags, void* param);
    /*! A mouse callback function.
      * This function handles mouse events and calls the appropriate on-functions.
      * \param event Id of the event that triggered the callback
      * \param x x Coordinate of the cursor at the time the event occured
      * \param y y Coordinate of the cursor at the time the event occured
      * \param flags OpenCV flags
      * \param param Pointer to user data
      */
    void mouseCallback(int event, int x, int y, int flags, void* param);

    /*! Overloadable response to left-click.*/
    virtual void onLeftClick();
    /*! Overloadable response to right-click.*/
    virtual void onRightClick();
    /*! Overloadable response to click-and drag.
      * Check leftDrag and rightDrag attributes if it is relevant which mouse button was held down.
      */
    virtual void onDragStop();
    /*! Overloadable response to keypress. */
    virtual void onKeyPress(int key);

    /*! Advances the directory iterator to the next non-directory file.*/
    bool nextImage();

public:
    /*! Thread object used to display dispImg and respond to user input*/
    boost::thread *t;
    /*! Simple boolean variable to check if thread is still running*/
    bool running;

    /*! Simple visualizer constructor.
      * This constructor produces a window which does no processing. Useful when all that is needed is to
      * display an image stream or to scroll through a folder full of images.
      * \param name Name of window to display images in
      */
    DisplayWindow(String name);
    /*! Window constructor with processing elements.
      * This constructor produces a window with a defined set of processing elements arranged into pipelines. The order
      * in which the processing elements are arranged in their own vector doesn't matter - they are applied in the order they
      * are listed in the pipeline vector.
      * \param name Name of window to display images in
      * \param prcElm Vector of processing elements to use.
      * \param pipelineVec Vector of pipelines
      */
    DisplayWindow(String name, std::vector<ProcessingElement*> prcElm, std::vector<std::vector<int > > pipelineVec);
    /*! Makes the object callable to enable threaded execution*/
    void operator()();

    /*! Displays the passed image until a new image is passed.
      * This function is fully threadsafe and should be used to push new images as fast as they can be acquired from
      * the image source. The image will be stored, processed and displayed.
      * \param image Image to display and process
      */
    virtual void display(const Mat image);
    /*! Processes an input image using a specific pipeline.
      * This function is used internally to process images but can also be called from outside to process a single image.
      * \param inputImage Image to process
      * \param outputImage Matrix to store the result into
      * \param tMode Pipeline to use for processing (tMode = 1 corresponds to pipeline[0])
      */
    virtual void process(Mat inputImage, Mat &outputImage, int tMode);

    /*! Sets a folder to display images from.
      * Use this function when the user should be able to advance to the next image manually. The directory is not traversed
      * recursively - only the root directory is searched for images.
      * \param dir Full path to directory containing only images or other directories.
      */
    void setImageFolder(std::string dir);
};
#endif
