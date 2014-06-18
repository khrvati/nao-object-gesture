nao-object-gesture
===========================

1. Introduction
----------------------------------------
NAO object gesture is a local module for the NAO humanoid robot. It provides:

* Color-based detection of objects based on a model learned from a set of image-object mask pairs
* Real-time tracking of an unlimited number of strongly interacting objects
* Automatic head tracking of any tracked object
* Gesture recognition invariant to robot head motion for an arbitrary number of user-specified gestures

The module can run on any NAO robot with firmware 1.14 or higher, and has a simple interface accesible from C++ or Python, taking advantage of the naoqi framework.


2. Requirements
----------------------------------------

To successfully build this module you will need:

* CMake version 2.8 or higher [http://www.cmake.org/]
* Boost version 1.46 or higher [http://www.boost.org/]
* OpenCV version 2.5 or higher [http://opencv.org/]
* qibuild installed and configured [https://community.aldebaran-robotics.com/doc/1-14/qibuild/getting_started.html#qibuild-getting-started]
* NAOQI C++ Cross Toolchain 1.14.5 for NAO 4 [https://community.aldebaran.com/]
* NAOQI C++ SDK for your system, if building in test mode

3. Compilation
----------------------------------------
These instructions are written for Linux distributions. A similar process might work for Windows or Mac systems.

Once qibuild, Boost and OpenCV are installed and able to be found by CMake, clone this git repo into a folder of your choosing. The module can be built as a remote application that connects to the robot over the network, for debug and display purposes, by setting the TESTMODE option in CMakeLists.txt to ON. If the module should be compiled as a remote module, set this option to OFF. Once you have done this, the module can be compiled.

First setup your qibuild by running
```
qibuild init
```
In the project's root folder. Then create a toolchain to build the module with. If building as a local module, it is recommended that you name the toolchain 'atom114' to be able to use the provided shell script for automatic code compilation and uploading to the robot. Do this by executing the following command:
```
qitoolchain create atom114 /path/to/your/NAOQI/Cross/Toolchain
```
Then you can either use the provided makefornao.sh shell script by running
```
sh makefornao.sh 127.0.0.1
```
and replacing "127.0.0.1" by your robot's network IP to automatically configure, make and upload the module to the proper folder.

If you would prefer not to do this, first configure the project using your preferred toolchain in release mode, then make it and upload it to the robot's /home/nao/naoqi/modules folder

Lastly, you should edit your robot's autoload.ini file, located in /home/nao/naoqi/preferences, and add /home/nao/naoqi/modules/libnao-object-gesture.so to your autoload list of moduler. Reset the robot's naoqi (using the robot's webpage or by running 'nao stop' and then 'nao start' on the robot) and you are ready to use the module.

4. Module use
--------------------------------------

Each class, attribute and method used by the module is fully documented using Doxygen. The external interface for the module can most easily be accessed while the module is running, by entering the robot's webpage and clicking on the module's name under "naoqi". This brings up the entire documentation for each externally accessible function.

Please also refer to the provided examples in Python for additional insight into the module's use.








