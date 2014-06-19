nao-object-gesture
===========================

1. Introduction
----------------------------------------
NAO object gesture is a local module for the NAO humanoid robot. It provides:

* Color-based detection of objects based on a model learned from a set of image-object mask pairs
* Real-time tracking of an unlimited number of strongly interacting objects
* Automatic head tracking of any tracked object
* Gesture recognition invariant to robot head motion for an arbitrary number of user-specified gestures

The module can run on any NAO robot with firmware 1.14 , and has a simple interface accesible from C++ or Python, taking advantage of the naoqi framework.


2. Requirements
----------------------------------------

To successfully build this module you will need:

* CMake version 2.8 or higher [http://www.cmake.org/]
* Boost version 1.46 or higher [http://www.boost.org/]
* OpenCV version 2.5 or higher [http://opencv.org/]
* The latest version of qibuild [https://community.aldebaran-robotics.com/doc/qibuild/beginner/getting_started.html]
* NAOQI C++ Cross Toolchain 1.14.5 for NAO 4 [https://community.aldebaran.com/]
* NAOQI C++ SDK for your system, if building in test mode

3. Compilation
----------------------------------------

### 3.1 Simple local module compilation
These instructions are written for Linux distributions. A similar process might work for Windows or Mac systems.

Once Boost and OpenCV are installed and able to be found by CMake, install qibuild by running
```
~$ sudo pip install qibuild
```
Then create an empty worktree folder for your qibuild projects in a location of your choice and navigate so that you are inside the folder. You can now initialize the folder as your qibuild worktree by running:
```
~/worktree$ qibuild init
```
Next, create a toolchain to build the module with. If building as a local module, it is recommended that you name the toolchain 'atom114' to be able to use the provided shell script for automatic code compilation and uploading to the robot. Do this by executing the following command:
```
~/worktree$ qitoolchain create atom114 /path/to/your/NAOQI/Cross/Toolchain/**toolchain.xml**
```
You can now clone the repo into your worktree. This is done by running
```
~/worktree$ git clone git@github.com:khrvati/nao-object-gesture
```
Next, convert the project for use with qibuild.
```
~/worktree$ cd nao-object-gesture
~/worktree/nao-object-gesture$ qibuild convert --go
```
You can now run the provided shell script, replacing NAO_ROBOT_IP by your robot's network IP to automatically configure, make and upload the module to the proper folder.
```
~/worktree/nao-object-gesture$ sh makefornao.sh NAO_ROBOT_IP
```

Lastly, you should edit your robot's autoload.ini file, located in /home/nao/naoqi/preferences, and add /home/nao/naoqi/modules/libnao-object-gesture.so to your autoload list of moduler. Reset the robot's naoqi (using the robot's webpage or by running 'nao stop' and then 'nao start' on the robot) and you are ready to use the module.

### 3.1 Remote test mode compilation
The above instructions will work for NAO version 1.14 and a local module. If you would like to compile the module as a remote application which connects to to the robot over the network, first change the TESTMODE option in CMakeLists.txt to ON. Then create a new toolchain for remote building, by running
```
~/worktree$ qitoolchain create remote-toolchain /path/to/your/NAOQI/C++/SDK/**toolchain.xml**
```
You can now configure and make the module by running

```
~/worktree$ cd nao-object-gesture
~/worktree/nao-object-gesture$ qibuild configure -c remote-toolchain
~/worktree/nao-object-gesture$ qibuild make -c remote-toolchain
```

For the module to run remotely, the config.ini file provided in the examples directory is needed. Copy this file into your build directory and edit it to reflect your NAO's network IP and port. Also, set the ImageDirectory parameter to point to a directory containing a training image set up as explained in 4. Module use. The tracking component will not work if you fail to provide a valid image directory.
If you would like to use a connected webcam instead of the NAO robot's camera, set the UseLocalCamera parameter to 1 and the Camera parameter to the hardware ID of the camera you would like to use. If you have a set of images you would like to test the segmentation on, set the UseImageSequence parameter to 1 and the ImageSequence parameter to point to a directory containing the images to be displayed and no other files or folders.

4. Module use
--------------------------------------

Each class, attribute and method used by the module is fully documented using Doxygen. The external interface for the module can most easily be accessed while the module is running, by entering the robot's webpage and clicking on the module's name under "naoqi". This brings up the entire documentation for each externally accessible function.

Please also refer to the provided examples in Python for additional insight into the module's use.








