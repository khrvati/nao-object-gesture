import sys
import motion
import almath
import math
import Image, ImageDraw
from naoqi import ALProxy, ALBroker, ALModule
import numpy as np

class ObjectTrackerModule(ALModule):
    def __init__(self, name):
        ALModule.__init__(self, name)
        self.tts = ALProxy("ALTextToSpeech")
        self.gestureProxy = ALProxy("NAOObjectGesture", myBroker)
        self.motionProxy = ALProxy("ALMotion", myBroker)
        self.memProxy = ALProxy("ALMemory", myBroker)

        self.motionProxy.setStiffnesses("Head", 1.0)
        self.gestureProxy.startTracker(15, 0)

        self.gestureProxy.addGesture("Drink", [2,6])
        self.gestureProxy.addGesture("FrogL", [1,0,7])
        self.gestureProxy.addGesture("FrogR", [3,4,5])

    def startTracker(self, camId):
        self.gestureProxy.startTracker(15, camId)
        self.gestureProxy.focusObject(-1)

    def stopTracker(self):
        self.gestureProxy.stopTracker()
        self.gestureProxy.stopFocus()

    def load(self, path, name):
        self.gestureProxy.loadDataset(path)
        self.gestureProxy.trackObject(name, -len(self.kindNames))
        self.memProxy.subscribeToMicroEvent(name, "ObjectTracker", name, "onObjGet")

    def onObjGet(self, key, value, message):
        id = -1
        if (key in self.kindNames):
            id = self.kindNames.index(key)
        else:
            return
        if (value != None):
            if (value[0] != 0):
                if (value[5]!=None):
                    print (value[5])
            else:
                if (value[1]!=None):
                    print (value[1])

    def unload(self):
        self.gestureProxy.stopTracker()
        for i in range(0, len(self.exists)):
            self.gestureProxy.removeObjectKind(0)
            self.gestureProxy.removeEvent(self.kindNames[i])
        self.gestureProxy.removeGesture("Drink")
        self.gestureProxy.removeGesture("FrogL")
        self.gestureProxy.removeGesture("FrogR")


if __name__ == '__main__':
    ObjectTracker = ObjectTrackerModule("ObjectTracker")
    ObjectTracker.load("/home/nao/ImageSets/"+folder , objectdef)
    ObjectTracker.startTracker(1)
    



