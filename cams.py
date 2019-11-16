import cv2
import numpy as np
import time

class OpenCVCam(object):
    def __init__(self, source=0):
        self.video_source = source  # for ip-camera source
        self.cap = cv2.VideoCapture(self.video_source)

    def read(self):
        time.sleep(0.04)
        res = self.cap.read()
        if not res[0]:
            self.cap = cv2.VideoCapture(self.video_source)
            res = self.cap.read()
        return res

    def grab(self):
        pass


cam1 = OpenCVCam("rtsp://10.100.43.15:554/stander/livestream/0/0")
cam2 = OpenCVCam("rtsp://10.100.43.16:554/stander/livestream/0/0")

while True:
    ret1 = cam1.read()
    ret2 = cam2.read()
    if ret1[0]:
        print(ret1[1].shape)
        shape = ret1[1].shape[0] / ret1[1].shape[1]
        w = 800
        h = int(w * shape)
        cv2.imshow("1", cv2.resize(ret1[1], (w, h)))
        cv2.waitKey(1)
    if ret2[0]:
        shape = ret1[1].shape[0] / ret1[1].shape[1]
        w = 800
        h = int(w * shape)
        # image = ret2[1]
        image = flipVertical = cv2.flip(ret2[1], 0)
        cv2.imshow("2", cv2.resize(image, (w, h)))
        cv2.waitKey(1)