import cv2
import numpy as np
import time


class OpenCVCam(object):
    def __init__(self, source=0):
        self.video_source = source  # for ip-camera source
        self.cap = cv2.VideoCapture(self.video_source)

    def get_image(self):
        _, res = self.cap.read()
        if not _:
            self.cap = cv2.VideoCapture(self.video_source)
            _, res = self.cap.read()
        return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    def grab(self):
        pass


if __name__ == "__main__":
    cam1 = OpenCVCam("rtsp://10.100.43.15:554/stander/livestream/0/0")
    cam2 = OpenCVCam("rtsp://10.100.43.16:554/stander/livestream/0/0")

    while True:
        ret1 = cam1.get_image()
        ret2 = cam2.get_image()
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
            image = ret2[1]
            cv2.imshow("2", cv2.resize(image, (w, h)))
            cv2.waitKey(1)
