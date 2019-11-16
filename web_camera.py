from io import BytesIO

import cv2


class Camera(object):
    def __init__(self, device_id=0):
        self.camera = cv2.VideoCapture(device_id)

    def get_image(self):
        _, img = self.camera.read()
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
