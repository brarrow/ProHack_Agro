import os
import cv2
import json

import numpy as np
from flask import Flask, Response, request

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from deeplabmodel import DeepLabModel
import cases
from spy_camera import OpenCVCam

models = {"mobnet": "mobnetv2.pb",
          "xception": "xception.pb"}
model = DeepLabModel(models["mobnet"])


class Server(object):
    glasses = None

    def __init__(self, _port):
        # setup Main loop
        # setup checker
        # setup Flask
        self.app = Flask(__name__)
        self.init_flask()
        self.glasses = None

    def init_flask(self):
        """

        Rules for apply get request.

        """
        self.app.add_url_rule('/video_feed', 'video_feed',
                              lambda: Response(self.generate_image("not_counter"),
                                               mimetype='multipart/x-mixed-replace; '
                                                        'boundary=frame'))
        self.app.add_url_rule('/glass', 'glass',
                              lambda: Response(self.glass(request.args),
                                               mimetype='application/json; '))

    @staticmethod
    def generate_image(type):
        """Video streaming generator function."""
        while True:
            try:
                frame = cv2.imencode('.jpg', cases.case_web(model))[1].tobytes()
            except Exception:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def run(self):
        self.app.run(host="0.0.0.0", port=5000, threaded=True)

    def glass(self, args):
        Server.glasses = [args.get("xmin"), args.get("ymin"), args.get("xmax"), args.get("ymax")]
        print(Server.glasses)
        return {"status": "OK"}


s = Server(5000)
s.run()

# img_path = "man_dataset/form/0d53018d-451c-40fe-bca2-6e5580851bba.jpg"
# cases.case_test_image(img_path)

# cases.case_images(model)

# cases.case_spy_cameras(model)

# cases.case_video(model)

# cam1 = OpenCVCam("rtsp://10.100.43.15:554/stander/livestream/0/0")
# cam2 = OpenCVCam("rtsp://10.100.43.16:554/stander/livestream/0/0")

# cases.case_cut_data()
# cases.case_write_video("cam1", cam1)
# cases.case_segment_video(model)