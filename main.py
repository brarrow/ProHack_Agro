import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deeplabmodel import DeepLabModel
import cases
from multiprocessing import Process
from spy_camera import OpenCVCam
from threading import Thread


models = {"mobnet": "mobnetv2.pb",
          "xception": "xception.pb"}
model = DeepLabModel(models["mobnet"])

# cases.case_web_camera(model)


# img_path = "images/one_H_P.jpg"
# cases.case_image(model, img_path)

# cases.case_images(model)

# cases.case_spy_cameras(model)

cases.case_video(model)


# cam1 = OpenCVCam("rtsp://10.100.43.15:554/stander/livestream/0/0")
# cam2 = OpenCVCam("rtsp://10.100.43.16:554/stander/livestream/0/0")

# process1 = Thread(target=cases.case_write_video, args=("cam1", cam1))
# process1.run()
# process2 = Thread(target=cases.case_write_video, args=("cam2", cam2))
# process2.run()