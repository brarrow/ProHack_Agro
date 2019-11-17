import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from deeplabmodel import DeepLabModel
import cases
from multiprocessing import Process
from spy_camera import OpenCVCam
from threading import Thread


models = {"mobnet": "mobnetv2.pb",
          "xception": "xception.pb"}
model = DeepLabModel(models["mobnet"])


# img_path = "man_dataset/form/c474c227-b16b-4eec-943a-d698e0162007.jpg"
# cases.case_test_image(img_path)

# cases.case_images(model)

cases.case_spy_cameras(model)

# cases.case_video(model)

cam1 = OpenCVCam("rtsp://10.100.43.15:554/stander/livestream/0/0")
cam2 = OpenCVCam("rtsp://10.100.43.16:554/stander/livestream/0/0")

# cases.case_write_video("cam2", cam2)
# cases.case_segment_video(model)