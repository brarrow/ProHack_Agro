import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from deeplabmodel import DeepLabModel
import cases

models = {"mobnet": "mobnetv2.pb",
          "xception": "xception.pb"}
model = DeepLabModel(models["mobnet"])

# cases.case_web_camera(model)


# img_path = "images/one_H_P.jpg"
# cases.case_image(model, img_path)

# cases.case_images(model)

cases.case_spy_cameras(model)

# cases.case_video_cameras(model)