import os

import cv2
import numpy as np

import image
from status import Status
from persons import Person
import uuid
import detect_shapes
from spy_camera import OpenCVCam
from keras.models import load_model
from keras import backend as K
import keras

status_ = Status()


def detecting(resized_im, seg_map, N=False,  H=False, P=False, G=False):
    img_with_mask = image.bitwise_images(resized_im, seg_map)
    res_img = img_with_mask.copy()
    # here need to split persons on image. All bottom methods for one person
    persons_masks = [img_with_mask]
    status_.persons = [Person(mask) for mask in persons_masks]

    for person in status_.persons:
        if N:
            pass
        if H:
            detect_helmets(person)
        if P:
            pass
        if G:
            pass
    # status check for all persons
    status_.check_status()
    status_.clear()
    return res_img


def detect_helmets(seg_person):
    tmp = seg_person.copy()
    tmp = image.crop_top_part(tmp, 16)
    tmp = image.get_white_regions(tmp)
    # print(np.sum(tmp)/(tmp.shape[0] * tmp.shape[1] * 3))
    if 11 < np.sum(tmp)/(tmp.shape[0] * tmp.shape[1] * 3) < 25:
        print("YES! HELMET!")
        exit()
        return True
    else:
        return False


def detect_costume(seg_person):
    tmp = seg_person.copy()
    tmp = image.crop_middle_part(tmp, 5, 5)
    if detect_shapes.get_shapes_for_costume(tmp):
        print("YES! COSTUME!")
        exit()
        return True
    else:
        return False


def detect_glasses(seg_person):
    tmp = seg_person.copy()
    tmp = image.crop_top_part(tmp, 6)
    detect_shapes.get_shapes_for_glasses(tmp)


def detect_cnn_costume(seg_person, keras_model):
        img0 = cv2.resize(seg_person, (100, 256))
        img_x = img0.astype('float32') / 255.0
        new = keras_model.predict_proba(img_x.reshape(1, 256, 100, 3))
        print("Sotr: ", new[0]*100)

def cut_head(person):

    pass


def segm_person(model, cam, imshow_name="image"):
    if isinstance(cam, OpenCVCam):
        original_im = cam.get_image()
    else:
        _, original_im = cam.read()

    if original_im is None:
        return
    resized_im, seg_map = model.run(original_im)

    # 15 - person category
    mask = seg_map == 15
    # no person on image, return
    mask = image.get_cv2_img(mask)
    mask = cv2.erode(mask, kernel=np.ones((5, 5), dtype=np.uint8))
    mask = cv2.dilate(mask, kernel=np.ones((5, 5), dtype=np.uint8))
    if not np.any(mask):
        return
    result = image.cut_holst_from_bin_roi(cv2.resize(mask, (1920, 1080)), original_im)
    return result
