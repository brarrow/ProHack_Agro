import os

import cv2
import numpy as np

import image
from status import Status
from persons import Person
import uuid
import detect_shapes
from spy_camera import VideoStreamWidget
from keras.models import load_model
from keras import backend as K
import keras
# import Glassess

status_ = Status()


def detecting(resized_im, seg_map, N=False,  H=False, P=False, G=False):

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
    try:
        tmp = seg_person.copy()
        tmp = image.crop_top_part(tmp, 16)
        tmp = image.get_white_regions(tmp)
        print(np.sum(tmp)/(tmp.shape[0] * tmp.shape[1] * 3))

        if 12 < np.sum(tmp)/(tmp.shape[0] * tmp.shape[1] * 3) < 50:
            return True
        else:
            return False
    except:
        return True


# def detect_glasses(seg_person):
#     categories = Glassess.run_detector(seg_person)
#     if "glasses" in categories or "sunglasses" in categories:
#         return True


def detect_costume(seg_person):
    tmp = seg_person.copy()
    tmp = image.crop_middle_part(tmp, 5, 3)
    if detect_shapes.get_shapes_for_costume(tmp):
        print("YES! COSTUME!")
        return True
    else:
        return False

def detect_glasses_t(img):
    img_copy = img[0:img.shape[0]//3]
    # cv2.imshow("123", img_copy)
    # cv2.waitKey(0)

    img_copy = cv2.threshold(img_copy, 10, 255,  cv2.THRESH_BINARY)[1]
    cv2.imshow("123", img_copy)
    cv2.waitKey(1)


# def detect_glasses(seg_person):
#     tmp = seg_person.copy()
#     tmp = image.crop_top_part(tmp, 6)
#     detect_shapes.get_shapes_for_glasses(tmp)


def detect_cnn_costume(seg_person, keras_model):
        img0 = cv2.resize(seg_person, (100, 256))
        img_x = img0.astype('float32') / 255.0
        new = keras_model.predict_proba(img_x.reshape(1, 256, 100, 3))
        if new[0][0] > new[0][1]:
            return False
        else:
            return True


def segm_person(model, cam, detect_H, detect_G, detect_C):
    try:
        if isinstance(cam, VideoStreamWidget):
            original_im = cam.get_image()
        else:
            _, original_im = cam.read()
    except:
        return
    if original_im is None:
        return
    resized_im, seg_map = model.run(original_im)

    # 15 - person category
    mask = seg_map == 15
    # no person on image, return
    mask = image.get_cv2_img(mask)
    cv2.imshow("r", mask)
    cv2.waitKey(1)
    if not np.any(mask):
        return
    else:
        mask = cv2.resize(mask, (1920, 1080))
        # cv2.imshow("r2", cv2.resize(mask * 255, (400,400)))
        # cv2.waitKey(1)

        mask[:, :600] = 0
        # cv2.imshow("r3", cv2.resize(mask* 255, (400,400)))
        # cv2.waitKey(1)
    result, (y, x, yh, xw) = image.cut_holst_from_bin_roi(mask, original_im)
    if detect_C:
        text = "Employee. Helmet: {}".format(detect_H)
    else:
        text = "Guest. Helmet: {}".format(detect_H)
    if detect_H and detect_C:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv2.rectangle(original_im, (x, y), (xw, yh), color, 2)
    # cv2.putText(original_im, text, (x, y), 0, 1, (0, 255, 0))
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    # set the rectangle background to white
    rectangle_bgr = (255, 255, 255)
    # make a black image
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    # set the text start position
    text_offset_x = x
    text_offset_y = y
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(original_im, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(original_im, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)
    return original_im, result
