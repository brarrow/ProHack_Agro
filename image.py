from io import BytesIO

import numpy
from PIL import Image
import numpy as np
import cv2


def load_img(path):
    img_cv = cv2.imread(path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_cv


def show(img):
    cv2.imshow("Image", img)
    cv2.waitKey(1)


def get_cv2_img(img_pil):
    return np.array(img_pil, dtype=np.uint8)


def get_pil_img(img_cv):
    _, encoded_image = cv2.imencode('.png', img_cv)
    return Image.open(BytesIO(encoded_image.tobytes()))


def bitwise_images(img, mask):
    mask = get_cv2_img(mask)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img


def crop_masked(img_masked):
    img_masked.copy()


def cut_holst_from_bin_roi(holst, img, padding=0):
    nz = np.nonzero(holst)
    sh = holst.shape
    minx = min(nz[0]) - padding
    minx = 0 if minx < 0 else minx
    maxx = max(nz[0]) + 1 + padding
    maxx = 0 if maxx > sh[0] else maxx
    miny = min(nz[1]) - padding
    miny = 0 if miny < 0 else miny
    maxy = max(nz[1]) + 1 + padding
    maxy = 0 if maxy > sh[1] else maxy
    seg_map = holst[minx:maxx, miny:maxy]
    img = img[minx:maxx, miny:maxy]
    return bitwise_images(img, seg_map)