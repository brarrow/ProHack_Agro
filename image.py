from io import BytesIO
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