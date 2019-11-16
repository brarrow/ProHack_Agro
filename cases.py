import detect
import visualization
from camera import Camera
import image
import os

detect_H = True

def case_camera(model):
    cam = Camera()
    while True:
        original_im = cam.get_image()
        resized_im, seg_map = model.run(original_im)
        res_img = detect.detecting(resized_im, seg_map, H=True)
        visualization.vis_segmentation_cv(res_img)


def case_image(model, img_path):
    original_im = image.load_img(img_path)
    resized_im, seg_map = model.run(original_im)
    res_img = detect.detecting(resized_im, seg_map, H=True)
    visualization.vis_segmentation_cv(res_img)


def case_images(model):
    rel_path = "images"
    for f in os.listdir(rel_path):
        original_im = image.load_img(os.path.join(rel_path, f))
        resized_im, seg_map = model.run(original_im)
        res_img = detect.detecting(resized_im, seg_map, H=True)
        visualization.vis_segmentation_cv(res_img)
