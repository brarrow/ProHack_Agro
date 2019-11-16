import detect
import visualization
from web_camera import Camera
import image
import os
from spy_camera import OpenCVCam

detect_H = True


def case_spy_cameras(model):
    cam1 = OpenCVCam("rtsp://10.100.43.15:554/stander/livestream/0/0")
    cam2 = OpenCVCam("rtsp://10.100.43.16:554/stander/livestream/0/0")
    while True:
        original_im = cam1.get_image()
        resized_im, seg_map = model.run(original_im)
        res_img = detect.detecting(resized_im, seg_map, H=True)
        visualization.vis_segmentation_cv(res_img=res_img)
        # visualization.vis_segmentation_cv(resized_im, res_img)


def case_video(model):
    vid = OpenCVCam("videos/1.avi")
    while True:
        original_im = vid.get_image()
        resized_im, seg_map = model.run(original_im)
        res_img = detect.detecting(resized_im, seg_map, H=True)
        visualization.vis_segmentation_cv(res_img=res_img)
        # visualization.vis_segmentation_cv(resized_im, res_img)


def case_web_camera(model):
    cam = Camera()
    while True:
        original_im = cam.get_image()
        resized_im, seg_map = model.run(original_im)
        res_img = detect.detecting(resized_im, seg_map, H=True)
        visualization.vis_segmentation_cv(resized_im, res_img)


def case_image(model, img_path):
    original_im = image.load_img(img_path)
    resized_im, seg_map = model.run(original_im)
    res_img = detect.detecting(resized_im, seg_map, H=True)
    visualization.vis_segmentation_cv(resized_im, res_img)


def case_images(model):
    rel_path = "images"
    for f in os.listdir(rel_path):
        original_im = image.load_img(os.path.join(rel_path, f))
        resized_im, seg_map = model.run(original_im)
        res_img = detect.detecting(resized_im, seg_map, H=True)
        visualization.vis_segmentation_cv(resized_im, res_img)



