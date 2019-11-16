import cv2
import numpy as np
import detect
import visualization
from web_camera import Camera
import image
import os
from spy_camera import OpenCVCam

detect_H = True


def case_spy_cameras(model):
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
        return image.bitwise_images(img, seg_map)
    count = 0
    if os.listdir("man_dataset"):
       count = int(max(os.listdir("man_dataset"), key=lambda x: int(x[:-4]))[:-4])

    cam1 = OpenCVCam("rtsp://10.100.43.15:554/stander/livestream/0/0")
    cam2 = OpenCVCam("rtsp://10.100.43.16:554/stander/livestream/0/0")
    while True:
        result = None
        original_im = cam1.get_image()
        resized_im, seg_map = model.run(original_im)
        try:
            # 15 - person category
            mask = seg_map == 15
            mask = cv2.erode(mask, kernel=np.ones(5, 5, dtype=np.uint8))
            mask = cv2.dilate(mask, kernel=np.ones(5, 5, dtype=np.uint8))
            result = cut_holst_from_bin_roi(cv2.resize(mask, (1920, 1080)), original_im)
            cv2.imwrite(os.path.join("man_dataset", "{}.jpg".format(count)),
                        image.bitwise_images(original_im, seg_map))
        except Exception:
            pass
        if result is not None:
            cv2.imshow("res_img1", result)
            cv2.waitKey(1)
            count += 1

        original_im = cam2.get_image()
        resized_im, seg_map = model.run(original_im)
        try:
            # 15 - person category
            mask = seg_map == 15
            mask = cv2.erode(mask, kernel=np.ones(5, 5, dtype=np.uint8))
            mask = cv2.dilate(mask, kernel=np.ones(5, 5, dtype=np.uint8))
            result = cut_holst_from_bin_roi(cv2.resize(mask, (1920, 1080)), original_im)
            cv2.imwrite(os.path.join("man_dataset", "{}.jpg".format(count)),
                        image.bitwise_images(original_im, seg_map))
        except Exception:
            pass

        if result is not None:
            cv2.imshow("res_img2", result)
            cv2.waitKey(1)

            count += 1


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



