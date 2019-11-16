import detect
import visualization
from detect import segm_person
from image import cut_holst_from_bin_roi
from web_camera import Camera
import image
import os
from spy_camera import OpenCVCam
import cv2
import time


detect_H = True


def case_spy_cameras(model):
    if not os.path.exists("man_dataset"):
        os.mkdir("man_dataset")

    cam1 = OpenCVCam("rtsp://10.100.43.15:554/stander/livestream/0/0")
    cam2 = OpenCVCam("rtsp://10.100.43.16:554/stander/livestream/0/0")
    print("In cycle, stream.")
    while True:
       segm_person(model, cam1, "res_img1")
       segm_person(model, cam2, "res_img2")


def case_video(model):
    vid = OpenCVCam("videos/no_glass_no_helmet/cam2.avi")
    while vid.cap.isOpened():
        segm_person(model, vid)
        # original_im = vid.get_image()
        # resized_im, seg_map = model.run(original_im)
        # res_img = detect.detecting(resized_im, seg_map, H=True)
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


def case_write_video(name, cap):

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('{}.avi'.format(name), fourcc, 20.0, (1920,1080))
    # out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))
    start_time = time.time()
    while time.time() - start_time < 40:
        frame = cap.get_image()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # write the flipped frame
        out.write(frame)

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release everything if job is finished
    # cap.release()
    out.release()
    cv2.destroyAllWindows()
