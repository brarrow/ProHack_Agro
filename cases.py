import detect
import visualization
from detect import segm_person
from image import cut_holst_from_bin_roi
import image
import os
from spy_camera import VideoStreamWidget
import cv2
import time
import uuid
import keras

detect_H = True


def case_spy_cameras(model):
    cam1 = VideoStreamWidget("rtsp://10.100.43.15:554/stander/livestream/0/0")
    cam2 = VideoStreamWidget("rtsp://10.100.43.16:554/stander/livestream/0/0")
    weights_name = 'weights_DNN_1.hdf5'
    keras_model = keras.models.load_model(weights_name)
    print("In cycle, stream.")
    while True:
        # cam1 = OpenCVCam("rtsp://10.100.43.15:554/stander/livestream/0/0")
        start_time = time.time()
        # print("Here1")
        seg_person = segm_person(model, cam1, "res_img1")
        # print("Here2")
        if seg_person is None:
            continue
        detect.detect_helmets(seg_person)
        # detect.detect_costume(seg_person)
        detect.detect_cnn_costume(seg_person, keras_model)
        image.show(seg_person)
        # segm_person(model, cam2, "res_img2")
        end_time = time.time()
        print("Inference time: ", end_time - start_time)
        # for i in range(2):
        #     cam1.cap.grab()


def case_video(model):
    path_to_video = os.path.join("videos", "no_glass", "cam1.avi") # helmet
    # path_to_video = os.path.join("videos", "igor", "cam1.avi")  # no helmet
    if not os.path.exists(path_to_video):
        print("Wrong path: ", path_to_video)
        return
    vid = cv2.VideoCapture(path_to_video)
    while vid.isOpened():
        seg_person = segm_person(model, vid)
        if seg_person is None:
            continue
        detect.detect_helmets(seg_person)
        detect.detect_costume(seg_person)
        image.show(seg_person)


def case_image(model, img_path):
    original_im = image.load_img(img_path)
    resized_im, seg_map = model.run(original_im)
    res_img = detect.detecting(resized_im, seg_map, H=True)
    visualization.vis_segmentation_cv(resized_im, res_img)


def case_test_image(seg_person_path):
    seg_person = image.load_img(seg_person_path)
    seg_person = cv2.cvtColor(seg_person, cv2.COLOR_BGR2RGB)
    # detect.detect_helmets(seg_person)
    # detect.detect_costume(seg_person)
    detect.detect_glasses(seg_person)


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
    while time.time() - start_time < 50:
        frame = cap.get_image()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.show(frame)

        # write the flipped frame
        out.write(frame)

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release everything if job is finished
    # cap.release()
    out.release()
    cv2.destroyAllWindows()


def case_segment_video(model):
    path_to_video = os.path.join("videos", "glass_no")
    for file in os.listdir(path_to_video):
        file_path = os.path.join(path_to_video, file)
        if not os.path.exists(file_path):
            print("Wrong path: ", path_to_video)
            return
        vid = cv2.VideoCapture(file_path)
        dir_to_save = os.path.join("man_dataset", "glass_no")
        if not os.path.exists(dir_to_save):
            os.makedirs(dir_to_save)
        while vid.isOpened():
            seg_person = segm_person(model, vid)
            if seg_person is None:
                continue
            cv2.imwrite(os.path.join(dir_to_save, "{}.jpg".format(uuid.uuid4())), seg_person)
            image.show(seg_person)


def case_cut_data():
    for dir in os.listdir("man_dataset"):
        for file in os.listdir(os.path.join("man_dataset", dir)):
            img_path = os.path.join("man_dataset", dir, file)
            img = image.load_img(img_path)

            # img = img[img.shape[0]//5:img.shape[0]]
            cv2.imwrite(img_path, img)
            # image.show(img)
