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
from collections import deque
import Glassess
import face_Detect

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def mean(self):
        return sum(self.items) / len(self.items)


cam1 = VideoStreamWidget("rtsp://10.100.43.15:554/stander/livestream/0/0")
# cam1 = VideoStreamWidget("C:\\Users\\green\\PycharmProjects\\ProHack_Agro\\videos\\glass_add\\cam2.avi")
cam2 = VideoStreamWidget("rtsp://10.100.43.16:554/stander/livestream/0/0")
weights_name = 'weights_DNN_1.hdf5'
keras_model = keras.models.load_model(weights_name)


def case_web(model):
    detect_H = False
    detect_G = True
    detect_C = True

    count = 5
    # print("GLASSES")


    last_H = Queue()
    last_C = Queue()

    # print("In cycle, stream.")
    # while True:
    start_time = time.time()
    try:
        # good_img, seg_person = segm_person(model, cam1, detect_H, detect_G, detect_C)
        Glassess.run(face_Detect.run(cam1.read()[1]))
    except:
        return None
    if seg_person is None:
        return None
        # if seg_person(model, cam2.get_image()):
        #     detect_H = False
        #     detect_G = False
        #     detect_C = False

    buf_G = detect.detect_glasses_t(seg_person)
    if buf_G:
        print("GLASSES")

    buf_H = detect.detect_helmets(seg_person)
    last_H.enqueue(int(buf_H))
    if last_H.size() > count:
        last_H.dequeue()

    if last_H.mean() >= 0.5:
        detect_H = True
    else:
        detect_H = False
    buf_C = detect.detect_costume(seg_person)
    # buf_C = detect.detect_cnn_costume(seg_person, keras_model)
    last_C.enqueue(int(buf_C))
    if last_C.size() > count:
        last_C.dequeue()
    if last_C.mean() > 0.3:
        detect_C = True
    else:
        detect_C = False
    print("C", last_C.items, last_C.mean())
    print("H", last_H.items, last_H.mean())
    return good_img
    # segm_person(model, cam2, "res_img2")
    # end_time = time.time()
    # print(last_H, float(sum(last_H) / len(last_H)))
    # print(last_C, float(sum(last_C) / len(last_C)))
    # print("Inference time: ", end_time - start_time)

    # for i in range(2):
    #     cam1.cap.grab()


def case_spy_cameras(model):
    detect_H = False
    detect_G = True
    detect_C = True

    count = 5

    last_H = Queue()
    last_C = Queue()

    cam1 = VideoStreamWidget("rtsp://10.100.43.15:554/stander/livestream/0/0")
    cam2 = VideoStreamWidget("rtsp://10.100.43.16:554/stander/livestream/0/0")

    weights_name = 'weights_DNN_1.hdf5'
    keras_model = keras.models.load_model(weights_name)
    print("In cycle, stream.")
    while True:
        start_time = time.time()
        try:
            good_img, seg_person = segm_person(model, cam1, detect_H, detect_G, detect_C)
        except:
            continue
        if seg_person is None:
            continue
            # if seg_person(model, cam2.get_image()):
            #     detect_H = False
            #     detect_G = False
            #     detect_C = False

        buf_G = detect.detect_glasses_t(seg_person)
        if buf_G:
            print("GLASSES")

        buf_H = detect.detect_helmets(seg_person)
        last_H.enqueue(int(buf_H))
        if last_H.size() > count:
            last_H.dequeue()

        if last_H.mean() >= 0.5:
            detect_H = True
        else:
            detect_H = False
        buf_C = detect.detect_costume(seg_person)
        # buf_C = detect.detect_cnn_costume(seg_person, keras_model)
        last_C.enqueue(int(buf_C))
        if last_C.size() > count:
            last_C.dequeue()
        if last_C.mean() > 0.3:
            detect_C = True
        else:
            detect_C = False
        print("C", last_C.items, last_C.mean())
        print("H", last_H.items, last_H.mean())
        image.show(good_img)
        # segm_person(model, cam2, "res_img2")
        end_time = time.time()
        # print(last_H, float(sum(last_H) / len(last_H)))
        # print(last_C, float(sum(last_C) / len(last_C)))
        print("Inference time: ", end_time - start_time)

        # for i in range(2):
        #     cam1.cap.grab()


path_to_video = os.path.join("videos", "task", "Desktop1.mp4")  # helmet
cam1 = cv2.VideoCapture(path_to_video)


def case_video(model):
    if not os.path.exists(path_to_video):
        print("Wrong path: ", path_to_video)
        return
    case_web(model)


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
    detect.detect_glasses_t(seg_person)


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
        print(dir)
        for file in os.listdir(os.path.join("man_dataset", dir)):
            img_path = os.path.join("man_dataset", dir, file)
            img = image.load_img(img_path)

            img = img[img.shape[0]//7:img.shape[0]]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, img)
            # image.show(img)
