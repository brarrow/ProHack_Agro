import os

import numpy as np
import tensorflow as tf
from PIL import Image
import image


class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, model_name):
        self.graph = tf.Graph()

        with open(os.path.join("models", model_name), 'rb') as f:
            graph_def = tf.compat.v1.GraphDef.FromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, img_cv):
        img_pil = image.get_pil_img(img_cv)
        width, height = img_pil.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = img_pil.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return image.get_cv2_img(resized_image), image.get_cv2_img(seg_map)