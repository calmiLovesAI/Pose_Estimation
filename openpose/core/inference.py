import tensorflow as tf

from configuration import OpenPoseCfg as cfg
from openpose.core.post_processing import Skeletonizer


def read_image(image_dir, h=cfg.input_size[0], w = cfg.input_size[1], c=cfg.input_size[2]):
    image = tf.io.read_file(filename=image_dir)
    image = tf.io.decode_image(contents=image, channels=c, dtype=tf.float32)
    image = tf.image.resize(image, [h, w])
    image = tf.expand_dims(image, axis=0)
    return image


class Inference:
    def __init__(self, image, model):
        self.image = image
        self.model = model

    def predict(self):
        pafs, kpts = self.model(self.image, training=False)
        pafs = pafs[0]    # (46, 46, 34)
        kpts = kpts[0]    # (46, 46, 18)
        return Skeletonizer(kpts, pafs).create_skeletons()