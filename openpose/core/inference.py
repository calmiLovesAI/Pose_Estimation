import tensorflow as tf

from configuration import OpenPoseCfg as cfg
from openpose.core.post_processing import Skeletonizer





class Inference:
    def __init__(self, image, model):
        self.image = image
        self.model = model

    def predict(self):
        _, _, _, pafs, _, kpts = self.model(self.image, training=False)
        pafs = pafs[0]    # (46, 46, 34)
        kpts = kpts[0]    # (46, 46, 18)
        return Skeletonizer(kpts, pafs).create_skeletons()