import cv2
import tensorflow as tf
import glob

from openpose.core.inference import Inference
from draw.visualization import SkeletonDrawer
from configuration import OpenPoseCfg as cfg


def read_image(image_dir, h=cfg.input_size[0], w = cfg.input_size[1], c=cfg.input_size[2]):
    image = tf.io.read_file(filename=image_dir)
    image = tf.io.decode_image(contents=image, channels=c, dtype=tf.float32)
    image = tf.image.resize(image, [h, w])
    image = tf.expand_dims(image, axis=0)
    return image


class TestImage:
    def __init__(self, model):
        self.model = model

    def process_single_image(self, filename):
        # 读取图片
        image_tensor = read_image(image_dir=filename)
        # 送入网络，得到预测输出
        skeletons = Inference(image_tensor, self.model).predict()
        # 将结果显示在原图片上
        image = cv2.imread(filename)
        skeleton_drawer = SkeletonDrawer(image)
        for skeleton in skeletons:
            skeleton.draw_skeleton(skeleton_drawer.joint_draw, skeleton_drawer.kpt_draw)

        return image

    def test_images(self, filenames):
        for pic_name in glob.glob(filenames + "*.jpg"):
            drawn_image = self.process_single_image(pic_name)
            output_filename = pic_name.split(".")[0] + "-result.jpg"
            cv2.imwrite(output_filename, drawn_image)