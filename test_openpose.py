import glob

import cv2
import tensorflow as tf

from openpose.core.model import CPM
from openpose.core.inference import read_image, Inference
from draw.visualization import SkeletonDrawer

from configuration import get_cfg



def test_single_picture(filename):
    # 读取图片
    image_tensor = read_image(image_dir=filename)



    # 送入网络，得到预测输出
    skeletons = Inference(image_tensor, model).predict()


    # 将结果显示在原图片上
    image = cv2.imread(filename)
    skeleton_drawer = SkeletonDrawer(image)
    for skeleton in skeletons:
        skeleton.draw_skeleton(skeleton_drawer.joint_draw, skeleton_drawer.kpt_draw)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_filename = filename.split(".")[0] + "-result.jpg"
    cv2.imwrite(output_filename, image)


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    cfg = get_cfg()

    model = CPM(mode=1)


    for pic_name in glob.glob(cfg.test_image_dir + "*.jpg"):
        print("正在处理图片{}......".format(pic_name))
        test_single_picture(pic_name)


