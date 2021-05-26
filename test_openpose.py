import tensorflow as tf

from openpose.core.model import CPM
from configuration import OpenPoseCfg as cfg
from openpose.utils.test_image import TestImage


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = CPM()
    model.load_weights(filepath=cfg.save_model_dir + "the_last_epoch")

    TestImage(model).test_images(filenames=cfg.test_image_dir)

