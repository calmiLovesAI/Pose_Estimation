import tensorflow as tf
import glob

from configuration import OpenPoseCfg
from openpose.data.augmentation import Transformer


def get_tfrecord_filenames(path):
    print("从"+path+"中提取TFRecords文件")
    tfrecord_files = glob.glob(path + "*")
    tfrecord_files.sort()
    if not tfrecord_files:
        raise ValueError("在"+path+"处未找到TFRecords文件")
    return tfrecord_files


class TFRecordDataset:
    def __init__(self, tfrecord_filenames, label_placement_func):
        self.label_place = label_placement_func
        self.tfrecords = tfrecord_filenames
        self.transformer = Transformer()
        self.img_aug = OpenPoseCfg.image_aug_on
        self.batch_size = OpenPoseCfg.batch_size

    def generate(self):
        dataset = tf.data.TFRecordDataset(filenames=self.tfrecords)
        dataset = dataset.map(self.transformer.read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.transformer.read_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.transformer.convert_label_to_tensors, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)

        if self.img_aug:
            dataset = dataset.map(self.transformer.image_aug, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.map(self.transformer.apply_mask, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.label_place, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.repeat()

        return dataset