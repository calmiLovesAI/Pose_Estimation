import tensorflow as tf


def int64_feature(value):
    if type(value) != list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_example(image_id, image_raw, size, kpts, joints, mask):
    kpts = tf.constant(kpts)
    kpts = tf.io.serialize_tensor(kpts).numpy()
    joints = tf.constant(joints)
    joints = tf.io.serialize_tensor(joints).numpy()
    mask = tf.constant(mask)
    mask = tf.io.serialize_tensor(mask).numpy()

    image_raw = image_raw.numpy()

    feature = {
        'id': int64_feature(image_id),
        'image_raw': bytes_feature(image_raw),
        'size': int64_feature(size),
        'kpts': bytes_feature(kpts),
        'joints': bytes_feature(joints),
        'mask': bytes_feature(mask)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()



class FileSharder:
    def __init__(self, file_writer, base_filename_format: str, records_per_file: int, verbose: bool = True):
        assert base_filename_format.format(0) != base_filename_format

        self._file_writer = file_writer
        self._base_filename_format = base_filename_format
        self._records_per_file = records_per_file
        self._example_counter = 0
        self._file_counter = 1
        self._verbose = verbose
        self._start_file()

    def __enter__(self):
        return self

    def _start_file(self):
        self._filename = self._base_filename_format.format(self._file_counter)
        if self._verbose: print("\nWriting file:" + self._filename, flush=True)
        self._writer = self._file_writer(self._filename)

    def _finish_file(self):
        self._writer.flush()
        self._writer.close()

    def _advance_file(self):
        self._finish_file()
        self._file_counter += 1
        self._example_counter = 0
        self._start_file()

    def write(self, item):
        self._writer.write(item)
        if self._verbose and not self._example_counter % 100: print(".", end="", flush=True)
        self._example_counter += 1
        if not self._example_counter % self._records_per_file:
            self._advance_file()

    def __exit__(self, *args):
        self._finish_file()