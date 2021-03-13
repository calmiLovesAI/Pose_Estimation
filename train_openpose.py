import tensorflow as tf


from openpose.core.model import CPM
from configuration import OpenPoseCfg as cfg
from openpose.data.parse_tfrecord import get_dataset


if __name__ == '__main__':
    # 获取数据集
    dataset = get_dataset()
    print(type(dataset))

    model = CPM()
    x = tf.random.normal(shape=[cfg.batch_size, *cfg.input_size])
    print("x 的形状：", x.shape)
    y = model(x, training=True)
    for _ in y:
        print(_.shape)
        # (2, 46, 46, 34)
        # (2, 46, 46, 34)
        # (2, 46, 46, 34)
        # (2, 46, 46, 34)
        # (2, 46, 46, 18)
        # (2, 46, 46, 18)
