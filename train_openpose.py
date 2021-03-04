import tensorflow as tf


from openpose.core.model import CPM

if __name__ == '__main__':
    print(tf.__version__)
    model = CPM()
    x = tf.random.normal(shape=[2, 368, 368, 3])
    y = model(x, training=True)
    for _ in y:
        print(_.shape)   # (2, 46, 46, 38), (2, 46, 46, 19), ...
