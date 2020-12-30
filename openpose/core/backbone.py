import tensorflow as tf


class VGG(tf.keras.layers.Layer):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same",
                                            activation=tf.keras.activations.relu)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same",
                                            activation=tf.keras.activations.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same",
                                            activation=tf.keras.activations.relu)
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same",
                                            activation=tf.keras.activations.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same",
                                            activation=tf.keras.activations.relu)
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same",
                                            activation=tf.keras.activations.relu)
        self.conv7 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same",
                                            activation=tf.keras.activations.relu)
        self.conv8 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same",
                                            activation=tf.keras.activations.relu)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.conv9 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same",
                                            activation=tf.keras.activations.relu)
        self.conv10 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same",
                                            activation=tf.keras.activations.relu)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool3(x)

        x = self.conv9(x)
        x = self.conv10(x)

        return x



def get_backbone():
    return VGG()