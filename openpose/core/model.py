import tensorflow as tf

from openpose.core.backbone import VGG


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation=True):
        super(Conv2D, self).__init__()
        self.activation = activation
        self.c = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same")

    def call(self, inputs, **kwargs):
        x = self.c(inputs)
        if self.activation:
            x = tf.nn.relu(x)
        return x


class Stage1(tf.keras.layers.Layer):
    def __init__(self):
        super(Stage1, self).__init__()
        self.s_conv1 = Conv2D(filters=128, kernel_size=3)
        self.s_conv2 = Conv2D(filters=128, kernel_size=3)
        self.s_conv3 = Conv2D(filters=128, kernel_size=3)
        self.s_conv4 = Conv2D(filters=512, kernel_size=1)
        self.s_conv5 = Conv2D(filters=38, kernel_size=1, activation=False)
        self.l_conv1 = Conv2D(filters=128, kernel_size=3)
        self.l_conv2 = Conv2D(filters=128, kernel_size=3)
        self.l_conv3 = Conv2D(filters=128, kernel_size=3)
        self.l_conv4 = Conv2D(filters=512, kernel_size=1)
        self.l_conv5 = Conv2D(filters=19, kernel_size=1, activation=False)

    def call(self, inputs, training=None, **kwargs):
        x = self.s_conv1(inputs)
        x = self.s_conv2(x)
        x = self.s_conv3(x)
        x = self.s_conv4(x)
        s_out = self.s_conv5(x)
        x = self.l_conv1(inputs)
        x = self.l_conv2(x)
        x = self.l_conv3(x)
        x = self.l_conv4(x)
        l_out = self.l_conv5(x)
        outputs = tf.concat(values=[s_out, l_out, inputs], axis=-1)
        return s_out, l_out, outputs


class StageT(tf.keras.layers.Layer):
    def __init__(self):
        super(StageT, self).__init__()
        self.s_conv_1 = Conv2D(filters=128, kernel_size=7)
        self.s_conv_2 = Conv2D(filters=128, kernel_size=7)
        self.s_conv_3 = Conv2D(filters=128, kernel_size=7)
        self.s_conv_4 = Conv2D(filters=128, kernel_size=7)
        self.s_conv_5 = Conv2D(filters=128, kernel_size=7)
        self.s_conv_6 = Conv2D(filters=128, kernel_size=1)
        self.s_conv_7 = Conv2D(filters=38, kernel_size=1, activation=False)

        self.l_conv_1 = Conv2D(filters=128, kernel_size=7)
        self.l_conv_2 = Conv2D(filters=128, kernel_size=7)
        self.l_conv_3 = Conv2D(filters=128, kernel_size=7)
        self.l_conv_4 = Conv2D(filters=128, kernel_size=7)
        self.l_conv_5 = Conv2D(filters=128, kernel_size=7)
        self.l_conv_6 = Conv2D(filters=128, kernel_size=1)
        self.l_conv_7 = Conv2D(filters=19, kernel_size=1, activation=False)

    def call(self, inputs, training=None, **kwargs):
        x = self.s_conv_1(inputs)
        x = self.s_conv_2(x)
        x = self.s_conv_3(x)
        x = self.s_conv_4(x)
        x = self.s_conv_5(x)
        x = self.s_conv_6(x)
        s_out = self.s_conv_7(x)
        x = self.l_conv_1(inputs)
        x = self.l_conv_2(x)
        x = self.l_conv_3(x)
        x = self.l_conv_4(x)
        x = self.l_conv_5(x)
        x = self.l_conv_6(x)
        l_out = self.l_conv_7(x)
        outputs = tf.concat(values=[s_out, l_out, inputs], axis=-1)
        return s_out, l_out, outputs


class CPM(tf.keras.Model):
    def __init__(self):
        super(CPM, self).__init__()
        self.backbone = VGG()
        self.transfer_conv1 = Conv2D(filters=256, kernel_size=3)
        self.transfer_conv2 = Conv2D(filters=128, kernel_size=3)

        self.stage_1 = Stage1()
        self.stage_2 = StageT()
        self.stage_3 = StageT()
        self.stage_4 = StageT()
        self.stage_5 = StageT()
        self.stage_6 = StageT()

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs, training=training)
        x = self.transfer_conv1(x)
        x = self.transfer_conv2(x)

        s_1, l_1, x = self.stage_1(x)
        s_2, l_2, x = self.stage_2(x)
        s_3, l_3, x = self.stage_3(x)
        s_4, l_4, x = self.stage_4(x)
        s_5, l_5, x = self.stage_5(x)
        s_6, l_6, _ = self.stage_6(x)

        return s_1, l_1, s_2, l_2, s_3, l_3, s_4, l_4, s_5, l_5, s_6, l_6