import tensorflow as tf

from openpose.core.backbone import VGG
from configuration import get_cfg


cfg = get_cfg()


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.bns = list()
        self.convs = list()
        self.acts = list()
        for i in range(3):
            if OpenPoseCfg.batch_norm_on:
                self.bns.append(tf.keras.layers.BatchNormalization())
            else:
                self.bns.append(tf.keras.layers.Layer())
            self.convs.append(tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same"))
            self.acts.append(tf.keras.layers.PReLU(shared_axes=[1, 2]))

    def call(self, inputs, training=None, **kwargs):
        x1 = self.bns[0](inputs, training=training)
        x1 = self.acts[0](self.convs[0](x1))

        x2 = self.bns[1](x1, training=training)
        x2 = self.acts[1](self.convs[1](x2))

        x3 = self.bns[2](x2, training=training)
        x3 = self.acts[2](self.convs[2](x3))

        return tf.concat(values=[x1, x2, x3], axis=-1)


class Stage0(tf.keras.layers.Layer):
    def __init__(self):
        super(Stage0, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=1, strides=1, padding="same")
        self.act1 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=1, strides=1, padding="same")
        self.act2 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same")
        self.act3 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same")
        self.act4 = tf.keras.layers.PReLU(shared_axes=[1, 2])

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)

        return x


class StageI(tf.keras.layers.Layer):
    def __init__(self, filters, output_filters, activation_func):
        super(StageI, self).__init__()
        self.drop_1 = tf.keras.layers.Dropout(rate=OpenPoseCfg.dropout_rate)
        self.conv_1 = ConvBlock(filters=filters)

        self.drop_2 = tf.keras.layers.Dropout(rate=OpenPoseCfg.dropout_rate)
        self.conv_2 = ConvBlock(filters=filters)

        self.drop_3 = tf.keras.layers.Dropout(rate=OpenPoseCfg.dropout_rate)
        self.conv_3 = ConvBlock(filters=filters)

        self.drop_3 = tf.keras.layers.Dropout(rate=OpenPoseCfg.dropout_rate)
        self.conv_3 = ConvBlock(filters=filters)

        self.drop_4 = tf.keras.layers.Dropout(rate=OpenPoseCfg.dropout_rate)
        self.conv_4 = ConvBlock(filters=filters)

        self.drop_5 = tf.keras.layers.Dropout(rate=OpenPoseCfg.dropout_rate)
        self.conv_5 = ConvBlock(filters=filters)

        self.conv6 = tf.keras.layers.Conv2D(filters=512, kernel_size=1, strides=1, padding="same")
        self.act = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.conv7 = tf.keras.layers.Conv2D(filters=output_filters, kernel_size=1, strides=1, padding="same", activation=activation_func)


    def call(self, inputs, training=None, **kwargs):
        if len(inputs) > 1:
            x = tf.concat(values=inputs, axis=-1)
        else:
            x = inputs[0]
        x = self.drop_1(x, training=training)
        x = self.conv_1(x)
        x = self.drop_2(x, training=training)
        x = self.conv_2(x)
        x = self.drop_3(x, training=training)
        x = self.conv_3(x)
        x = self.drop_4(x, training=training)
        x = self.conv_4(x)
        x = self.drop_5(x, training=training)
        x = self.conv_5(x)

        x = self.conv6(x)
        x = self.act(x)
        x = self.conv7(x)

        return x


class CPM(tf.keras.Model):
    def __init__(self, mode=0):
        super(CPM, self).__init__()
        self.mode = mode
        self.paf_num_filters = OpenPoseCfg.paf_num_filters
        self.heatmap_num_filters = OpenPoseCfg.heatmap_num_filters

        self.backbone = VGG()
        self.stage_0 = Stage0()
        self.stage_1 = StageI(filters=96, output_filters=self.paf_num_filters,
                              activation_func=tf.keras.activations.linear)
        self.stage_2 = StageI(filters=128, output_filters=self.paf_num_filters,
                              activation_func=tf.keras.activations.linear)
        self.stage_3 = StageI(filters=128, output_filters=self.paf_num_filters,
                              activation_func=tf.keras.activations.linear)
        self.stage_4 = StageI(filters=128, output_filters=self.paf_num_filters,
                              activation_func=tf.keras.activations.linear)
        self.stage_5 = StageI(filters=96, output_filters=self.heatmap_num_filters,
                              activation_func=tf.keras.activations.tanh)
        self.stage_6 = StageI(filters=128, output_filters=self.heatmap_num_filters,
                              activation_func=tf.keras.activations.tanh)

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs, training=training)
        x = self.stage_0(x)

        s1 = self.stage_1([x], training=training)
        s2 = self.stage_2([s1, x], training=training)
        s3 = self.stage_3([s2, x], training=training)
        s4 = self.stage_4([s3, x], training=training)
        s5 = self.stage_5([s4, x], training=training)
        s6 = self.stage_6([s5, s4, x], training=training)

        if self.mode == 0:
            return [s1, s2, s3, s4, s5, s6]
        else:
            return [s4, s6]