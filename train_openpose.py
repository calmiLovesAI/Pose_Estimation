import tensorflow as tf


from openpose.core.model import CPM
from openpose.core.loss import OpenPoseLoss
from configuration import OpenPoseCfg as cfg
from openpose.data.parse_tfrecord import get_dataset


if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 获取数据集
    dataset = get_dataset()

    # 获取模型
    model = CPM()
    # x = tf.random.normal(shape=[cfg.batch_size, *cfg.input_size])
    # print("x 的形状：", x.shape)
    # y = model(x, training=True)
    # for _ in y:
    #     print(_.shape)
    #     # (2, 46, 46, 34)
    #     # (2, 46, 46, 34)
    #     # (2, 46, 46, 34)
    #     # (2, 46, 46, 34)
    #     # (2, 46, 46, 18)
    #     # (2, 46, 46, 18)

    start_epoch = 0
    # 加载权重
    if cfg.load_weights_before_training:
        model.load_weights(filepath=cfg.save_model_dir + "epoch-{}".format(cfg.load_weights_from_epoch))
        print("加载epoch-{}权重成功！".format(cfg.load_weights_from_epoch))
        start_epoch = cfg.load_weights_from_epoch + 1

    loss = OpenPoseLoss()
    optimizer = tf.keras.optimizers.Adam()
    loss_metrics = tf.keras.metrics.Mean()

    def train_steps(images, labels):
        with tf.GradientTape() as tape:
            pred = model(images, training=True)
            loss_value = loss(y_true=labels, y_pred=pred)
        gradients = tape.gradient(target=loss_value, sources=model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        loss_metrics.update_state(values=loss_value)


    for epoch in range(start_epoch, cfg.epochs):
        for step, batch_data in enumerate(dataset):
            train_images, train_labels = batch_data[0], batch_data[1]
            train_steps(train_images, train_labels)
            print("Epoch: {}/{}, step: {}, loss: {}".format(epoch,
                                                            cfg.epochs,
                                                            step,
                                                            loss_metrics.result()))
        loss_metrics.reset_states()

        if epoch % cfg.save_frequency == 0:
            model.save_weights(filepath=cfg.save_model_dir + "epoch-{}".format(epoch), overwrite=False, save_format="tf")

    model.save_weights(filepath=cfg.save_model_dir + "the_last_epoch", overwrite=False, save_format="tf")
