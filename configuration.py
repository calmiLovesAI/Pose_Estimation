


class OpenPoseCfg:

    # 输入图片大小 : (H, W, C)
    input_size = (368, 368, 3)

    # 训练超参数
    batch_size = 2
    epochs = 100

    # COCO数据集
    coco_root = "./datasets/COCO/2017/"
    coco_train_images = coco_root + "train2017/"
    coco_train_labels = coco_root + "annotations/"


    # 网络结构参数
    batch_norm_on = False
    dropout_rate = 0
    paf_num_filters = 17 * 2
    heatmap_num_filters = 18