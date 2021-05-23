import tensorflow as tf
import numpy as np
import cv2

from pycocotools.coco import COCO
from configuration import OpenPoseCfg
from openpose.data.tfrecord_utils import FileSharder, encode_example



def reshape_kpts(keypoints):
    kpts_np = np.array(keypoints, dtype=np.float32)
    kpts_np = np.reshape(kpts_np, (OpenPoseCfg.dataset_num_keypoints, 3))
    return kpts_np


def map_new_kpts(keypoints):
    new_kpts = []
    for kpt_name, kpt_def in OpenPoseCfg.KEYPOINTS_DEF.items():
        ds_idxs = kpt_def["ds_idxs"]
        assert type(ds_idxs) is int or (type(ds_idxs) is tuple and len(ds_idxs) == 2)

        if type(ds_idxs) is tuple:
            first_kpt = keypoints[ds_idxs[0]]
            second_kpt = keypoints[ds_idxs[1]]
            new_kpt = np.array(middle_kpt(first_kpt, second_kpt), dtype=np.float32)
        else:
            new_kpt = keypoints[ds_idxs]
        new_kpts.append(new_kpt)
    return new_kpts


def middle_kpt(kpt1, kpt2):
    if kpt1[2] == 0 or kpt2[2] == 0:
        return [0, 0, 0]
    else:
        return [
                (kpt1[0] + kpt2[0]) / 2,
                (kpt1[1] + kpt2[1]) / 2,
                min(kpt1[2], kpt2[2])
                ]


def transform_keypts(keypoints, size):
    X = np.array(keypoints[..., 0], dtype=np.float32)
    Y = np.array(keypoints[..., 1], dtype=np.float32)
    keypoints[..., 0] = Y
    keypoints[..., 1] = X

    keypoints[:, :, 0:2] = keypoints[:, :, 0:2] / size
    return keypoints


def create_all_joints(all_keypts):
    """create a joints tensor from keypoints tensor, according to COCO joints
    :param all_keypts - tensor of shape (number of persons,number of kpts(DS_NUM_KEYPOINTS),3)
    :return tensor of shape (number of persons,number of joints(19),5)"""

    def create_joints(keypts):
        joints = []
        for joint_name, joint_def in OpenPoseCfg.JOINTS_DEF.items():
            kp1_name, kp2_name = joint_def["kpts"]
            kp1_idx = OpenPoseCfg.KEYPOINTS_DEF[kp1_name]["idx"]
            kp2_idx = OpenPoseCfg.KEYPOINTS_DEF[kp2_name]["idx"]
            kp1 = keypts[kp1_idx]
            kp2 = keypts[kp2_idx]
            if kp1[2] == 0 or kp2[2] == 0:
                # if either of the keypoints is missing, the joint is zero
                new_joint = (0, 0, 0, 0, 0)
                joints.append(new_joint)
                continue
            # create new joint from both keypoint coords, with the visibility being the minimum of either keypoint
            new_joint = (*kp1[0:2], *kp2[0:2], min(kp1[2], kp2[2]))
            joints.append(new_joint)
        return np.array(joints, dtype=np.float32)

    all_joints = [create_joints(x) for x in all_keypts]  # for each person

    # numpify result transpose joints
    return np.array(all_joints, dtype=np.float32).transpose((1, 0, 2))


if __name__ == '__main__':
    annotation_file = OpenPoseCfg.train_anns
    print("开始处理："+annotation_file)
    coco = COCO(annotation_file)
    category = 1
    imgIds = coco.getImgIds(catIds=[category])
    imgIds.sort()

    print("选取的图片数量：", len(imgIds))

    filename_format = OpenPoseCfg.train_tfrecords + "-{:03}.tfrecords"
    with FileSharder(tf.io.TFRecordWriter, filename_format, OpenPoseCfg.images_per_tfrecord) as writer:
        for img_id in imgIds:
            image_info = coco.loadImgs(ids=img_id)[0]
            size = [image_info['height'], image_info['width']]

            annIds = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ids=annIds)

            person_keypoints = []
            for annotation in anns:
                if annotation['num_keypoints'] > 0:
                    kpts = annotation['keypoints']
                    kpts = reshape_kpts(kpts)
                    kpts = map_new_kpts(kpts)

                    person_keypoints.append(kpts)
            if not person_keypoints:
                continue

            person_keypoints = np.array(person_keypoints, dtype=np.float32)
            keypoints = transform_keypts(person_keypoints, np.array(size, dtype=np.int))   # shape: (N, 18, 3(y, x, visibility))
            tr_joint = create_all_joints(keypoints)   # shape: (17, N, 5(y1, x1, y2, x2, visibility))
            tr_keypoints = keypoints.transpose((1, 0, 2))     # shape: (18, N, 3)


            total_mask = np.zeros(size, dtype=np.float32)

            for annotation in anns:
                if annotation['num_keypoints'] == 0:
                    single_mask = coco.annToMask(annotation)
                    total_mask = np.max([single_mask, total_mask], axis=0)

            total_mask = cv2.resize(total_mask, dsize=OpenPoseCfg.model_output_size)
            total_mask = (total_mask > 0.01).astype(np.int16)

            kernel = np.ones((5, 5), np.uint8)
            total_mask = cv2.dilate(total_mask, kernel)
            total_mask = total_mask.astype(np.bool)
            total_mask = np.invert(total_mask)
            total_mask = total_mask.astype(np.float32)

            try:
                img_path = OpenPoseCfg.coco_images + "/" + image_info['file_name']
                image_raw = tf.io.read_file(img_path)
            except:
                print("无法读取文件：%s" % img_path)
                continue
            example = encode_example(img_id, image_raw, size, tr_keypoints, tr_joint, total_mask)
            writer.write(example)
