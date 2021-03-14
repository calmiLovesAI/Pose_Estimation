import tensorflow as tf


from configuration import OpenPoseCfg


class Transformer:
    def __init__(self):
        self.label_height = OpenPoseCfg.model_output_size[0]
        self.label_width = OpenPoseCfg.model_output_size[1]
        self.image_size = OpenPoseCfg.input_size[:2]
        self.kpt_heamap_gaussian_sigma_sq = OpenPoseCfg.KPT_HEATMAP_GAUSSIAN_SIGMA_SQ
        self.paf_gaussian_sigma_sq = OpenPoseCfg.PAF_GAUSSIAN_SIGMA_SQ
        self.paf_num_filters = OpenPoseCfg.paf_num_filters
        self.heatmap_num_filters = OpenPoseCfg.heatmap_num_filters
        self.joints_def = OpenPoseCfg.JOINTS_DEF
        self.joints_sides = OpenPoseCfg.JOINTS_SIDES
        self.keypoints_sides = OpenPoseCfg.KEYPOINTS_SIDES

        self.contrast_range = OpenPoseCfg.contrast_range
        self.saturation_range = OpenPoseCfg.saturation_range
        self.hue_range = OpenPoseCfg.hue_range
        self.brightness_range = OpenPoseCfg.brightness_range

        self.feature_description = {
            'id'       : tf.io.FixedLenFeature([1], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'size'     : tf.io.FixedLenFeature([2], tf.int64),
            'kpts'     : tf.io.FixedLenFeature([], tf.string),
            'joints'   : tf.io.FixedLenFeature([], tf.string),
            'mask'     : tf.io.FixedLenFeature([], tf.string)
        }

        self.init_grid()

    def init_grid(self):
        y_grid = tf.linspace(0.0, 1.0, self.label_height)
        x_grid = tf.linspace(0.0, 1.0, self.label_width)
        y, x = tf.meshgrid(y_grid, x_grid, indexing='ij')
        self.grid = tf.stack(values=[y, x], axis=-1)

    def read_tfrecord(self, serialized_example):
        parsed = tf.io.parse_single_example(serialized=serialized_example, features=self.feature_description)

        image_id = parsed['id']
        image_raw = parsed['image_raw']
        size = parsed['size']

        kpts = tf.io.parse_tensor(parsed['kpts'], tf.float32)
        joints = tf.io.parse_tensor(parsed['joints'], tf.float32)
        mask = tf.io.parse_tensor(parsed['mask'], tf.float32)
        mask = tf.ensure_shape(mask, ([self.label_height, self.label_width]))
        mask = tf.expand_dims(mask, axis=-1)

        kpts = tf.RaggedTensor.from_tensor(kpts)
        joints = tf.RaggedTensor.from_tensor(joints)

        return {"id": image_id, "image_raw": image_raw, "size": size, "kpts": kpts, "joints": joints, "mask": mask}

    def read_image(self, e):
        image_raw = e['image_raw']
        image_tensor = tf.io.decode_jpeg(image_raw, channels=3)
        image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
        image_tensor = tf.image.resize(image_tensor, size=self.image_size)

        new_e = {}
        new_e.update(e)
        new_e.pop('image_raw')
        new_e['image'] = image_tensor
        return new_e

    def keypoints_spots_vmapfn(self, kpts_tensor):
        """This transforms the keypoint coords coming from the dataset into gaussian spots label tensor
        this version of the function works via a map_fn.
        *does not support batched input
        :param kpts_tensor - must be a tf.RaggedTensor of shape (num of keypoints(17 for coco),n,3) where n is the number of persons
        :return tf.Tensor of shape (num of keypoints(17 for coco),IMAGE_HEIGHT,IMAGE_WIDTH)"""
        kpts_tensor = kpts_tensor.to_tensor()  # seems to be mandatory for map_fn
        all_dists = tf.map_fn(self.keypoints_layer,
                              kpts_tensor)  # ,parallel_iterations=20) for cpu it has no difference, maybe for gpu it will

        raw = tf.exp((-(all_dists ** 2) / self.kpt_heamap_gaussian_sigma_sq))
        result = tf.where(raw < 0.001, 0.0, raw)

        result = tf.transpose(result, (1, 2, 0))  # must transpose to match the model output
        result = tf.ensure_shape(result, ([self.label_height, self.label_width, self.heatmap_num_filters]),
                                 name="kpts_ensured_shape")
        return result

    def keypoints_layer(self, kpts_layer):
        """This transforms a single layer of keypoints (such as 3 keypoints of type 'right shoulder')
        the keypoint_distance creates an array of the distances from each keypoint
        and this reduces them to a single array by the  of the distances.
        :param kpts_layer must be a tf.Tensor of shape (n,3)"""
        layer_dists = tf.map_fn(self.keypoint_distance, kpts_layer)
        return tf.math.reduce_min(layer_dists, axis=0)

    def keypoint_distance(self, kpt):
        """This transforms a single keypoint into an array of the distances from the keypoint
        :param kpt must be tf.Tensor of shape (x,y,a) where a is either 0,1,2 for missing,invisible and visible"""
        if kpt[2] == tf.constant(0.0):
            return tf.ones((self.label_height, self.label_width),
                           dtype=tf.float32)  # maximum distance in case of empty kpt, not ideal but meh
        else:
            ortho_dist = self.grid - kpt[0:2]
            return tf.linalg.norm(ortho_dist, axis=-1)

    def joints_PAFs(self, joints_tensor):
        """This transforms the joints coords coming from the dataset into vector fields label tensor
        *does not support batched input
        :param joints_tensor - must be a tf.RaggedTensor of shape (num of joints(19 for coco),number of persons,3)
        :return tf.Tensor of shape (IMAGE_HEIGHT,IMAGE_WIDTH,num of joints(19 for coco)*2)"""
        joints_tensor = joints_tensor.to_tensor()  # seems to be mandatory for map_fn
        all_pafs = tf.map_fn(self.layer_PAF,
                             joints_tensor)
        # ,parallel_iterations=20) for cpu it has no difference, maybe for gpu it will
        # this must be executed in the packing order, to produce the layers in the right order
        result = tf.stack(all_pafs)

        result = tf.where(abs(result) < 0.001, 0.0, result)  # stabilize numerically

        # must transpose to fit the label (NJOINTS,LABEL_HEIGHT, LABEL_WIDTH, 2) to
        # [LABEL_HEIGHT, LABEL_WIDTH,PAF_NUM_FILTERS=NJOINTS*2]
        result = tf.transpose(result, [1, 2, 0, 3])
        result_y = result[..., 0]
        result_x = result[..., 1]
        result = tf.concat((result_y, result_x), axis=-1)

        result = tf.ensure_shape(result, ([self.label_height, self.label_width, self.paf_num_filters]),
                                 name="paf_ensured_shape")
        return result

    def layer_PAF(self, joints):
        """ Makes a combined PAF for all joints of the same type
        and reduces them to a single array by averaging the vectors out
        *does not support batched input
        :param joints must be a tf.Tensor of shape (n,5)
        :return a tensor of shape (LABEL_HEIGHT, LABEL_WIDTH, 2)"""
        layer_PAFS = tf.map_fn(self.single_PAF, joints)
        combined = tf.math.reduce_sum(layer_PAFS,
                                      axis=0)  # averages the vectors out to combine the fields in case they intersect
        return combined

    def single_PAF(self, joint):
        """ Makes a single vector valued PAF (part affinity field) array
        *does not support batched input
        :param joint a 1D tensor of (x1,y1,x2,y2,visibility)
        :return a tensor of shape (LABEL_HEIGHT, LABEL_WIDTH, 2)
        """
        jpts = tf.reshape(joint[0:4], (2, 2))  # reshape to ((x1,y1),(x2,y2))
        if joint[4] == tf.constant(0.0) or tf.reduce_all(jpts[1] - jpts[0] == 0.0):
            return tf.zeros((self.label_height, self.label_width, 2), dtype=tf.float32)  # in case of empty joint
        else:
            # this follows the OpenPose paper of generating the PAFs
            vector_full = jpts[1] - jpts[0]  # get the joint vector
            vector_length = tf.linalg.norm(vector_full)  # get joint length
            vector_hat = vector_full / vector_length  # get joint unit vector
            normal_vector = tf.stack((-vector_hat[1], vector_hat[0]))

            vectors_from_begin = self.grid - jpts[0]  # get grid of vectors from first joint point
            vectors_from_end = self.grid - jpts[1]  # get grid of vectors from second joint point

            projections = tf.tensordot(vectors_from_begin, vector_hat, 1)  # get projection on the joint unit vector
            n_projections = tf.tensordot(vectors_from_begin, normal_vector,
                                         1)  # get projection on the joint normal unit vector

            dist_from_begin = tf.linalg.norm(vectors_from_begin, axis=-1)  # get distances from the beginning, and end
            dist_from_end = tf.linalg.norm(vectors_from_end, axis=-1)

            begin_gaussian_mag = tf.exp(
                (-(dist_from_begin ** 2) / self.paf_gaussian_sigma_sq))  # compute gaussian bells
            end_gaussian_mag = tf.exp((-(dist_from_end ** 2) / self.paf_gaussian_sigma_sq))
            normal_gaussian_mag = tf.exp((-(n_projections ** 2) / self.paf_gaussian_sigma_sq))

            limit = (0 <= projections) & (
                        projections <= vector_length)  # cutoff the joint before beginning and after end
            limit = tf.cast(limit, tf.float32)
            bounded_normal_gaussian_mag = normal_gaussian_mag * limit  # bound the normal distance by the endpoints

            max_magnitude = tf.math.reduce_max((begin_gaussian_mag, end_gaussian_mag, bounded_normal_gaussian_mag),
                                               axis=0)

            vector_mag = tf.stack((max_magnitude, max_magnitude), axis=-1)

            result = vector_mag * vector_hat  # broadcast joint direction vector to magnitude field
            return result

    def convert_label_to_tensors(self, e):
        kpts = self.keypoints_spots_vmapfn(e['kpts'])
        pafs = self.joints_PAFs(e['joints'])

        new_e = {}
        new_e.update(e)
        new_e.pop('joints')

        new_e["pafs"] = pafs
        new_e["kpts"] = kpts

        return new_e

    def image_aug(self, e):
        image = e['image']
        image = tf.image.random_contrast(image=image, lower=1-self.contrast_range, upper=1+self.contrast_range)
        image = tf.image.random_saturation(image=image, lower=1-self.saturation_range, upper=1+self.saturation_range)
        image = tf.image.random_brightness(image=image, max_delta=self.brightness_range)
        image = tf.image.random_hue(image=image, max_delta=self.hue_range)
        image = tf.clip_by_value(t=image, clip_value_min=0.0, clip_value_max=1.0)

        new_e = {}
        new_e.update(e)
        new_e["image"] = image
        return new_e

    def apply_mask(self, e):
        """Transforms a dict data element:
        applies background persons mask to keypoints and PAF tensors as last channel,
        to be used by the special masked loss"""

        mask = e['mask']
        pafs = e["pafs"]
        kpts = e["kpts"]

        print("kpts: {}, pafs: {}, mask: {}".format(kpts.shape, pafs.shape, mask.shape))

        kpts = tf.concat([kpts, mask], axis=-1)  # add mask as zero channel to inputs
        pafs = tf.concat([pafs, mask], axis=-1)

        new_e = {}
        new_e.update(e)
        new_e["pafs"] = pafs
        new_e["kpts"] = kpts

        return new_e