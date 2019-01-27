# -*- coding:utf-8 -*-
"""
Author: tsing
Time: 2019-01-12 24:00
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf 
slim = tf.contrib.slim
import numpy as np
import os 
import cv2
import sys 
sys.path.append("../")
from tqdm import tqdm
from config import cfgs 
from models.network.model_net3 import Detect_Model 
from libs.encode_decode import encode_decode 
from libs.preprocessing import ssd_vgg_preprocessing, preprocessing
from libs.tools import tools, drawbox, img

image_shape = [cfgs.train_image_shape['height'], cfgs.train_image_shape['width']]
class LoadModel_Predict():
    def __init__(self, checkpoint_path, checkpoint_step=None):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_step = checkpoint_step
        self.restore_model()

    @staticmethod
    def get_latest_ckpt(path):
        """get least checkpoint path
        """
        if path.startswith('~'):
            path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if os.path.isdir(path):
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt is not None:
                ckpt_path = ckpt.model_checkpoint_path
            else:
                ckpt_path = None
        else:
            ckpt_path = path
        return ckpt_path

    def restore_model(self):
        """restore model
        """
        tf.logging.set_verbosity(tf.logging.INFO)
        # step 1: Define the calculation process ===================================
        with tf.name_scope('Evaluate'):
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                global_step = tf.train.get_or_create_global_step()
                self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
                processed_image,_,_,_,_ = ssd_vgg_preprocessing.preprocess_image(self.image_placeholder, None, None, None, None,
                                                                                out_shape=image_shape,
                                                                                data_format='NHWC',
                                                                                is_training=False)
                b_image = tf.expand_dims(processed_image, axis=0)
                # NOTE is_training must be True
                net = Detect_Model(b_image, batch_size=1, num_classes=len(cfgs.label_num_map), is_training=True)
                self.pred = tf.expand_dims(tf.argmax(net.logits, axis=3, output_type=tf.int32), axis=3)

        
        # step 2: Restore session and graph ========================================
        sess_config = tf.ConfigProto(log_device_placement=False,
                                     allow_soft_placement=True)
        if cfgs.gpu_memory_fraction < 0:
            sess_config.gpu_options.allow_growth = True
        elif cfgs.gpu_memory_fraction > 0:
            sess_config.gpu_options.per_process_gpu_memory_fraction
        if cfgs.using_moving_average:
            # 创建一个指数移动平均类 variable_averages
            variable_averages = tf.train.ExponentialMovingAverage(cfgs.moving_average_decay)
            # 将variable_averages作用于当前模型中所有可训练的变量上，得到 variables_averages_op操作符
            variables_to_restore = variable_averages.variables_to_restore(tf.trainable_variables())
            # 建立'op':op的字典
            variables_to_restore[global_step.op.name] = global_step
        else:
            variables_to_restore = slim.get_varibles_to_restore()
        saver = tf.train.Saver(var_list=variables_to_restore)
        self.sess = tf.Session(config=sess_config)
        if not self.checkpoint_step:
            checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint = '{}model.ckpt-{}'.format(self.checkpoint_path, self.checkpoint_step)
        saver.restore(self.sess, self.get_latest_ckpt(checkpoint))
        tf.logging.info('\n'+"@@@"*20+"\nRestore Model >>>\n\t"+"{}\nModel Restored !!!\n".format(os.path.abspath(checkpoint))+"@@@"*20)


    def predict(self, image, convert_to_mask=True):
        """predict function

        Args:
            Ndarray image: whitened img ;channel is BGR
        Returns:
            Tensor mask: multi_class mask
        """
        with tf.device('/gpu:0'):
            cls_map = self.sess.run(self.pred, feed_dict={self.image_placeholder: image})
            return cls_map


if __name__ == "__main__":
    model = LoadModel_Predict(cfgs.checkpoint_path, 98579)
    image_names = os.listdir(cfgs.test_image_dir)
    image_names.sort()
    pbar = tqdm(iterable=image_names, total=len(image_names), ncols=80, leave=True)
    for image_name in pbar:
        image_path = os.path.join(cfgs.test_image_dir, image_name)
        pbar.set_description('Evaluating:{}'.format(image_path))
        image = cv2.imread(image_path)
        if not isinstance(image, np.ndarray):
            continue
        cls_map, multi_mask = model.predict(image, convert_to_mask=True)
        bboxes = encode_decode.mask_to_bboxes_v2(cls_map[0], image.shape[: 2])
        image_bbox = drawbox.draw_box(image, bboxes)
        demo_saver = tools.make_folder(cfgs.test_result_dir)
        cv2.imwrite(os.path.join(demo_saver, image_name), image_bbox)
        # if isinstance(multi_mask, np.ndarray):
        #     
        #     cv2.imwrite(os.path.join(demo_saver, image_name), multi_mask[0])