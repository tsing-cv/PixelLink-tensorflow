# -*- coding:utf-8 -*-
"""
Author: tsing
Time: 2018-12-27
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
slim = tf.contrib.slim
import sys
sys.path.append("../")
import time, datetime
from dataio.write_and_read_tfrecord import Read_Tfrecord
from config import cfgs
from models.network.model_net import Detect_Model
# from models.network.model_net2 import Detect_Model
from libs.tools import tools


class Train():
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.batch_queue = Read_Tfrecord(cfgs.TFRECORD, cfgs.batch_size).create_batch_queue()
        self.train_op, self.total_loss = self.create_clones(self.batch_queue)
        self.train(self.train_op, self.total_loss)

    @staticmethod
    def get_update_op():
        """
        Extremely important for BatchNorm
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            return tf.group(*update_ops)
        return None

    @staticmethod
    def sum_gradients(clone_grads):                        
        averaged_grads = []
        for grad_and_vars in zip(*clone_grads):
            grads = []
            var = grad_and_vars[0][1]
            try:
                for g, v in grad_and_vars:
                    assert v == var
                    grads.append(g)
                grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
            except:
                import pdb
                pdb.set_trace()
            
            averaged_grads.append((grad, v))
            
            # tf.summary.histogram("variables_and_gradients_" + grad.op.name, grad)
            # tf.summary.histogram("variables_and_gradients_" + v.op.name, v)
            # tf.summary.scalar("variables_and_gradients_" + grad.op.name+\
            #       '_mean/var_mean', tf.reduce_mean(grad)/tf.reduce_mean(var))
            # tf.summary.scalar("variables_and_gradients_" + v.op.name+'_mean',tf.reduce_mean(var))
        return averaged_grads

    def create_clones(self, batch_queue):        
        with tf.device('/cpu:0'):
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(cfgs.learning_rate,
                                                    global_step,
                                                    decay_steps=200,
                                                    decay_rate=0.9998,
                                                    staircase=True)
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.99, name='Momentum')

            
        # place clones
        cls_link_loss = 0 # for summary only
        gradients = []
        for clone_idx, gpu in enumerate(cfgs.gpus):
            reuse = clone_idx > 0
            with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
                with tf.name_scope('clone_{}'.format(clone_idx)) as clone_scope:
                    with tf.device(gpu) as clone_device:
                        b_image, b_cls_label, b_cls_weight, b_link_label, b_link_weight, b_gtboxes_and_labels = batch_queue #.dequeue()

                        # build model and loss
                        net = Detect_Model(b_image, is_training = True)
                        def reset_dims(image):
                            shape = image.shape.as_list()
                            if len(shape) != 4:
                                tf.expand_dims(image, axis=-1)
                        image_with_bbox = tools.draw_boxes_with_categories_and_scores_rotate(b_image[0], 
                                                                                    tf.concat((b_gtboxes_and_labels[0,:,:-2], 
                                                                                    tf.reshape(b_gtboxes_and_labels[0,:,-2], [-1, 1])), axis=1), 
                                                                                    b_gtboxes_and_labels[0,:,-1])
                        tf.summary.image('Image/origin', [tf.expand_dims(b_image[0], axis=0), tf.expand_dims(tf.cast(image_with_bbox, tf.float32), axis=0)])
                        for cls,num in cfgs.label_num_map.items():
                            if num != 0:
                                tf.summary.image('Mask/{}_labels'.format(cls), 
                                                 reset_dims(tf.expand_dims(tf.cast(b_cls_label[0, :, :, num-1], tf.float32), axis=0)))
                                tf.summary.image('Mask/{}_logits'.format(cls), 
                                                 reset_dims(tf.expand_dims(net.cls_logits[0, :, :, num-1, 1], axis=0)))

                        if cfgs.use_link and len(cfgs.label_num_map)-1 > 1:
                            net.build_loss(cls_labels = b_cls_label, 
                                            cls_weights = b_cls_weight, 
                                            link_labels = b_link_label, 
                                            link_weights = b_link_weight)
                        else:
                            net.build_loss(cls_labels = b_cls_label, 
                                            cls_weights = b_cls_weight)                           
                        # gather losses
                        losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                        # assert len(losses) ==  2
                        total_clone_loss = tf.add_n(losses) / len(cfgs.gpus)
                        cls_link_loss += total_clone_loss
                        # gather regularization loss and add to clone_0 only
                        if clone_idx == 0:
                            regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                            total_clone_loss = total_clone_loss + regularization_loss

                        # compute clone gradients
                        clone_gradients = optimizer.compute_gradients(total_clone_loss)
                        gradients.append(clone_gradients)
        
        tf.summary.scalar('learning_rate', learning_rate)    
        tf.summary.scalar('Loss/cls_link_loss', cls_link_loss)
        tf.summary.scalar('Loss/regularization_loss', regularization_loss)
        tf.summary.scalar('Loss/total_loss', total_clone_loss)
        
        # add all gradients together
        # note that the gradients do not need to be averaged, because the average operation has been done on loss.
        averaged_gradients = self.sum_gradients(gradients)
        apply_grad_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)
        
        train_ops = [apply_grad_op]
        
        bn_update_op = self.get_update_op()
        if bn_update_op is not None:
            train_ops.append(bn_update_op)
        
        # moving average
        if cfgs.using_moving_average:
            tf.logging.info('using moving average in training, with decay = %f'%(0.01))
            ema = tf.train.ExponentialMovingAverage(cfgs.moving_average_decay)
            ema_op = ema.apply(tf.trainable_variables())
            with tf.control_dependencies([apply_grad_op]):
                # ema after updating
                train_ops.append(tf.group(ema_op))
            
        train_op = control_flow_ops.with_dependencies(train_ops, cls_link_loss, name='train_op')

        return train_op, total_clone_loss

    def train(self, train_op, total_loss):
        summary_hook = tf.train.SummarySaverHook(save_steps=20,
                                                 output_dir=cfgs.checkpoint_path,
                                                 summary_op=tf.summary.merge_all())
        logging_hook = tf.train.LoggingTensorHook(tensors={'total_loss': 'clone_0/add_1:0', 'global_step': 'global_step:0'}, 
                                                  every_n_iter=1) 
        sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
        if cfgs.gpu_memory_fraction < 0:
            sess_config.gpu_options.allow_growth = True
        elif cfgs.gpu_memory_fraction > 0:
            sess_config.gpu_options.per_process_gpu_memory_fraction = cfgs.gpu_memory_fraction
        
        with tf.train.MonitoredTrainingSession(master='', is_chief=True,
            hooks = [tf.train.StopAtStepHook(last_step=cfgs.max_number_of_steps),  
                    tf.train.NanTensorHook(total_loss),
                    summary_hook, logging_hook],
            config = sess_config, 
            checkpoint_dir = cfgs.checkpoint_path,
            save_checkpoint_steps = 1000,
            stop_grace_period_secs = 120) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


if __name__ == "__main__":
    Train()