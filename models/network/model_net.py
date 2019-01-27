# -*-coding:utf-8 -*-
"""
Author: tsing
Time: 2018-12-18
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf 
slim = tf.contrib.slim
import os 
print (os.getcwd())
from models.network import model_net3
from config import cfgs 

# *********************************************************************
#      OHNM
# *********************************************************************
default_num_neg_mask = int(0.04*cfgs.train_image_shape["height"]*cfgs.train_image_shape["width"]*(len(cfgs.label_num_map)-1))
def OHNM_single_image(scores, n_pos, neg_mask):
    n_neg = tf.cond(n_pos>0, 
                    lambda: n_pos * cfgs.max_neg_pos_ratio,
                    lambda: tf.constant(default_num_neg_mask, dtype=tf.int32))
    max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))
    n_neg = tf.cast(tf.maximum(n_neg, max_neg_entries), tf.int32)

    def has_neg():
        # scores where mask == 0 will be leaved
        neg_scores = tf.boolean_mask(scores, neg_mask)
        vals, _ = tf.nn.top_k(-neg_scores, k=n_neg)
        threshold = vals[-1]
        return tf.logical_and(neg_mask, scores<=-threshold)

    def no_neg():
        return tf.zeros_like(neg_mask)

    selected_neg_mask = tf.cast(tf.cond(n_neg>0, has_neg, no_neg), tf.int32)
    return selected_neg_mask

def OHNM_batch(neg_scores, pos_mask, neg_mask, batch_size=cfgs.batch_size):
    selected_neg_mask = []
    for image_idx in range(batch_size):
        image_neg_scores = neg_scores[image_idx, :]
        image_neg_mask = neg_mask[image_idx, :]
        image_pos_mask = pos_mask[image_idx, :]
        n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
        selected_neg_mask.append(OHNM_single_image(image_neg_scores, n_pos, image_neg_mask))

    return tf.stack(selected_neg_mask)

# *********************************************************************
#      Detect_Model
# *********************************************************************
class Detect_Model():
    def __init__(self, inputs, is_training):
        self.inputs = inputs 
        self.is_training = is_training
        self.basenet()
        self.logits_scores()

    def basenet(self):
        """Base Network
        """
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer = tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as sc:
                self.arg_scope = sc
                self.net, self.end_points = vgg.basenet(inputs = self.inputs, dilation=cfgs.within_dilation)

    def _horizontal_conv1x1_layer(self, input_layer, num_classes, scope):
        """Horizotal conv1x1
        """
        with slim.arg_scope(self.arg_scope):
            logits = slim.conv2d(input_layer, num_classes, [1, 1], stride=1, activation_fn=None, 
                                scope='scope_from_{}'.format(scope), normalizer_fn=None)
            if cfgs.dropout_ratio > 0 and cfgs.dropout_ratio <= 1:
                keep_prob = 1. - cfgs.dropout_ratio
                tf.logging.info('Using Dropout, with keep_prob = {}'.format(keep_prob))
                logits = tf.nn.dropout(logits, keep_prob)
            
            return logits

    def _upsample_layer(self, layer, target_layer):
        """Upsample resize

        resize shape of layer to target_layer's shape
        """
        target_shape = tf.shape(target_layer)[1: -1]
        return tf.image.resize_images(layer, target_shape)

    def _cascade_two_layer(self, scope, num_classes, fuse_method):
        with tf.variable_scope(scope):
            print ("{}\n{}".format("@@@"*20, "Cascade Layer"))
            for i, current_layer_name in enumerate(reversed(cfgs.feat_layers)):
                # print (i, current_layer_name)
                current_layer = self.end_points[current_layer_name]
                current_score_map = self._horizontal_conv1x1_layer(current_layer, num_classes, current_layer_name)
                # 判断是不是最小的特征层
                if current_layer_name == cfgs.feat_layers[-1]:
                    cascade_layer = current_score_map
                else:
                    upsample_layer = self._upsample_layer(cascade_layer, current_score_map)
                    if fuse_method == 'concat':
                        cascade_layer = tf.concat([current_score_map, upsample_layer], axis=3)
                    elif fuse_method == "sum":
                        cascade_layer = current_score_map + upsample_layer
                        
                    print ("\t{}".format(cascade_layer))
            print ("@@@"*20)
        return cascade_layer

    def _flat(self, values):
        shape = values.shape.as_list()
        return tf.reshape(values, shape=[shape[0], -1, shape[-1]])
    
    def logits_scores(self):
        """Get logits, scores and flatten them
        """
        if cfgs.fuse_method == "sum":
            self.cls_logits = self._cascade_two_layer(scope='fuse_feature_cls', num_classes=(len(cfgs.label_num_map)-1)*2, fuse_method='sum')
            if cfgs.use_link and len(cfgs.label_num_map)-1 > 1:
                self.link_logits = self._cascade_two_layer(scope='fuse_feature_link', num_classes=cfgs.num_neighbours*2, fuse_method='sum')
        elif cfgs.fuse_method == "concat128":
            base_map = self._cascade_two_layer(scope='fuse_feature', num_classes=128, fuse_method='concat')
            self.cls_logits = self._horizontal_conv1x1_layer(base_map, (len(cfgs.label_num_map)-1)*2, scope='cls')
            if cfgs.use_link and len(cfgs.label_num_map)-1 > 1:
                self.link_logits = self._horizontal_conv1x1_layer(base_map, cfgs.num_neighbours*2, scope='link')
        elif cfgs.fuse_method == "sum128":
            base_map = self._cascade_two_layer(scope='fuse_feature', num_classes=128, fuse_method='sum')
            self.cls_logits = self._horizontal_conv1x1_layer(base_map, (len(cfgs.label_num_map)-1)*2, scope='cls')
            if cfgs.use_link and len(cfgs.label_num_map)-1 > 1:
                self.link_logits = self._horizontal_conv1x1_layer(base_map, cfgs.num_neighbours*2, scope='link')


        cls_shape = self.cls_logits.shape.as_list()
        self.cls_logits = tf.reshape(self.cls_logits,
                                    [cls_shape[0], cls_shape[1], cls_shape[2], (len(cfgs.label_num_map)-1), 2])
        self.cls_scores = tf.nn.softmax(self.cls_logits)
        self.cls_logits_flatten = self._flat(self.cls_logits)
        self.cls_scores_flatten = self._flat(self.cls_scores)
        self.cls_pos_scores = self.cls_scores[:, :, :, :, 1]
        
        if cfgs.use_link and len(cfgs.label_num_map)-1 > 1:
            link_shape = self.link_logits.shape.as_list()
            self.link_logits = tf.reshape(self.link_logits,
                                        [link_shape[0], link_shape[1], link_shape[2], cfgs.num_neighbours, 2])
            self.link_scores = tf.nn.softmax(self.link_logits)
            self.link_pos_scores = self.link_scores[:, :, :, :, 1]

    def build_loss(self, cls_labels, cls_weights,
                   link_labels=None, link_weights=None):
        # class loss **********************************************************
        cls_labels_flatten = tf.reshape(cls_labels, [cfgs.batch_size, -1])
        pos_cls_weights_flatten = tf.reshape(cls_weights, [cfgs.batch_size, -1])
        pos_mask = tf.greater(cls_labels_flatten, 0)
        neg_mask = tf.equal(cls_labels_flatten, 0)

        n_pos = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.float32))
        with tf.name_scope('cls_loss'):
            def no_pos():
                return tf.constant(0.0)
            def has_pos():
                assert self.cls_logits_flatten.shape[: 2] == pos_mask.shape[: 2], \
                    "The first two dims is not equal\nlogits shape is:{}\nlabels shape is:{}".format(
                        self.cls_logits_flatten.shape, pos_mask.shape)
                cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=self.cls_logits_flatten,
                                labels=tf.cast(pos_mask, dtype=tf.int32))
                # print (self.cls_scores_flatten)
                cls_neg_scores = self.cls_scores_flatten[:, :, 0]
                selected_neg_cls_mask = OHNM_batch(cls_neg_scores, pos_mask, neg_mask)

                cls_weights = pos_cls_weights_flatten+tf.cast(selected_neg_cls_mask, tf.float32)*cfgs.OHNM_weight_lambda
                n_neg = tf.cast(tf.reduce_sum(selected_neg_cls_mask), tf.float32)
                return tf.reduce_sum(cls_loss*cls_weights) / (n_neg+n_pos)
            cls_loss = tf.cond(n_pos>0, has_pos, no_pos)
            tf.add_to_collection(tf.GraphKeys.LOSSES, cls_loss)
            tf.summary.scalar("Loss/cls_loss", cls_loss)

        # link loss ***********************************************************
        if cfgs.use_link and len(cfgs.label_num_map)-1 > 1:
            with tf.name_scope('link_loss'):
                def no_pos():
                    # there are two losses, negative loss and positive loss
                    return tf.constant(.0), tf.constant(.0)

                def has_pos():
                    assert self.link_logits_flatten.shape[: 2] == pos_mask.shape[: 2], \
                        "The first two dims is not equal\nlogits shape is:{}\nlabels shape is:{}".format(
                            self.link_logits_flatten.shape, pos_mask.shape)
                    link_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=self.link_logits,
                                    labels=tf.cast(link_labels, dtype=tf.int32))

                    def get_loss(label):
                        link_mask = tf.equal(link_labels, label)
                        link_weight = link_weights * tf.cast(link_mask, tf.float32)
                        n_links = tf.reduce_sum(link_weight)
                        loss = tf.reduce_sum(link_loss*link_weight)/n_links
                        return loss

                    neg_loss = get_loss(0)
                    pos_loss = get_loss(1)
                    return neg_loss, pos_loss

                neg_link_loss, pos_link_loss = tf.cond(n_pos>0, has_pos, no_pos)
                link_loss = pos_link_loss + neg_link_loss * 1.0
                tf.add_to_collection(tf.GraphKeys.LOSSES, link_loss)
                tf.summary.scalar("Link/link_loss", link_loss)
                tf.summary.scalar("Link/pos_link_loss", pos_link_loss)
                tf.summary.scalar("Link/neg_link_loss", neg_link_loss)



