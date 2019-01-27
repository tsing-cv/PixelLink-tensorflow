# -*- coding:utf-8 -*-
"""
Author: wqq
Time: 2018-12-21
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import cv2
from config import cfgs 

color_map = [(144, 238, 144), (139, 0, 0), (205, 121, 205), (139, 0, 139), (155, 100, 23), (144, 30, 255), 
            (255, 130, 171), (255, 0, 0), (255, 127, 0), (255, 160, 122), (192, 255, 62), (0, 139, 139), (48, 255, 155), (0, 0, 139), (255, 0, 127),
            (127, 255, 0), (121, 205, 205), (255, 0, 255), (0, 255, 255), (230, 230, 100), (155, 48, 255),
            (30, 144, 255), (219, 112, 147), (0, 191, 255), (171, 130, 255), (205, 41, 144), (23, 100, 155), (255, 62, 150), 
            (255, 144, 30), (255, 165, 0), (0, 139, 0)]
rgb_mean = [123., 117., 104.]

def make_folder(path):
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)

    return path


def draw_box_cv(img, boxes, labels, scores=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) + np.array(rgb_mean)
    # img = img + np.array([104., 117., 123.])
    boxes = np.cast['int64'](boxes)#boxes.astype(np.int64)
    labels = np.cast['int32'](labels)#labels.astype(np.int32)
    img = np.array(img*255/np.max(img), np.uint8)

    num_of_object = 0
    for i, box in enumerate(boxes):
        if labels[i] <= 0:
            continue
        if len(box) == 5:
            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
            # print (x_c, y_c, w, h, theta)
            num_of_object += 1

            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            cv2.drawContours(img, [rect], -1, color_map[labels[i]], 2)
            cv2.rectangle(img,
                        pt1=(x_c, y_c),
                        pt2=(x_c+50, y_c+12),
                        color=color_map[labels[i]],
                        thickness=-1)
        else:
            points = np.array([[box[0], box[1]], 
                                [box[2], box[3]], 
                                [box[4], box[5]], 
                                [box[6], box[7]]], np.int32)
            pts = points.reshape((-1, 1, 2))
            cv2.polylines(img, np.array([pts], np.int32), True, color_map[labels[i]], 2)
        label_num_map_reverse = {v:k for k,v in cfgs.label_num_map.items()}
        category = label_num_map_reverse[labels[i]]
        if scores:
            category = category+": "+str(scores[i])
        cv2.putText(img,
                    text=category,
                    org=(x_c, y_c+10),
                    fontFace=1,
                    fontScale=1,
                    thickness=2,
                    color=color_map[labels[i]+1])
    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=1,
                color=(255, 0, 0))
    img = img[:, :, ::-1]
    return img
def draw_boxes_with_categories_and_scores_rotate(image, boxes, labels, scores=None):
    if scores:
        img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                        inp=[image, boxes, labels, scores],
                                        Tout=[tf.uint8])
    else:
        img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                        inp=[image, boxes, labels],
                                        Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(image))
    return img_tensor_with_boxes


def draw_boxes_with_categories_and_scores_batch(image, boxes, labels, scores=None):
    batch_image = []
    for image_idx in range(cfgs.batch_size):
        img = image[image_idx, ...]
        box = boxes[image_idx]
        label = labels[image_idx]
        score = scores[image_idx] if scores else None
        img_tensor_with_boxes = draw_boxes_with_categories_and_scores_rotate(img, box, label, score)
        batch_image.append(img_tensor_with_boxes)
    return batch_image
