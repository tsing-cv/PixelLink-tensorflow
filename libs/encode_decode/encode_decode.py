from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import numpy as np
from collections import Counter
import cv2
import scipy.misc as scm
from libs.tools import img, tools, drawbox
from libs.bbox import bbox_convert
from config import cfgs
import pyximport; pyximport.install()    
from libs.encode_decode import decode

H, W = cfgs.score_map_shape
USE_MUIT_CLASS_MASK = cfgs.use_mult_class_mask
USE_MUIT_LEVEL_MASK = cfgs.use_mult_level_mask
if USE_MUIT_LEVEL_MASK:
    NUM_CLASSES = len(cfgs.label_num_map)-1
else:
    NUM_CLASSES = 1
NUM_NEIGHBOURS = cfgs.num_neighbours
WEIGHT_THRESHHOLD = cfgs.weight_threshhold
CLS_BORDER_WEITGHT_LAMBDA = cfgs.cls_border_weight_lambda
USE_LINK = cfgs.use_link
MINIMUM_MASK_REMAINED = cfgs.minimum_mask_remained
# =============================================================
# 获取临近点的坐标
# =============================================================
def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
            (x - 1, y), (x + 1, y),  \
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def get_neighbours_4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]


def get_neighbours(x, y):
    if NUM_NEIGHBOURS == 4:
        return get_neighbours_4(x, y)
    else:
        return get_neighbours_8(x, y)
    
def get_neighbours_fn():
    if NUM_NEIGHBOURS == 4:
        return get_neighbours_4, 4
    else:
        return get_neighbours_8, 8

def is_valid_cord(x, y, w, h):
    """
    判断一个坐标是否是合规的坐标
    """
    return x >=0 and x < w and y >= 0 and y < h

#=====================Ground Truth Calculation Begin==================
# ====================================================================
# 计算cls_label, cls_weight, link_label, link_weight
# TODO 在多分类的情况下，不再使用link
# ====================================================================
def tf_encode_single_image(xs, ys, labels):
    cls_label, cls_weight, link_label, link_weight, rects_with_labels = tf.py_func(encode_single_image, 
                                                                [xs, ys, labels],
                                                                [tf.int32, tf.float32, tf.int32, tf.float32, tf.float32])

    cls_label.set_shape([H, W, NUM_CLASSES])
    cls_weight.set_shape([H, W, NUM_CLASSES])
    link_label.set_shape([H, W, NUM_NEIGHBOURS])
    link_weight.set_shape([H, W, NUM_NEIGHBOURS])
    rects_with_labels = tf.reshape(rects_with_labels, [-1, 6])
    return cls_label, cls_weight, link_label, link_weight, rects_with_labels


def encode_single_image(normed_xs, normed_ys, labels):
    """
    Args:
        xs, ys: both in shape of (N, 4), 
            and N is the number of bboxes,
            their values are normalized to [0,1]
        labels: shape = (N,)
             -1: ignored
              1: text
    Return:
        cls_label
        cls_weight
        link_label
        link_weight
        rects_with_labels:(N, 6) each one is [x_c, y_c, w, h, theta, label]
    """
    # validate the args
    assert np.ndim(normed_xs) == 2
    assert np.shape(normed_xs)[-1] == 4
    assert np.shape(normed_xs) == np.shape(normed_ys)
    assert len(normed_xs) == len(labels)
    
    xs = normed_xs * W
    ys = normed_ys * H

    cls_label = np.zeros([H, W, NUM_CLASSES], dtype = np.int32)
    cls_weight = np.zeros([H, W, NUM_CLASSES], dtype = np.float32)
    link_label = np.zeros((H, W, NUM_NEIGHBOURS), dtype = np.int32)
    link_weight = np.ones((H, W, NUM_NEIGHBOURS), dtype = np.float32)

    bbox_masks = []
    border_masks = []
    mask = np.zeros([H, W], dtype = np.int32)
    rects_with_labels = []
    for cls,num in cfgs.label_num_map.items():
        if num <= 0:
            continue
        for bbox_idx, (bbox_xs, bbox_ys) in enumerate(list(zip(xs, ys))):
            # 1: preprocessing coordinates & filter background and ignore
            if np.max(bbox_xs)<0 or np.min(bbox_xs)>W:
                continue
            elif np.max(bbox_ys)<0 or np.min(bbox_ys)>H:
                continue
            else:
                bbox_xs, bbox_ys = np.clip(bbox_xs, 0, W), np.clip(bbox_ys, 0, H)
            

            # 2: [x1,y1,x2,y2,x3,y3,x4,y4,label] => [x_c,y_c,w,h, theta, label]
            eight_cords_rect = np.hstack((np.stack((bbox_xs, bbox_ys), axis=1).flatten(), labels[bbox_idx])).reshape([-1, 9])
            angle_cords = bbox_convert.back_forward_convert(eight_cords_rect, True).flatten()
            rects_with_labels.append(angle_cords)
            # print ("{}\nTwo Coordinate Type:\n\t{}\n\t{}\n".format("==="*20, eight_cords_rect, angle_cords, "==="*20))
            
            if labels[bbox_idx] == num:
                label = num if USE_MUIT_CLASS_MASK else 1
                level = num-1 if USE_MUIT_LEVEL_MASK else 0

                # 3: generate bbox mask and border mask
                bbox_mask_each_bbox = mask.copy()
                bbox_contours = img.points_to_contours(list(zip(bbox_xs, bbox_ys)))
                img.draw_contours(bbox_mask_each_bbox, bbox_contours, idx = -1, color = 1, border_width = -1)
                if np.sum(bbox_mask_each_bbox) <= MINIMUM_MASK_REMAINED:
                    continue
                bbox_masks.append(bbox_mask_each_bbox)
                bbox_border_mask_each_bbox = mask.copy()
                img.draw_contours(bbox_border_mask_each_bbox, bbox_contours, -1, color = 1, border_width = cfgs.bbox_border_width)
                border_masks.append(bbox_border_mask_each_bbox)
                
                # 4: encode cls_label and cls_weight
                cls_label[:, :, level] = np.clip(np.where(bbox_mask_each_bbox, cls_label[:, :, level]+label, cls_label[:, :, level]), a_min=0, a_max=label)
                if cfgs.cls_weight_with_border_balanced:
                    # ---------------------avoid overlap area have too large weight------------------------------
                    # TODO address overlap areas
                    before_max = np.max(cls_weight[:, :, level])
                    after_max = 1/np.sum(bbox_mask_each_bbox) if np.sum(bbox_mask_each_bbox)>0 else 1
                    cls_weight[:, :, level] += bbox_mask_each_bbox*after_max
                    np.clip(cls_weight[:, :, level], a_min=0, a_max=np.max([before_max, after_max]))
                    # ---------------------set larger weight for border area ------------------------------------
                    cls_weight[:, :, level] += bbox_border_mask_each_bbox *CLS_BORDER_WEITGHT_LAMBDA 
                
                
                # 5: encode link_label and link_weight
                if USE_LINK and not USE_MUIT_LEVEL_MASK:
                    bbox_cls_cords = np.where(bbox_mask_each_bbox)
                    link_label[bbox_cls_cords] = 1
                    bbox_border_cords = np.where(bbox_border_mask_each_bbox)
                    border_points = list(zip(*bbox_border_cords))
                    for y, x in border_points:
                        neighbours = get_neighbours(x, y)
                        for n_idx, (nx, ny) in enumerate(neighbours):
                            if not is_valid_cord(nx, ny, W, H) or not bbox_mask_each_bbox[nx, ny]:
                                link_label[y, x, n_idx] = 0


    # num_positive_bboxes_each_cls = Counter(labels)
    # num_positive_pixels_each_cls = np.sum(cls_label, axis=2)
    num_total_positive_pixels = np.sum(cls_label)
    # overlap area will greater then 1
    if cfgs.cls_weight_with_border_balanced:
        cls_weight = num_total_positive_pixels*cls_weight
        cls_weight = np.clip(cls_weight, a_min=0, a_max=WEIGHT_THRESHHOLD)  
    else:
        cls_weight += cls_label
    cls_weight = np.cast["float32"](cls_weight)
    if USE_LINK and not USE_MUIT_LEVEL_MASK:
        link_weight *= cls_weight

    return cls_label, cls_weight, link_label, link_weight, rects_with_labels
#=====================Ground Truth Calculation End====================



#============================Decode Begin=============================
# ====================================================================
# score => mask => bbox
# ====================================================================
def tf_decode_score_map_to_mask_in_batch(cls_scores, link_scores):
    """
    由得分图到mask
    :param cls_scores: 类别得分
    :param link_scores: link得分
    :return:
    """
    masks = tf.py_func(decode_batch, 
                       [cls_scores, link_scores], tf.int32)
    b, h, w = cls_scores.shape.as_list()
    masks.set_shape([b, h, w])
    return masks

def decode_batch(cls_scores, link_scores, 
                 cls_conf_threshold = 0.7, link_conf_threshold = 0.8):
    
    batch_size = cls_scores.shape[0]
    batch_mask = []
    for image_idx in range(batch_size):
        image_pos_cls_scores = cls_scores[image_idx, :, :]
        image_pos_link_scores = link_scores[image_idx, :, :, :]    
        mask = decode_image(image_pos_cls_scores, image_pos_link_scores, 
                            cls_conf_threshold, link_conf_threshold)
        batch_mask.append(mask)
    return np.asarray(batch_mask, np.int32)

def decode_image(cls_scores, link_scores, 
                 cls_conf_threshold, link_conf_threshold):
    if cfgs.decode_image_by_join:
        mask =  decode.decode_image_by_join(cls_scores, link_scores, 
                 cls_conf_threshold, link_conf_threshold)
        return mask
    else:
        # TODO
        return None #decode_image_by_border(cls_scores, link_scores, cls_conf_threshold, link_conf_threshold)

def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta]. 
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h

def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x
    
    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y
    
    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

def mask_to_bboxes(mask, image_shape, min_area = 25, min_aspect_ratio = 0.001):
    image_h, image_w = image_shape[0: 2]
    bboxes = []
    max_bbox_idx = mask.max()
    mask = img.resize(img = mask, size = (image_w, image_h), 
                      interpolation = cv2.INTER_NEAREST)
    
    for bbox_idx in range(1, max_bbox_idx + 1):
        bbox_mask = mask == bbox_idx
        # if bbox_mask.sum() < 10:
        #     continue
        cnts = img.find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)
        
        w, h = rect[2: -1]

        if min(w, h) < min_aspect_ratio*min(image_w, image_h):
            continue
        if rect_area < min_area:
            continue
        # if max(w, h) * 1.0 / min(w, h) < 2:
        #     continue
        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)
        
    return bboxes


def mask_to_bboxes_v2(mask, image_shape, min_area = 25, min_aspect_ratio = 0.001):
    image_h, image_w = image_shape[0: 2]
    mask = img.resize(img = mask, size = (image_w, image_h), 
                      interpolation = cv2.INTER_NEAREST)
    # img.imshow('1',mask)
    
    bboxes = []
    for cls,num in cfgs.label_num_map.items():
        if num == 0:
            continue
        cls_mask = mask==num
        # img.imshow('2', cls_mask)
        cnts = img.find_contours(cls_mask)
        if len(cnts) == 0:
            continue
        for cnt in cnts:
            # cnt = cnts[0]
            rect, rect_area = min_area_rect(cnt)
            w, h = rect[2: -1]

            if min(w, h) < min_aspect_ratio*min(image_w, image_h):
                continue
            if rect_area < min_area:
                continue
            # if max(w, h) * 1.0 / min(w, h) < 2:
            #     continue
            xys = rect_to_xys(rect, image_shape)
            
            bboxes.append(np.hstack((xys, num)))
    return bboxes

#============================Decode End===============================
