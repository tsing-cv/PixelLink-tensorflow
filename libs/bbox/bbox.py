#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import tensorflow as tf
import cv2
import os
# print (os.getcwd())
from libs.tools import img


def bboxes_resize(bbox_ref, bboxes, xs, ys, name=None):
    """
    根据参考框resize边框，假定变换后为[0, 0, 1, 1]，对更新裁剪后的图片有用
    """
    # Tensors inputs.
    with tf.name_scope(name, 'bboxes_resize'):
        h_ref = bbox_ref[2] - bbox_ref[0]
        w_ref = bbox_ref[3] - bbox_ref[1]
        
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        xs = xs - bbox_ref[1]
        ys = ys - bbox_ref[0]
        
        # Scale.
        s = tf.stack([h_ref, w_ref, h_ref, w_ref])
        bboxes = bboxes / s
        xs = xs / w_ref
        ys = ys / h_ref
        
        return bboxes, xs, ys



# def bboxes_filter_center(labels, bboxes, scope=None):
#     """Filter out bounding boxes whose center are not in
#     the rectangle [0, 0, 1, 1] + margins. The margin Tensor
#     can be used to enforce or loosen this condition.
# 
#     Return:
#       labels, bboxes: Filtered elements.
#     """
#     with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
#         cy = (bboxes[:, 0] + bboxes[:, 2]) / 2.
#         cx = (bboxes[:, 1] + bboxes[:, 3]) / 2.
#         mask = tf.greater(cy, 0.)
#         mask = tf.logical_and(mask, tf.greater(cx, 0.))
#         mask = tf.logical_and(mask, tf.less(cy, 1.))
#         mask = tf.logical_and(mask, tf.less(cx, 1.))
#         # Boolean masking...
#         labels = tf.boolean_mask(labels, mask)
#         bboxes = tf.boolean_mask(bboxes, mask)
#         return labels, bboxes

def bboxes_filter_overlap(labels, bboxes,xs, ys, threshold, scope=None, assign_value = None):
    """
    根据参考框[0, 0, 1, 1]过滤边框。

    Return:
      labels, bboxes: Filtered (or newly assigned) elements.
    """
    with tf.name_scope(scope, 'bboxes_filter_overlap', [labels, bboxes]):
        scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),bboxes)
        
        if assign_value is not None:
            mask = scores < threshold
            # *******************************************************
            # NOTE because we always assign background with label 0
            # *******************************************************
            mask = tf.logical_and(mask, tf.greater(labels, 0))
            labels = tf.where(mask, tf.ones_like(labels) * assign_value, labels)
        else:
            mask = scores > threshold
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
            scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),bboxes)
            xs = tf.boolean_mask(xs, mask)
            ys = tf.boolean_mask(ys, mask)
        return labels, bboxes, xs, ys


def bboxes_filter_by_shorter_side(labels, bboxes, xs, ys, min_lenght_of_side = 16, max_lenght_of_side = 32, assign_value = None):
    """
    通过最短边长过滤边框
    """
    with tf.name_scope('bboxes_filter_by_shorter_side', [labels, bboxes]):
        bbox_rects = img.tf_min_area_rect(xs, ys)
        ws, hs = bbox_rects[:, 2], bbox_rects[:, 3]
        shorter_sides = tf.minimum(ws, hs)
        if assign_value is not None:
            mask = tf.logical_or(shorter_sides < min_lenght_of_side, shorter_sides > max_lenght_of_side)
            mask = tf.logical_and(mask, tf.greater(labels, 0))
            labels = tf.where(mask, tf.ones_like(labels) * assign_value, labels)
        else:
            mask = tf.logical_and(shorter_sides >= min_lenght_of_side, shorter_sides <= max_lenght_of_side)
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
            xs = tf.boolean_mask(xs, mask)
            ys = tf.boolean_mask(ys, mask)
        return labels, bboxes, xs, ys
    
def safe_divide(numerator, denominator, name):
    """
    两个值相除，如果分母为0，返回0
    Args:
        numerator: A real `Tensor`.
        denominator: A real `Tensor`, with dtype matching `numerator`.
        name: Name for the returned op.
    Returns:
        0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(tf.greater(denominator, 0),
                    tf.divide(numerator, denominator),
                    tf.zeros_like(numerator),
                    name=name)

def bboxes_intersection(bbox_ref, bboxes, name=None):
    """
    计算参考框和一个边框集的重合。
    Args:
        bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
        bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
        (N,) Tensor with relative intersection.
    """
    with tf.name_scope(name, 'bboxes_intersection'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = safe_divide(inter_vol, bboxes_vol, 'intersection')
        return scores


def bboxes_matching(bboxes, gxs, gys, gignored, matching_threshold = 0.5, scope=None):
    """
    方向框集和groundthruth集匹配
    不接受批次输入。
    算法是这样的，对每个检测框，核对是否有一个groundtruth框与之对应，如果没有，就当做错误的正类，如果该
    groundtruth框与已经有配对，也当做错误的正类，细节参考Pascal VOC的文档

    Args:
        rbboxes: Nx4 Tensors. Detected objects, sorted by score;
        gbboxes: Groundtruth bounding boxes. May be zero padded, hence
        zero-class objects are ignored.
        matching_threshold: Threshold for a positive match.
    Return: Tuple of:
       n_gbboxes: Scalar Tensor with number of groundtruth boxes (may difer from
         size because of zero padding).
       tp_match: (N,)-shaped boolean Tensor containing with True Positives.
       fp_match: (N,)-shaped boolean Tensor containing with False Positives.
    """
    with tf.name_scope(scope, 'bboxes_matching_single',[bboxes, gxs, gys, gignored]):
        # Number of groundtruth boxes.
        gignored = tf.cast(gignored, dtype = tf.bool)
        n_gbboxes = tf.count_nonzero(tf.logical_not(gignored))
        # Grountruth matching arrays.
        gmatch = tf.zeros(tf.shape(gignored), dtype=tf.bool)
        grange = tf.range(tf.size(gignored), dtype=tf.int32)
        
        # Number of detected boxes        
        n_bboxes = tf.shape(bboxes)[0]
        rshape = (n_bboxes, )
        # True/False positive matching TensorArrays.
        # ta is short for TensorArray
        ta_tp_bool = tf.TensorArray(tf.bool, size=n_bboxes, dynamic_size=False, infer_shape=True)
        ta_fp_bool = tf.TensorArray(tf.bool, size=n_bboxes, dynamic_size=False, infer_shape=True)
        
        n_ignored_det = 0
        # Loop over returned objects.
        def m_condition(i, ta_tp, ta_fp, gmatch, n_ignored_det):
            r = tf.less(i, tf.shape(bboxes)[0])
            return r

        def m_body(i, ta_tp, ta_fp, gmatch, n_ignored_det):
            # Jaccard score with groundtruth bboxes.
            rbbox = bboxes[i, :]
            # rbbox = tf.Print(rbbox, [rbbox])
            jaccard = bboxes_jaccard(rbbox, gxs, gys)

            # Best fit, checking it's above threshold.
            idxmax = tf.cast(tf.argmax(jaccard, axis=0), dtype = tf.int32)
            
            jcdmax = jaccard[idxmax]
            match = jcdmax > matching_threshold
            existing_match = gmatch[idxmax]
            not_ignored = tf.logical_not(gignored[idxmax])

            n_ignored_det = n_ignored_det + tf.cast(gignored[idxmax], tf.int32)
            # TP: match & no previous match and FP: previous match | no match.
            # If ignored: no record, i.e FP=False and TP=False.
            tp = tf.logical_and(not_ignored, tf.logical_and(match, tf.logical_not(existing_match)))
            ta_tp = ta_tp.write(i, tp)
            
            fp = tf.logical_and(not_ignored, tf.logical_or(existing_match, tf.logical_not(match)))
            ta_fp = ta_fp.write(i, fp)
            
            # Update grountruth match.
            mask = tf.logical_and(tf.equal(grange, idxmax), tf.logical_and(not_ignored, match))
            gmatch = tf.logical_or(gmatch, mask)
            return [i+1, ta_tp, ta_fp, gmatch,n_ignored_det]
        # Main loop definition.
        i = 0
        [i, ta_tp_bool, ta_fp_bool, gmatch, n_ignored_det] = tf.while_loop(m_condition, m_body,
                                                                [i, ta_tp_bool, ta_fp_bool, gmatch, n_ignored_det],
                                                                parallel_iterations=1,
                                                                back_prop=False)
        # TensorArrays to Tensors and reshape.
        tp_match = tf.reshape(ta_tp_bool.stack(), rshape)
        fp_match = tf.reshape(ta_fp_bool.stack(), rshape)

        return n_gbboxes, tp_match, fp_match

def bboxes_jaccard(bbox, gxs, gys):
    """
    计算边框的jaccard值
    :param bbox:
    :param gxs:
    :param gys:
    :return:
        jaccard值，也就IoU值
    """
    jaccard = tf.py_func(np_bboxes_jaccard, [bbox, gxs, gys], tf.float32)
    jaccard.set_shape([None, ])
    return jaccard

def np_bboxes_jaccard(bbox, gxs, gys):
    # assert np.shape(bbox) == (8,)
    bbox_points = np.reshape(bbox, (4, 2))
    cnts = img.points_to_contours(bbox_points)
    
    # contruct a 0-1 mask to draw contours on
    xmax = np.max(bbox_points[:, 0])
    xmax = max(xmax, np.max(gxs)) + 10
    ymax = np.max(bbox_points[:, 1])
    ymax = max(ymax, np.max(gys)) + 10
    mask = img.black((ymax, xmax))
    
    # draw bbox on the mask
    bbox_mask = mask.copy()
    img.draw_contours(bbox_mask, cnts, idx = -1, color = 1, border_width = -1)
    jaccard = np.zeros((len(gxs),), dtype = np.float32)
    # draw ground truth 
    for gt_idx, gt_bbox in enumerate(zip(gxs, gys)):
        gt_mask = mask.copy()
        gt_bbox = np.transpose(gt_bbox)
        # assert gt_bbox.shape == (4, 2)
        gt_cnts = img.points_to_contours(gt_bbox)
        img.draw_contours(gt_mask, gt_cnts, idx = -1, color = 1, border_width = -1)
        
        intersect = np.sum(bbox_mask * gt_mask)
        union = np.sum(bbox_mask + gt_mask >= 1)
        # assert intersect == np.sum(bbox_mask * gt_mask)
        # assert union == np.sum((bbox_mask + gt_mask) > 0)
        iou = intersect * 1.0 / union
        jaccard[gt_idx] = iou
    return jaccard
    
