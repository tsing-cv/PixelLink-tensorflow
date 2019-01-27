# -*- coding:utf-8 -*-
"""
Author: wqq
Time: 2018-12-27
"""
from enum import Enum, IntEnum
import numpy as np

import tensorflow as tf
from libs.tools import img
from libs.bbox import bbox
from config import cfgs

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import tensor_shape
import cv2

slim = tf.contrib.slim

# Resizing strategies.
Resize = IntEnum('Resize', ('NONE',                # Nothing!
                            'CENTRAL_CROP',        # Crop (and pad if necessary).
                            'PAD_AND_RESIZE',      # Pad, and resize to output shape.
                            'WARP_RESIZE'))        # Warp resize.
_R_MEAN, _G_MEAN, _B_MEAN = cfgs.rgb_mean
MAX_EXPAND_SCALE = cfgs.max_expand_scale
LABEL_IGNORE = -1
USING_SHORTER_SIDE_FILTERING = True
USE_ROT_90 = True
USE_ROT_RANDOM = True
MIN_LENGHT_OF_SIDE = cfgs.min_lenght_of_side
AREA_RANGE = cfgs.crop_area_range
# ***************************************************************************
# NOTE function tools
# ***************************************************************************
def _assert(cond, ex_type, msg):
    """A polymorphic assert, works with tensors and boolean expressions.
    If `cond` is not a tensor, behave like an ordinary assert statement, except
    that a empty list is returned. If `cond` is a tensor, return a list
    containing a single TensorFlow assert op.
    Args:
      cond: Something evaluates to a boolean value. May be a tensor.
      ex_type: The exception class to use.
      msg: The error message.
    Returns:
      A list, containing at most one assert op.
    """
    if _is_tensor(cond):
        return [control_flow_ops.Assert(cond, [msg])]
    else:
        if not cond:
            raise ex_type(msg)
        else:
            return []


def _is_tensor(x):
    """Returns `True` if `x` is a symbolic tensor-like object.
    Args:
      x: A python object to check.
    Returns:
      `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
    """
    return isinstance(x, (tf.Tensor, tf.Variable))


def _ImageDimensions(image):
    """Returns the dimensions of an image tensor.
    Args:
      image: A 3-D Tensor of shape `[height, width, channels]`.
    Returns:
      A list of `[height, width, channels]` corresponding to the dimensions of the
        input image.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def _Check3DImage(image, require_static=True):
    """Assert that we are working with properly shaped image.
    Args:
      image: 3-D Tensor of shape [height, width, channels]
        require_static: If `True`, requires that all dimensions of `image` are
        known and non-zero.
    Raises:
      ValueError: if `image.shape` is not a 3-vector.
    Returns:
      An empty list, if `image` has fully defined dimensions. Otherwise, a list
        containing an assert op is returned.
    """
    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' must be three-dimensional.")
    if require_static and not image_shape.is_fully_defined():
        raise ValueError("'image' must be fully defined.")
    if any(x == 0 for x in image_shape):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                         image_shape)
    if not image_shape.is_fully_defined():
        return [tf.assert_positive(tf.shape(image),
                                   ["all dims of 'image.shape' " "must be > 0."])]
    else:
        return []

def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    """Resize an image and bounding boxes.
    """
    # Resize image.
    with tf.name_scope('resize_image'):
        height, width, channels = _ImageDimensions(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image




# ***************************************************************************
# NOTE Ndarray preprocessing
# ***************************************************************************
class Array_Image_Preprocessing():
    @staticmethod
    def np_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
        """Re-convert to original image distribution, and convert to int if
        necessary. Numpy version.

        Returns:
        Centered image.
        """
        img = np.copy(image)
        img += np.array(means, dtype=img.dtype)
        if to_int:
            img = img.astype(np.uint8)
        return img

    @staticmethod
    def rotate_image(image, xs, ys, angle_range=[-20, 10]):
        low,high = angle_range
        rotation_angle = np.random.randint(low, high)
        scale = np.random.uniform(low = 1.05, high = 1.25)
        # scale = 1.0
        h, w = image.shape[0:2]
        # rotate image
        image, M = img.rotate_about_center(image, rotation_angle, scale = scale)
        
        nh, nw = image.shape[0: 2]
        
        # rotate bboxes
        xs = xs * w
        ys = ys * h
        def rotate_xys(xs, ys):
            xs = np.reshape(xs, -1)
            ys = np.reshape(ys, -1)
            xs, ys = np.dot(M, np.transpose([xs, ys, 1]))
            xs = np.reshape(xs, (-1, 4))
            ys = np.reshape(ys, (-1, 4))
            return xs, ys
        xs, ys = rotate_xys(xs, ys)
        xs = xs * 1.0 / nw
        ys = ys * 1.0 / nh
        xmin = np.min(xs, axis = 1)  
        xmax = np.max(xs, axis = 1)
        ymin = np.min(ys, axis = 1)
        ymax = np.max(ys, axis = 1)
        
        bboxes = np.clip(np.transpose(np.asarray([ymin, xmin, ymax, xmax])), a_min=0, a_max=1)
        image = np.asarray(image, np.uint8)
        
        return image, bboxes, xs, ys

# ***************************************************************************
# NOTE Tensor preprocessing
# ***************************************************************************
class Tensor_Image_Preprocessing():
    # --------------------------------------------------------------
    #   NOTE whiten and unwhiten
    # --------------------------------------------------------------
    @staticmethod
    def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
        """Subtracts the given means from each image channel.

        Returns:
            the centered image.
        """
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        mean = tf.constant(means, dtype=image.dtype)
        image = image - mean
        return image

    @staticmethod
    def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
        """Re-convert to original image distribution, and convert to int if
        necessary.

        Returns:
        Centered image.
        """
        mean = tf.constant(means, dtype=image.dtype)
        image = image + mean
        if to_int:
            image = tf.cast(image, tf.int32)
        return image


    def tf_summary_image(self, image, bboxes, name='image', unwhitened=False):
        """Add image with bounding boxes to summary.
        """
        if unwhitened:
            image = self.tf_image_unwhitened(image)
        image = tf.expand_dims(image, 0)
        bboxes = tf.expand_dims(bboxes, 0)
        image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
        tf.summary.image(name, image_with_box)


    # --------------------------------------------------------------
    #   NOTE 
    # --------------------------------------------------------------
    @staticmethod
    def apply_with_random_selector(x, func, num_cases):
        """Computes func(x, sel), with sel sampled from [0...num_cases-1].

        Args:
            x: input Tensor.
            func: Python function to apply.
            num_cases: Python int32, number of cases to sample sel from.

        Returns:
            The result of func(x, sel), where func receives the value of the
            selector as a python integer, but sel is sampled dynamically.
        """
        sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
        # Pass the real x only to one of the func calls.
        return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
                for case in range(num_cases)])[0]

    # --------------------------------------------------------------
    #   NOTE random color
    # --------------------------------------------------------------
    @staticmethod
    def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
        """Distort the color of a Tensor image.

        Each color distortion is non-commutative and thus ordering of the color ops
        matters. Ideally we would randomly permute the ordering of the color ops.
        Rather then adding that level of complication, we select a distinct ordering
        of color ops for each preprocessing thread.

        Args:
            image: 3-D Tensor containing single image in [0, 1].
            color_ordering: Python int, a type of distortion (valid values: 0-3).
            fast_mode: Avoids slower ops (random_hue and random_contrast)
            scope: Optional scope for name_scope.
        Returns:
            3-D Tensor color-distorted image on range [0, 1]
        Raises:
            ValueError: if color_ordering not in [0, 3]
        """
        with tf.name_scope(scope, 'distort_color', [image]):
            if fast_mode:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                else:
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                elif color_ordering == 1:
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                elif color_ordering == 2:
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                elif color_ordering == 3:
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                else:
                    raise ValueError('color_ordering must be in [0, 3]')
            # The random_* ops do not necessarily clamp.
            return tf.clip_by_value(image, 0.0, 1.0)

    # --------------------------------------------------------------
    #   NOTE rot image and bboxes
    # --------------------------------------------------------------
    @staticmethod
    def tf_rotate_image(image, xs, ys):
        image, bboxes, xs, ys = tf.py_func(Array_Image_Preprocessing().rotate_image, [image, xs, ys], 
                                            [tf.uint8, tf.float32, tf.float32, tf.float32])
        image.set_shape([None, None, 3])
        bboxes.set_shape([None, 4])
        xs.set_shape([None, 4])
        ys.set_shape([None, 4])
        return image, bboxes, xs, ys

    # --------------------------------------------------------------
    #   NOTE flip image
    # --------------------------------------------------------------
    @staticmethod
    def random_flip_left_right(image, bboxes, seed=None):
        """Random flip left-right of an image and its bounding boxes.
        """
        def flip_bboxes(bboxes):
            """Flip bounding boxes coordinates.
            """
            bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                              bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
            return bboxes

        def fix_image_flip_shape(image, result):
            """Set the shape to 3 dimensional if we don't know anything else.
            Args:
            image: original image size
            result: flipped or transformed image
            Returns:
            An image whose shape is at least None,None,None.
            """
            image_shape = image.get_shape()
            if image_shape == tensor_shape.unknown_shape():
                result.set_shape([None, None, None])
            else:
                result.set_shape(image_shape)
            return result

        # Random flip. Tensorflow implementation.
        with tf.name_scope('random_flip_left_right'):
            image = tf.convert_to_tensor(image, name='image')
            _Check3DImage(image, require_static=False)
            uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
            mirror_cond = tf.less(uniform_random, .5)
            # Flip image.
            result = tf.cond(mirror_cond,
                            lambda: tf.reverse(image, [1]),
                            lambda: image)
            # Flip bboxes.
            bboxes = tf.cond(mirror_cond,
                            lambda: flip_bboxes(bboxes),
                            lambda: bboxes)
            return fix_image_flip_shape(image, result), bboxes

    @staticmethod
    def distorted_bounding_box_crop(image,
                                    labels,
                                    bboxes,
                                    xs, ys, 
                                    min_object_covered=0.,
                                    aspect_ratio_range=[0.8, 1.2],
                                    area_range=[0.85, 1.0],
                                    max_attempts=400,
                                    scope=None):
        """Generates cropped_image using a one of the bboxes randomly distorted.

        See `tf.image.sample_distorted_bounding_box` for more documentation.

        Args:
            image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
            bbox: 2-D float Tensor of bounding boxes arranged [num_boxes, coords]
                where each coordinate is [0, 1) and the coordinates are arranged
                as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
                image.
            min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
                area of the image must contain at least this fraction of any bounding box
                supplied.
            aspect_ratio_range: An optional list of `floats`. The cropped area of the
                image must have an aspect ratio = width / height within this range.
            area_range: An optional list of `floats`. The cropped area of the image
                must contain a fraction of the supplied image within in this range.
            max_attempts: An optional `int`. Number of attempts at generating a cropped
                region of the image of the specified constraints. After `max_attempts`
                failures, return the entire image.
            scope: Optional scope for name_scope.
        Returns:
            A tuple, a 3-D Tensor cropped_image and the distorted bbox
        """
        with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes, xs, ys]):
            # Each bounding box has shape [1, num_boxes, box coords] and
            # the coordinates are ordered [ymin, xmin, ymax, xmax].
            num_bboxes = tf.shape(bboxes)[0]
            def has_bboxes():
                return bboxes, labels, xs, ys
            def no_bboxes():
                xmin = tf.random_uniform((1,1), minval = 0, maxval = 0.9)
                ymin = tf.random_uniform((1,1), minval = 0, maxval = 0.9)
                w = tf.constant(0.1, dtype = tf.float32)
                h = w
                xmax = xmin + w
                ymax = ymin + h
                rnd_bboxes = tf.concat([ymin, xmin, ymax, xmax], axis = 1)
                rnd_labels = tf.constant([0], dtype = tf.int64)
                rnd_xs = tf.concat([xmin, xmax, xmax, xmin], axis = 1)
                rnd_ys = tf.concat([ymin, ymin, ymax, ymax], axis = 1)
                
                return rnd_bboxes, rnd_labels, rnd_xs, rnd_ys
            
            bboxes, labels, xs, ys = tf.cond(num_bboxes > 0, has_bboxes, no_bboxes)
            # NOTE confirm bboxes is in [0, 1]
            bboxes = tf.clip_by_value(bboxes, 0, 1)
            # tf.image.sample_distorted_bounding_box返回的3个tensor表示一个bbox：
            # 前两个分别是它的左上角坐标和宽高，可直接用于裁剪原图；
            # 最后一个用坐标表示这个bbox, shape=[1, 1, 4]， 在这里用于当作reference来调整原图的bbox。
            bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                                                            tf.shape(image),
                                                            bounding_boxes=tf.expand_dims(bboxes, 0),
                                                            min_object_covered=min_object_covered,
                                                            aspect_ratio_range=aspect_ratio_range,
                                                            area_range=area_range,
                                                            max_attempts=max_attempts,
                                                            use_image_if_no_bounding_boxes=True)
            distort_bbox = distort_bbox[0, 0]
            # Crop the image to the specified bounding box.
            cropped_image = tf.slice(image, bbox_begin, bbox_size)
            # Restore the shape since the dynamic slice loses 3rd dimension.
            cropped_image.set_shape([None, None, 3])

            # Update bounding boxes: resize and filter out.
            bboxes, xs, ys = bbox.bboxes_resize(distort_bbox, bboxes, xs, ys)
            # labels, bboxes, xs, ys = bbox.bboxes_filter_overlap(labels, bboxes, xs, ys, 
            #             threshold=0.1, assign_value = LABEL_IGNORE)
            return cropped_image, labels, bboxes, xs, ys, distort_bbox

    @staticmethod
    def resize_image_bboxes_with_crop_or_pad(image, bboxes, xs, ys,
                                            target_height, target_width):
        """Crops and/or pads an image to a target width and height.
        Resizes an image to a target width and height by either centrally
        cropping the image or padding it evenly with zeros.

        If `width` or `height` is greater than the specified `target_width` or
        `target_height` respectively, this op centrally crops along that dimension.
        If `width` or `height` is smaller than the specified `target_width` or
        `target_height` respectively, this op centrally pads with 0 along that
        dimension.
        Args:
            image: 3-D tensor of shape `[height, width, channels]`
            target_height: Target height.
            target_width: Target width.
            Raises:
            ValueError: if `target_height` or `target_width` are zero or negative.
        Returns:
            Cropped and/or padded image of shape
                `[target_height, target_width, channels]`
        """
        def bboxes_crop_or_pad(bboxes, xs, ys,
                               height, width,
                               offset_y, offset_x,
                               target_height, target_width):
            """Adapt bounding boxes to crop or pad operations.
            Coordinates are always supposed to be relative to the image.

            Arguments:
            bboxes: Tensor Nx4 with bboxes coordinates [y_min, x_min, y_max, x_max];
            height, width: Original image dimension;
            offset_y, offset_x: Offset to apply,
                negative if cropping, positive if padding;
            target_height, target_width: Target dimension after cropping / padding.
            """
            with tf.name_scope('bboxes_crop_or_pad'):
                # Rescale bounding boxes in pixels.
                scale = tf.cast(tf.stack([height, width, height, width]), bboxes.dtype)
                bboxes = bboxes * scale
                xs *= tf.cast(width, bboxes.dtype)
                ys *= tf.cast(height, bboxes.dtype)
                # Add offset.
                offset = tf.cast(tf.stack([offset_y, offset_x, offset_y, offset_x]), bboxes.dtype)
                bboxes = bboxes + offset
                xs += tf.cast(offset_x, bboxes.dtype)
                ys += tf.cast(offset_y, bboxes.dtype)
                
                # Rescale to target dimension.
                scale = tf.cast(tf.stack([target_height, target_width,
                                          target_height, target_width]), bboxes.dtype)
                bboxes = bboxes / scale
                xs = xs / tf.cast(target_width, xs.dtype)
                ys = ys / tf.cast(target_height, ys.dtype)
                return bboxes, xs, ys

        with tf.name_scope('resize_with_crop_or_pad'):
            image = tf.convert_to_tensor(image, name='image')

            assert_ops = []
            assert_ops += _Check3DImage(image, require_static=False)
            assert_ops += _assert(target_width > 0, ValueError,
                                'target_width must be > 0.')
            assert_ops += _assert(target_height > 0, ValueError,
                                'target_height must be > 0.')

            image = control_flow_ops.with_dependencies(assert_ops, image)
            # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
            # Make sure our checks come first, so that error messages are clearer.
            if _is_tensor(target_height):
                target_height = control_flow_ops.with_dependencies(assert_ops, target_height)
            if _is_tensor(target_width):
                target_width = control_flow_ops.with_dependencies(assert_ops, target_width)

            def max_(x, y):
                if _is_tensor(x) or _is_tensor(y):
                    return tf.maximum(x, y)
                else:
                    return max(x, y)

            def min_(x, y):
                if _is_tensor(x) or _is_tensor(y):
                    return tf.minimum(x, y)
                else:
                    return min(x, y)

            def equal_(x, y):
                if _is_tensor(x) or _is_tensor(y):
                    return tf.equal(x, y)
                else:
                    return x == y

            height, width, _ = _ImageDimensions(image)
            width_diff = target_width - width
            offset_crop_width = max_(-width_diff // 2, 0)
            offset_pad_width = max_(width_diff // 2, 0)

            height_diff = target_height - height
            offset_crop_height = max_(-height_diff // 2, 0)
            offset_pad_height = max_(height_diff // 2, 0)

            # Maybe crop if needed.
            height_crop = min_(target_height, height)
            width_crop = min_(target_width, width)
            cropped = tf.image.crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                                    height_crop, width_crop)
            bboxes, xs, ys = bboxes_crop_or_pad(bboxes, xs, ys,
                                                height, width,
                                                -offset_crop_height, -offset_crop_width,
                                                height_crop, width_crop)
            # Maybe pad if needed.
            resized = tf.image.pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                                   target_height, target_width)
            bboxes, xs, ys = bboxes_crop_or_pad(bboxes, xs, ys,
                                                height_crop, width_crop,
                                                offset_pad_height, offset_pad_width,
                                                target_height, target_width)

            # In theory all the checks below are redundant.
            if resized.get_shape().ndims is None:
                raise ValueError('resized contains no shape.')

            resized_height, resized_width, _ = _ImageDimensions(resized)

            assert_ops = []
            assert_ops += _assert(equal_(resized_height, target_height), ValueError,
                                'resized height is not correct.')
            assert_ops += _assert(equal_(resized_width, target_width), ValueError,
                                'resized width is not correct.')

            resized = control_flow_ops.with_dependencies(assert_ops, resized)
            return resized, bboxes, xs, ys


    def random_rotate90(self, image, bboxes, xs, ys):
        with tf.name_scope('random_rotate90'):
            k = tf.random_uniform([], 0, 10000)
            k = tf.cast(k, tf.int32)
            
            image_shape = tf.shape(image)
            h, w = image_shape[0], image_shape[1]
            image = tf.image.rot90(image, k = k)
            bboxes, xs, ys = self.rotate90(bboxes, xs, ys, k)
            return image, bboxes, xs, ys

    @staticmethod
    def tf_rotate_point_by_90(x, y, k):
        return tf.py_func(img.rotate_point_by_90, [x, y, k], 
                        [tf.float32, tf.float32])
        
    def rotate90(self, bboxes, xs, ys, k):
        
        ymin, xmin, ymax, xmax = [bboxes[:, i] for i in range(4)]
        xmin, ymin = self.tf_rotate_point_by_90(xmin, ymin, k)
        xmax, ymax = self.tf_rotate_point_by_90(xmax, ymax, k)
        
        new_xmin = tf.minimum(xmin, xmax)
        new_xmax = tf.maximum(xmin, xmax)
        
        new_ymin = tf.minimum(ymin, ymax)
        new_ymax = tf.maximum(ymin, ymax)
        
        bboxes = tf.stack([new_ymin, new_xmin, new_ymax, new_xmax])
        bboxes = tf.transpose(bboxes)

        xs, ys = self.tf_rotate_point_by_90(xs, ys, k)
        return bboxes, xs, ys



def preprocess_for_train(image, labels, bboxes, xs, ys,
                         out_shape, data_format='NHWC',
                         scope='ssd_preprocessing_train'):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing.

    Returns:
        A preprocessed image.
    """
    fast_mode = False
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        
        # =======================NOTE method 1:rot
        if USE_ROT_90:
            rnd = tf.random_uniform((), minval = 0, maxval = 1)
            def rotate():
                return Tensor_Image_Preprocessing().random_rotate90(image, bboxes, xs, ys)
            def no_rotate():
                return image, bboxes, xs, ys
            image, bboxes, xs, ys = tf.cond(tf.less(rnd, 0.5), rotate, no_rotate)
        if USE_ROT_RANDOM:
            rnd = tf.random_uniform((), minval = 0, maxval = 1)
            def rot_random():
                return Tensor_Image_Preprocessing().tf_rotate_image(image, xs, ys)
            def no_rot_random():
                return image, bboxes, xs, ys
            image, bboxes, xs, ys = tf.cond(tf.less(rnd, 0.5), rot_random, no_rot_random)
        
        # =======================NOTE method 2:resize
        if MAX_EXPAND_SCALE > 1:
            rnd2 = tf.random_uniform((), minval = 0, maxval = 1)
            def expand():
                scale = tf.random_uniform([], minval = 1, maxval = MAX_EXPAND_SCALE, dtype=tf.float32)
                image_shape = tf.cast(tf.shape(image), dtype = tf.float32)
                image_h, image_w = image_shape[0], image_shape[1]
                target_h = tf.cast(image_h * scale, dtype = tf.int32)
                target_w = tf.cast(image_w * scale, dtype = tf.int32)
                return Tensor_Image_Preprocessing().resize_image_bboxes_with_crop_or_pad(
                             image, bboxes, xs, ys, target_h, target_w)
            def no_expand():
                return image, bboxes, xs, ys
            image, bboxes, xs, ys = tf.cond(tf.less(rnd2, 0.5), expand, no_expand)

        # =======================NOTE method 3
        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # =======================NOTE method 4:crop
        # Distort image and bounding boxes.
        dst_image, labels, bboxes, xs, ys, distort_bbox = Tensor_Image_Preprocessing().distorted_bounding_box_crop(
                                                            image, labels, bboxes, xs, ys,
                                                            aspect_ratio_range=[0.8, 1.2],
                                                            area_range=AREA_RANGE)
        # Resize image to output size.
        dst_image = resize_image(dst_image, out_shape,
                                method=tf.image.ResizeMethod.BILINEAR,
                                align_corners=False)
        
        # =======================NOTE method 5
        # Filter bboxes using the length of shorter sides
        if USING_SHORTER_SIDE_FILTERING:
             xs = xs * out_shape[1]
             ys = ys * out_shape[0]
             labels, bboxes, xs, ys = bbox.bboxes_filter_by_shorter_side(labels, 
                                                                        bboxes, xs, ys, 
                                                                        min_lenght_of_side = MIN_LENGHT_OF_SIDE, 
                                                                        max_lenght_of_side = np.infty, 
                                                                        assign_value = LABEL_IGNORE)
             xs = xs / out_shape[1]
             ys = ys / out_shape[0]
             
        # =======================NOTE method 6:distort color
        # Randomly distort the colors. There are 4 ways to do it.
        dst_image = Tensor_Image_Preprocessing().apply_with_random_selector(dst_image,
                        lambda x, ordering: Tensor_Image_Preprocessing().distort_color(x, ordering, fast_mode),
                        num_cases=4)

        # =======================NOTE method 7:whiten
        # Rescale to VGG input scale.
        image = dst_image * 255.
        image = Tensor_Image_Preprocessing().tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes, xs, ys


def preprocess_for_eval(image, labels, bboxes, xs, ys,
                        out_shape, data_format='NHWC',
                        resize=Resize.WARP_RESIZE,
                        do_resize = True,
                        scope='ssd_preprocessing_train'):
    """Preprocess an image for evaluation.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        out_shape: Output shape after pre-processing (if resize != None)
        resize: Resize strategy.

    Returns:
        A preprocessed image.
    """
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = tf.to_float(image)
        image = Tensor_Image_Preprocessing().tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        
        if do_resize:
            if resize == Resize.NONE:
                pass
            else:
                image = resize_image(image, out_shape,
                                    method=tf.image.ResizeMethod.BILINEAR,
                                    align_corners=False)

        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes, xs, ys


def preprocess_image(image,
                     labels = None,
                     bboxes = None,
                     xs = None, ys = None,
                     out_shape = None,
                     data_format = 'NHWC',
                     is_training=False,
                     **kwargs):
    """Pre-process an given image.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        is_training: `True` if we're preprocessing the image for training and
            `False` otherwise.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing. If `is_training` is `False`, then this value
            is used for rescaling.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing. If `is_training` is `False`, this value is
            ignored. Otherwise, the resize side is sampled from
            [resize_size_min, resize_size_max].
    Returns:
        A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, labels, bboxes, xs, ys,
                                    out_shape=out_shape,
                                    data_format=data_format)
    else:
        return preprocess_for_eval(image, labels, bboxes, xs, ys,
                                   out_shape=out_shape,
                                   data_format=data_format,
                                   **kwargs)


