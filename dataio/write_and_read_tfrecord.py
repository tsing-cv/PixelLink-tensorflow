# -*- coding:utf-8 -*-
"""
Author: tsing
Time: 2018-12-21
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf 
slim = tf.contrib.slim
import numpy as np 
import os 
import sys
from tqdm import tqdm
import io
import cv2
import scipy.misc as scm
import matplotlib.pyplot as plt
sys.path.append("../")
# print (os.getcwd())
# sys.path.append(r'C:\Users\tsing\Desktop\reimplement')
from config import cfgs
from libs.tools import tools
from libs.encode_decode import encode_decode
from libs.preprocessing import ssd_vgg_preprocessing

class Write_Tfrecord():
    def __init__(self, dataset_dir, tfrecord_dir, is_train=True):
        self.dataset_dir = dataset_dir
        self.tfrecord_dir = tfrecord_dir
        self.is_train = is_train

    @staticmethod
    def int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def bytes_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def image_and_gtbox_label(self, image_name_with_format, img_dir, gt_dir):
        """Read the annotation of the image

        Args:
            Str image_name_with_format: image name without format
            List shape: [h, w]
        Returns:
            List gtbox_label: annotations of image
                [[x1,y1,x2,y2,x3,y3,x4,y4,cls1], [...]]
        """
        image_name,_ = os.path.splitext(image_name_with_format)
        image_data = tf.gfile.FastGFile(os.path.join(img_dir, image_name_with_format), 'rb').read()
        array_image = scm.imread(io.BytesIO(image_data))
        # scm.imshow(array_image)
        h, w = array_image.shape[: 2]
        gtbox_labels = []
        txt_path = os.path.join(gt_dir, "{}.txt".format(image_name)) 
        with open(txt_path, 'r', encoding='utf-8') as annot:
            for line in annot:
                line = line.strip().split(',')
                box,label = list(map(float, line[: 8])), int(cfgs.label_num_map[line[8]])
                # assert np.array(box).any() >= 0
                box = np.array(box)/([w, h]* 4)
                box = np.concatenate([box, [label]])
                gtbox_labels.append(box)

        return image_name, image_data, gtbox_labels

    @staticmethod
    def get_list(obj, idx):
        obj = np.asarray(obj)
        if len(obj) > 0:
            # print (list(obj[:, idx]))
            return list(obj[:, idx])
        return []

    def convert_to_tfexample(self, image_data, image_name, image_format, gtbox_label):
        if len(gtbox_label) == 0:
            print (image_name, "has no gtbox_label")
        example = tf.train.Example(features = tf.train.Features(feature={
                    "image/filename": self.bytes_feature(image_name),
                    "image/encode": self.bytes_feature(image_data),
                    "image/format": self.bytes_feature(image_format),
                    "label/x1": self.float_feature(self.get_list(gtbox_label, 0)),
                    "label/y1": self.float_feature(self.get_list(gtbox_label, 1)),
                    "label/x2": self.float_feature(self.get_list(gtbox_label, 2)),
                    "label/y2": self.float_feature(self.get_list(gtbox_label, 3)),
                    "label/x3": self.float_feature(self.get_list(gtbox_label, 4)),
                    "label/y3": self.float_feature(self.get_list(gtbox_label, 5)),
                    "label/x4": self.float_feature(self.get_list(gtbox_label, 6)),
                    "label/y4": self.float_feature(self.get_list(gtbox_label, 7)),
                    "label/cls": self.int64_feature(list(map(int, self.get_list(gtbox_label, 8)))),
                    }))
        return example

    def convert_to_tfrecord(self):
        img_dir = os.path.join(self.dataset_dir, 'JPEG')
        assert tf.gfile.Exists(img_dir), "Image folder {} are not exits;Or your images' folder name is not JPEG".format(img_dir)
        gt_dir = os.path.join(self.dataset_dir, 'ANNOT')
        assert tf.gfile.Exists(gt_dir), "Annotations folder {} are not exits;Or your annotations' folder name is not ANNOT".format(gt_dir)
        
        images = tf.gfile.ListDirectory(img_dir)
        tfrecord_path = tools.make_folder(self.tfrecord_dir)
        if self.is_train:
            tfrecord_path = os.path.join(os.path.abspath(self.tfrecord_dir), "{}_{}.tfrecord".format(cfgs.Dataset_name, 'train'))
        else:
            tfrecord_path = os.path.join(os.path.abspath(self.tfrecord_dir), "{}_{}.tfrecord".format(cfgs.Dataset_name, 'test'))
        print ('{}\nTfrecord Saving Path:\n\t{}\n{}'.format("@@"*30, tfrecord_path, "@@"*30))
        
        # with tf.io.TFRecordWriter(tfrecord_path) as tfrecord_writer: # NOTE tf1.12+
        with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:
            pbar = tqdm(images, total=len(images), ncols=80)
            for image_name_with_format in pbar:
                
                image_name, image_data, gtbox_label = self.image_and_gtbox_label(image_name_with_format, img_dir, gt_dir)
                pbar.set_description(image_name)
                # cv2.imshow('1', array_image)
                # cv2.waitKey(100000)
                # print (np.shape(gtbox_label))
                example = self.convert_to_tfexample(image_name = image_name.encode(), 
                                                    image_data = image_data,
                                                    image_format = 'jpg'.encode(), 
                                                    gtbox_label = gtbox_label, 
                                                    )
                tfrecord_writer.write(example.SerializeToString())

class Read_Tfrecord():
    def __init__(self, tfrecord_dir, batch_size, is_train=True):
        self.tfrecord_dir = os.path.abspath(tfrecord_dir)
        self.batch_size = batch_size
        self.is_train = is_train
    

    def __get_filenames(self):
        if self.is_train:
            return os.path.join(self.tfrecord_dir, "{}_{}.tfrecord".format(cfgs.Dataset_name, 'train'))
        else:
            return os.path.join(self.tfrecord_dir, "{}_{}.tfrecord".format(cfgs.Dataset_name, 'test'))

    def get_dataset(self, dataset_dir, reader=None):
        # ********* step 1 *************************
        # 将example反序列化成存储之前的格式
        keys_to_features = {
            'image/encode': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
            'label/x1': tf.VarLenFeature(dtype=tf.float32),
            'label/x2': tf.VarLenFeature(dtype=tf.float32),
            'label/x3': tf.VarLenFeature(dtype=tf.float32),
            'label/x4': tf.VarLenFeature(dtype=tf.float32),
            'label/y1': tf.VarLenFeature(dtype=tf.float32),
            'label/y2': tf.VarLenFeature(dtype=tf.float32),
            'label/y3': tf.VarLenFeature(dtype=tf.float32),
            'label/y4': tf.VarLenFeature(dtype=tf.float32),
            'label/cls': tf.VarLenFeature(dtype=tf.int64),
            }
        # ********* step 2 *************************
        # 将反序列化的数据组装成更高级的格式。
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encode', 'image/format'),
            'filename': slim.tfexample_decoder.Tensor('image/filename'),
            'x1': slim.tfexample_decoder.Tensor('label/x1'),
            'x2': slim.tfexample_decoder.Tensor('label/x2'),
            'x3': slim.tfexample_decoder.Tensor('label/x3'),
            'x4': slim.tfexample_decoder.Tensor('label/x4'),
            'y1': slim.tfexample_decoder.Tensor('label/y1'),
            'y2': slim.tfexample_decoder.Tensor('label/y2'),
            'y3': slim.tfexample_decoder.Tensor('label/y3'),
            'y4': slim.tfexample_decoder.Tensor('label/y4'),
            'cls': slim.tfexample_decoder.Tensor('label/cls')
            }
        # ********* step 3 *************************
        # 解码器，进行解码
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        labels_to_names = {v:k for k,v in cfgs.label_num_map.items()}
        items_to_descriptions = {'image': 'A color image of varying height and width.',
                                 'label': 'A list of labels, one per each object. bbox and class',
                                }
        # ********* step 4 *************************
        # dataset对象定义了数据集的文件位置，解码方式等元信息
        return slim.dataset.Dataset(data_sources=dataset_dir,
                                    reader=tf.TFRecordReader,
                                    decoder=decoder,
                                    num_samples=cfgs.num_samples["train"],
                                    items_to_descriptions=items_to_descriptions,
                                    num_classes= len(cfgs.label_num_map),
                                    labels_to_names=labels_to_names)


    def create_batch_queue(self):
        print ("{}\n{}\n\t{}\n{}".format("@@@"*20, "Tfrecord Dir Is:", self.__get_filenames(), "@@@"*20))
        dataset = self.get_dataset(self.__get_filenames())

        with tf.device('/cpu:0'):
            # ********* step 5 *************************
            # 获取数据，获取到的数据是单个数据，还需要对数据进行预处理，组合数据
            with tf.name_scope(cfgs.Dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                                dataset,
                                num_readers=1,
                                common_queue_capacity=500 * cfgs.batch_size,
                                common_queue_min=100 * cfgs.batch_size,
                                shuffle=self.is_train)
            # Get for SSD network: image, labels, bboxes.
            [image, x1, x2, x3, x4, y1, y2, y3, y4, glabel] = provider.get(['image','x1','x2','x3','x4',
                                                                            'y1','y2','y3','y4','cls',
                                                                            ])
            #shape = (N, 4)
            gxs = tf.transpose(tf.stack([x1, x2, x3, x4])) 
            gys = tf.transpose(tf.stack([y1, y2, y3, y4]))

            xmin, ymin = tf.reduce_min(gxs, axis=1), tf.reduce_min(gys, axis=1)
            xmax, ymax = tf.reduce_max(gxs, axis=1), tf.reduce_max(gys, axis=1)
            gbboxes = tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))
            
            image = tf.identity(image, 'input_image')
            
            # ********* step 6 *************************
            # Pre-processing image, labels and bboxes.
            image, glabel, gbboxes, gxs, gys = ssd_vgg_preprocessing.preprocess_image(
                                                    image, glabel, gbboxes, gxs, gys, 
                                                    out_shape = [cfgs.train_image_shape["height"], cfgs.train_image_shape["width"]],
                                                    data_format = cfgs.data_format, 
                                                    is_training = self.is_train)

            image = tf.identity(image, 'processed_image')
            # calculate ground truth
            cls_label, cls_weight, link_label, link_weight, rects_with_labels = encode_decode.tf_encode_single_image(gxs, gys, glabel)


            # ********* step 7 *************************
            # batch them
            with tf.name_scope(cfgs.Dataset_name + '_batch'):
                # b_image, b_cls_label, b_cls_weight, b_link_label, b_link_weight, b_rects_with_labels = tf.train.batch(
                #                                 [image, cls_label, cls_weight, link_label, link_weight, rects_with_labels],
                #                                 batch_size = self.batch_size,
                #                                 num_threads= 8,
                #                                 dynamic_pad = True,
                #                                 capacity = 10*cfgs.batch_size)
                batch_queue = tf.train.batch([image, cls_label, cls_weight, link_label, link_weight, rects_with_labels],
                                              batch_size = self.batch_size,
                                              num_threads= 8,
                                              dynamic_pad = True,
                                              capacity = 10*cfgs.batch_size)
            
            # ********* step 8 *************************
            # NOTE alternative: but when lenth is not determinated, we cannot use prefetch_queue
            # when use queue, get data must use batch_queue.dequeue()
            # with tf.name_scope(cfgs.Dataset_name + '_prefetch_queue'):
            #     batch_queue = slim.prefetch_queue.prefetch_queue([b_image, 
            #                                                       b_cls_label, 
            #                                                       b_cls_weight, 
            #                                                       b_link_label, 
            #                                                       b_link_weight
            #                                                       ],
            #                                                       capacity = 3*cfgs.batch_size) 

        return batch_queue  



if __name__ == "__main__":
    # Write_Tfrecord(cfgs.DATASET, cfgs.TFRECORD).convert_to_tfrecord()
    batch_queue = Read_Tfrecord(cfgs.TFRECORD, cfgs.batch_size).create_batch_queue()
    # print (batch_queue.dequeue())
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        i = 0
        try:
            while not coord.should_stop() and i < 200:
                b_image, b_cls_label, b_cls_weight, b_link_label, b_link_weight, br = sess.run(batch_queue)
                print ("{}\nClass And Label:\n\tcls label:{}\n\tcls weight{}\n\tlink label{}\n\tlink weight{}\n{}".format(
                    '==='*20,  b_cls_label.shape, b_cls_weight.shape, b_link_label.shape, b_link_weight.shape, '==='*20))
                img = tools.draw_box_cv(b_image[0], np.hstack((br[0,:,:-2], np.reshape(br[0,:,-2], [-1, 1]))), br[0,:,-1])
                plt.figure(figsize=(24, 18))
                plt.subplot(241)
                plt.imshow(b_image[0,:,:,0])
                plt.title("image")
                plt.subplot(242)
                plt.imshow(b_cls_label[0,:,:,0]*120)
                plt.title("cls_label")
                plt.subplot(243)
                print ("summary:",np.max(b_cls_weight[0,:,:,0]))
                plt.imshow(b_cls_weight[0,:,:,0])
                plt.title("cls_weight")
                plt.subplot(244)
                plt.imshow(b_link_label[0][:, :, 0]*255)
                plt.title("link_label 0")
                plt.subplot(245)
                plt.imshow(b_link_label[0][:, :, 1]*255)
                plt.title("link_label 1")
                plt.subplot(246)
                plt.imshow(b_link_weight[0][:, :, 0]*255)
                plt.title("img")
                plt.imshow(img)
                plt.title("img")
                plt.show()
                # print (br)

                i += 1
        except tf.errors.OutOfRangeError:
            print ('done')
        finally:
            coord.request_stop()
        coord.join(threads)    



