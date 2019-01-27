# NOTE version ===============================================
Time = 20190110
Dataset_name = 'ICDAR2015'
# NOTE dataset ===============================================
DATASET = '../dataset'
TFRECORD = '../dataset/TFRECORD'

label_num_map = {
    "background_label": 0,
    "text_label": 1
    }

# NOTE train   ===============================================
using_moving_average = True
learning_rate = 1.0
moving_average_decay = 0.999
max_number_of_steps = 300000
gpu_memory_fraction = -1
batch_size = 2
num_preprocessing_threads = 8
num_epochs = 30000
gpus = (0,)

# NOTE test    ===============================================
test_image_dir = '../demo/images'
test_result_dir = '../demo/results'

# NOTE image   ===============================================
num_samples = {
    "train": 2000,
    "valid": 0,
    }
train_image_shape = {
    "height": 640, 
    "width": 640
    }

strides = [1] #[2]
score_map_shape = (int(train_image_shape["height"] / strides[0]), int(train_image_shape["width"] / strides[0]))

cls_conf_threshold = 0.7
link_conf_threshold = 0.8
decode_image_by_join = True
data_format = 'NHWC'
rgb_mean = [123., 117., 104.]

# NOTE output   ===============================================
checkpoint_path = "../output/{}{}/".format(Dataset_name, Time)

# NOTE encode_decode ==========================================
min_area = 225
min_lenght_of_side = 4
weight_threshhold = 200
min_length_ratio = 0.007
crop_area_range = [0.7, 1.0]
max_expand_scale = 1.1
cls_weight_with_border_balanced = True
cls_border_weight_lambda = 0.002
bbox_border_width = 1
use_mult_level_mask = False
use_mult_class_mask = False
minimum_mask_remained = 20

# NOTE network ================================================
fuse_method = "concat128"
dropout_ratio = 0
feat_layers = ['conv2_2', 'conv3_3', 'conv4_3', 'conv5_3', 'fc7']
max_neg_pos_ratio = 3
num_neighbours = 4
within_dilation = True
use_link = False
OHNM_weight_lambda = 1

freeze_batch_norm = True

