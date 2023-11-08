
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.OUTPUT_SUB_DIR = ''
_C.LOG_DIR = ''
# _C.GPUS = (3,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
# _C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.EXTRA = CN(new_allowed=True)

# _C.LOSS = CN()
# _C.LOSS.USE_OHEM = False
# _C.LOSS.OHEMTHRES = 0.9
# _C.LOSS.OHEMKEEP = 100000
# _C.LOSS.CLASS_BALANCE = True

# DATASET related params
_C.DATASET = CN()
_C.DATASET.DATASET = 'pituitary'
_C.DATASET.ROOT = ''
_C.DATASET.CSV_FILE_ROOT = ''
_C.DATASET.NUM_CLASSES = 3
_C.DATASET.TRAIN_SET = 'image_centroid_fold1_train.csv'
# _C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'image_centroid_fold1_val.csv'
_C.DATASET.IMAGE_ROOT= '/workspace/zhmao/data/my_dataset/JPEGImages'
_C.DATASET.MASK_ROOT= '/workspace/zhmao/data/my_dataset/SegmentationClass_bak'

# training
_C.TRAIN = CN()

_C.TRAIN.IMAGE_SIZE = [1280, 736]  # width * height
# _C.TRAIN.BASE_SIZE = 2048
# _C.TRAIN.DOWNSAMPLERATE = 1
# _C.TRAIN.FLIP = True
# _C.TRAIN.MULTI_SCALE = True
# _C.TRAIN.SCALE_FACTOR = 16
_C.TRAIN.LOSS_WEIGHT = [1.0, 0.8, 0.2]

_C.TRAIN.LR_FACTOR = 0.1
# _C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
# _C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.STAGE1_EPOCH = 300
# _C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
# only using some training samples
# _C.TRAIN.NUM_SAMPLES = 0

# testing
_C.TEST = CN()

_C.TEST.IMAGE_SIZE = [1280, 736]  # width * height
# _C.TEST.BASE_SIZE = 2048

_C.TEST.BATCH_SIZE_PER_GPU = 32
# only testing some samples
# _C.TEST.NUM_SAMPLES = 0

_C.TEST.MODEL_FILE = ''
# _C.TEST.FLIP_TEST = False
# _C.TEST.MULTI_SCALE = False
# _C.TEST.CENTER_CROP_TEST = False
# _C.TEST.SCALE_LIST = [1]

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

