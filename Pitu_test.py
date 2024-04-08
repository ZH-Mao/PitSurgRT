# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import os
import random
from lib.models.seg_hrnet import HighResolutionNet
from lib.datasets.pituitary import PitDataset
from lib.utils.utils import create_logger
from lib.core.function_v4 import test
from lib.config import update_config
from lib.config import config
import segmentation_models_pytorch as smp
import torch.optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import timeit
import time
import logging
import pprint
import argparse

from lib.utils.modelsummary import get_model_summary

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)
    parser.add_argument('--cfg',
                        default=r'/workspace/zhmao/code-d/Pituitary-Segment-Centroid/experiments/pituitary/seg_hrnet_w48_train_736x1280_sgd_lr1e-2_bs_6_epoch350_4loss_2stage_v4_fold5.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('--model',
                        default=r'/workspace/zhmao/data/HRNet_with_DICE_BD_Wing_FL_2stage/pituitary/seg_hrnet_w48_train_736x1280_sgd_lr1e-2_bs_6_epoch350_4loss_2stage_v4_fold5/train_best_mIoU.pth',
                        help='trained model file',
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    model = HighResolutionNet(config)
    model.init_weights(config.MODEL.PRETRAINED)

    if args.model:
        trained_dict = torch.load(args.model)
        logger.info('=> loading trained model {}'.format(args.model))
        model_dict = model.state_dict()
        trained_dict = {k: v for k, v in trained_dict.items()
                        if k in model_dict.keys()}
        model_dict.update(trained_dict)
        model.load_state_dict(model_dict)
    else:
        print('No trained model is found, you are using pretrained model')

    model = model.to(device)
    model = torch.nn.DataParallel(model)


    # prepare data
    test_dataset = PitDataset(config, is_train=False)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    start = timeit.default_timer()

    test(config, testloader, model, sv_dir=final_output_dir, device=device)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int32((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
