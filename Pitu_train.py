# ------------------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'  #GPU id

import argparse
import os
import pprint

import logging
import time
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim
from tensorboardX import SummaryWriter

from lib.config import config
from lib.config import update_config
from lib.core.function import train, validate
from lib.core.bdl_losses import GeneralizedDice, SurfaceLoss, DiceLoss
from lib.utils.utils import create_logger
from lib.datasets.pituitary import PitDataset
from lib.models.seg_hrnet import HighResolutionNet
import random
from lib.core import mmwing_loss, focal_loss
import torch.optim as optim
from itertools import chain


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
                        default=r'/home/zhehua/codes/Pituitary-Segment-Centroid/experiments/pituitary/seg_hrnet_w48_train_736x1280_sgd_lr1e-2_bs_6_4loss_2stage_v4_fold1.yaml',
                        help='experiment configure file name',
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

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    model = HighResolutionNet(config)
    model.init_weights(config.MODEL.PRETRAINED)              
    model = model.to(device)
    model = torch.nn.DataParallel(model)


    train_dataset = PitDataset(config, is_train=True)
    test_dataset = PitDataset(config, is_train=False)
       
    # Resample
    if config.TRAIN.RESAMPLE:
        target = torch.tensor([torch.sum(cpts_presence[:, 0])
                            for _, _, _, cpts_presence, _, _  in train_dataset])
        class_count = np.bincount(target)
        class_weights = 1./torch.tensor(class_count, dtype=float)
        weights = class_weights[target.long()]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True)

        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=sampler)
    else:
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=None)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=None)

    Seg_loss = DiceLoss(idc=[0, 1, 2])
    Seg_loss2 = SurfaceLoss(idc=[1, 2])
    Landmark_loss = mmwing_loss.WingLoss()
    # the focal loss is implemented based on BCEWithLogitsLoss, input should be logits
    Landmark_loss2 = focal_loss.FocalLoss()
    loss_weight=torch.tensor(config.TRAIN.LOSS_WEIGHT).to(device)

    
    for param in model.module.output1.parameters():
        param.requires_grad = False
    for param in model.module.detector1.parameters():
        param.requires_grad = False
    for param in model.module.output2.parameters():
        param.requires_grad = False
    for param in model.module.detector2.parameters():
        param.requires_grad = False

    shared_params = []
    head1_params = list(chain(model.module.last_layer.parameters()))
    head2_params = list(chain(model.module.output1.parameters(), model.module.detector1.parameters()))
    head3_params = list(chain(model.module.output2.parameters(), model.module.detector2.parameters()))
    
    head1_param_ids = set([id(param) for param in head1_params])
    head2_param_ids = set([id(param) for param in head2_params])
    head3_param_ids = set([id(param) for param in head3_params])
    
    for param in model.module.parameters():
        if id(param) not in head1_param_ids and id(param) not in head2_param_ids and id(param) not in head3_param_ids:
            shared_params.append(param)

    optimizer = optim.SGD([
        {'params': shared_params, 'lr': config.TRAIN.LR},
        {'params': head1_params, 'lr': config.TRAIN.LR},
        {'params': head2_params, 'lr': 0.0},
        {'params': head3_params, 'lr': 0.0}], 
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WD,
        nesterov=config.TRAIN.NESTEROV)

    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    epoch_iters = np.int32(train_dataset.__len__() /
                           config.TRAIN.BATCH_SIZE_PER_GPU)
    train_best_mIoU = 0
    best_mIoU = 0
    train_best_mpck20 = float('-inf')
    best_mpck20 = float('-inf')
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            train_best_mIoU = checkpoint['train_best_mIoU']
            train_best_mpck20 = checkpoint['train_best_mpck20']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        if epoch < config.TRAIN.STAGE1_EPOCH:
            stage = 1
            train_total_loss, train_mIoU, train_IoU, train_accuracy, train_recall, train_precision, train_mdistance, train_mpck20,\
                 = train(config, epoch, config.TRAIN.END_EPOCH, epoch_iters, config.TRAIN.LR, num_iters,trainloader, optimizer, model, 
                        Seg_loss, Seg_loss2, Landmark_loss, Landmark_loss2, writer_dict, device, stage, loss_weight)
        else:
            stage = 2
            if epoch == config.TRAIN.STAGE1_EPOCH:
                train_best_mIoU =0
                best_mIoU = 0
                for param in model.module.output1.parameters():
                    param.requires_grad = True
                for param in model.module.detector1.parameters():
                    param.requires_grad = True 
            train_total_loss, train_mIoU, train_IoU, train_accuracy, train_recall, train_precision, train_mdistance, train_mpck20,\
                    = train(config, epoch, config.TRAIN.END_EPOCH, epoch_iters, config.TRAIN.LR, num_iters,trainloader, optimizer, model, 
                            Seg_loss, Seg_loss2, Landmark_loss, Landmark_loss2, writer_dict, device, stage, loss_weight)

        valid_loss, mIoU, IoU_array, accuracy, recall, precision, valid_mDistance, mpck20 = validate(
                config, testloader, model, 
                Seg_loss, Seg_loss2, Landmark_loss, Landmark_loss2, writer_dict, device, stage, loss_weight)

        if args.local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'train_best_mIoU': train_best_mIoU,
                'train_best_mpck20': train_best_mpck20,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

            if train_mpck20 > train_best_mpck20:
                train_best_mpck20 = train_mpck20
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'train_best_mpck20.pth'))

            if train_mIoU > train_best_mIoU:
                train_best_mIoU = train_mIoU
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'train_best_mIoU.pth'))
            msg = 'Train==> Loss:{:.3f}, mIoU:{: 4.4f}, Acc:{: 4.4f}, mRecall:{: 4.4f}, mPrecision:{: 4.4f}, Best_mIoU:{: 4.4f},'\
                'mDistance:{: 4.4f}, Best_MPCK20:{: 4.4f}'.format(
                    train_total_loss, train_mIoU, train_accuracy, train_recall.mean(
                    ), train_precision.mean(), train_best_mIoU, train_mdistance, train_best_mpck20)

            metric = 'Train_Metric==> IoU:{}, Recall:{}, Precision: {}, MPCK20:{: 4.4f}'.format(
                train_IoU, train_recall, train_precision, train_mpck20)
            logging.info(msg)
            logging.info(metric)

            if mpck20 > best_mpck20:
                best_mpck20 = mpck20
            if mIoU > best_mIoU:
                best_mIoU = mIoU
            msg = 'Val==> Loss:{:.3f}, mIoU:{: 4.4f}, Acc:{: 4.4f}, mRecall:{: 4.4f}, mPrecision:{: 4.4f}, Best_mIoU:{: 4.4f},'\
                ' mDistance:{: 4.4f}, Best_MPCK20:{: 4.4f}'.format(
                    valid_loss, mIoU, accuracy, recall.mean(), precision.mean(), best_mIoU, valid_mDistance, best_mpck20)

            metric = 'Val_Metric==> IoU_array:{}, Recall:{}, Precision:{}, MPCK20:{: 4.4f}'.format(
                IoU_array, recall, precision, mpck20)
            logging.info(msg)
            logging.info(metric)

            if epoch == end_epoch - 1:
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'final_state.pth'))

                writer_dict['writer'].close()
                end = timeit.default_timer()
                logger.info('Hours: %d' % np.int32((end-start)/3600))
                logger.info('Done')


if __name__ == '__main__':
    main()
