# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
# import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from ..utils.utils import AverageMeter
from ..utils.utils import get_confusion_matrix
from ..utils.utils import adjust_learning_rate
from ..utils.utils import get_world_size, get_rank
# import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw
import cv2


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,trainloader, optimizer, model, Seg_loss, Seg_loss2,
          Landmark_loss, Landmark_loss2, writer_dict, device, stage, loss_weight, scheduler=None):

    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_seg_loss = AverageMeter()
    ave_bound_loss = AverageMeter()
    ave_Wing_loss = AverageMeter()
    ave_FL_loss = AverageMeter()
    
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    # global_steps is the number of epoches rather than steps
    global_steps = writer_dict['train_global_steps']
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    total_num_inThresh = 0
    total_num_points = 0
    total_distance = 0
    total_num_Present= 0 

    for i_iter, batch in enumerate(tqdm(trainloader)):
        images, labels, cpts_gt, cpts_presence, _, dist_map_label= batch
        size = labels.size()
        images = images.to(device)
        labels = labels.long().to(device)
        cpts_gt = cpts_gt.to(device)
        cpts_presence = cpts_presence.to(device)
        total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)
        
        dist_map_label = dist_map_label.to(device)

        seg_out, cpts_out = model(images)
        ph, pw = seg_out.size(2), seg_out.size(3)
        h, w = labels.size(1), labels.size(2)
        if ph != h or pw != w:
            seg_out = F.interpolate(input=seg_out, size=(h, w), mode='bilinear')

        seg_loss = Seg_loss(seg_out, labels)
        
        cpts_out = torch.reshape(cpts_out, (cpts_gt.size(0), cpts_gt.size(1), cpts_gt.size(2)))
        if stage == 1:
            cpts_out = cpts_out.detach()
            seg_loss2 = Seg_loss2(seg_out, dist_map_label)
            cpts_loss = 0
            cpts_loss2 = 0
        else:  
            seg_loss2 = Seg_loss2(seg_out, dist_map_label)
            cpts_loss = Landmark_loss(cpts_out, cpts_gt)
            cpts_loss2 = Landmark_loss2(cpts_out, cpts_gt)

        # calculate euclidean_distance between predicted and ground-truth landmarks
        norm_squared_distance = torch.square(cpts_out-cpts_gt).detach() * cpts_presence
        squared_distance = torch.zeros_like(norm_squared_distance)
        squared_distance[:, :, 0] = norm_squared_distance[:, :, 0]*(1280**2)
        squared_distance[:, :, 1] = norm_squared_distance[:, :, 1]*(736**2)
        euclidean_distance = torch.sum(
            squared_distance, dim=(2), keepdim=True)
        euclidean_distance = torch.sqrt(euclidean_distance.squeeze(dim=2))
        # calculate how many points are within 144 pixels from their corresponding ground truth
        num_inThresh = ((euclidean_distance >= 0) & (
            euclidean_distance <= 144)).float()
        num_inThresh = num_inThresh*cpts_presence[:, :, 0]
        total_num_inThresh += torch.sum(num_inThresh)
        total_num_Present += torch.sum(cpts_presence[:, :, 0])
        total_distance += torch.sum(euclidean_distance)

        cpts_loss = cpts_loss * cpts_presence
        cpts_loss2 = cpts_loss2 * cpts_presence

        if torch.sum(cpts_presence) > 0:
            cpts_loss = torch.sum(cpts_loss) / torch.sum(cpts_presence)
            cpts_loss2 = torch.sum(cpts_loss2) / torch.sum(cpts_presence)
        else:
            cpts_loss = torch.sum(cpts_loss)
            cpts_loss2 = torch.sum(cpts_loss2)
            

        if torch.isnan(cpts_loss):
            print("cpts_loss is nan")

        if torch.isnan(seg_loss):
            print("seg_loss is nan")
            
        seg_loss      = seg_loss*loss_weight[0]
        seg_loss2     = seg_loss2*loss_weight[1]
        cpts_loss     = cpts_loss*loss_weight[2]
        cpts_loss2    = cpts_loss2*loss_weight[3]
        loss          = seg_loss + seg_loss2 + cpts_loss + cpts_loss2
        
        if torch.isnan(loss):
            print("loss is nan")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_seg_loss.update(seg_loss.item())
        ave_bound_loss.update(seg_loss2.item())
        ave_Wing_loss.update(cpts_loss.item())
        ave_FL_loss.update(cpts_loss2.item())

        confusion_matrix += get_confusion_matrix(
            labels,
            seg_out,
            size,
            config.DATASET.NUM_CLASSES,
            config.TRAIN.IGNORE_LABEL)
        lr = base_lr * ((1-float(i_iter+cur_iters)/num_iters)**(0.9))
        
        if stage==1:
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
        else:
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.param_groups[2]['lr'] = lr
            
    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    train_accuracy = tp.sum()/pos.sum()*100
    train_recall = (tp/np.maximum(1.0, pos))*100
    train_precision = (tp/np.maximum(1.0, res))*100
    train_IoU = (tp / np.maximum(1.0, pos + res - tp))*100
    train_mIoU = train_IoU[-2:].mean()
    train_mdistance = total_distance/total_num_Present
    train_mpck20 = total_num_inThresh/total_num_Present*100
    train_total_loss = ave_loss.average()
    train_segmentation_loss = ave_seg_loss.average()
    train_bound_loss = ave_bound_loss.average()
    train_Wing_loss = ave_Wing_loss.average()
    train_FL_loss = ave_FL_loss.average()
    

    msg = 'Epoch: [{}/{}], Time: {:.2f}, ' \
        'lr: {:.6f}, Train_total_Loss: {:.6f}, Train_seg_loss: {:.6f}, Train_bound_loss:{:.6f},Train_Wing_loss: {:.6f}, Train_FL_loss: {:.6f}'.format(
            epoch, num_epoch, batch_time.average(
            ), optimizer.param_groups[0]['lr'], train_total_loss, train_segmentation_loss, train_bound_loss, train_Wing_loss, train_FL_loss
        )
    logging.info(msg)

    # Here we add_scalar every config.PRINT_FREQ.
    # Since in the same epoch, the global step is the same, when add_scalar, it will overwrite the previous one.
    writer.add_scalar('Loss/train_total_loss', train_total_loss, global_steps)
    writer.add_scalar('Seg_loss/train_segmentation_loss',
                      train_segmentation_loss, global_steps)
    writer.add_scalar('Boundary_loss/train_boundary_loss',
                      train_bound_loss, global_steps)
    writer.add_scalar('Landmark_loss/train_landmark_loss',
                      train_Wing_loss, global_steps)
    writer.add_scalar('Landmark_loss2/train_landmark_loss2',
                      train_FL_loss, global_steps)
    writer.add_scalar('Mean_distance/train_mdistance',
                      train_mdistance, global_steps)
    writer.add_scalar('mIoU/train_mIoU', train_mIoU, global_steps)
    writer.add_scalar('MPCK20/train_mpck20', train_mpck20, global_steps)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_steps)

    writer_dict['train_global_steps'] = global_steps + 1
    return train_total_loss, train_mIoU, train_IoU, train_accuracy, train_recall, train_precision, train_mdistance, train_mpck20


def validate(config, testloader, model, Seg_loss, Seg_loss2, Landmark_loss, Landmark_loss2, writer_dict, device, stage, loss_weight):
    model.eval()
    ave_loss = AverageMeter()
    ave_seg_loss = AverageMeter()
    ave_bound_loss = AverageMeter()
    ave_Wing_loss = AverageMeter()
    ave_FL_loss = AverageMeter()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    total_num_inThresh = 0
    total_num_points = 0
    total_distance = 0
    total_num_Present= 0 

    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, label, cpts_gt, cpts_presence, _, dist_map_label = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)
            cpts_gt = cpts_gt.to(device)
            cpts_presence = cpts_presence.to(device)
            total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)
            
            dist_map_label = dist_map_label.to(device)

            seg_pre, cpts_pre = model(image)
            pred = F.interpolate(input=seg_pre, size=(
                size[-2], size[-1]), mode='bilinear')
            seg_loss = Seg_loss(pred, label)
            
            cpts_pre = torch.reshape(
                cpts_pre, (cpts_pre.size(0), cpts_gt.size(1), cpts_gt.size(2)))
            if stage == 1:
                seg_loss2=Seg_loss2(pred, dist_map_label)
                cpts_loss = 0
                cpts_loss2 = 0
            else:
                seg_loss2=Seg_loss2(pred, dist_map_label)
                cpts_loss = Landmark_loss(cpts_pre, cpts_gt)
                cpts_loss2 = Landmark_loss2(cpts_pre, cpts_gt)

            # calculate euclidean_distance between predicted and ground-truth landmarks
            norm_squared_distance = torch.square(cpts_pre-cpts_gt) * cpts_presence
            squared_distance = torch.zeros_like(norm_squared_distance)
            squared_distance[:, :, 0] = norm_squared_distance[:, :, 0]*(1280**2)
            squared_distance[:, :, 1] = norm_squared_distance[:, :, 1]*(736**2)
            euclidean_distance = torch.sum(
                squared_distance, dim=(2), keepdim=True)
            euclidean_distance = torch.sqrt(euclidean_distance.squeeze(dim=2))
            # calculate how many points are within 144 pixels from their corresponding ground truth
            num_inThresh = ((euclidean_distance >= 0) & (
                euclidean_distance <= 144)).float()
            num_inThresh = num_inThresh*cpts_presence[:, :, 0]
            total_num_inThresh += torch.sum(num_inThresh)
            total_num_Present += torch.sum(cpts_presence[:, :, 0])
            total_distance += torch.sum(euclidean_distance)
            

            cpts_loss = cpts_loss * cpts_presence
            cpts_loss2 = cpts_loss2 * cpts_presence
            if torch.sum(cpts_presence) > 0:
                cpts_loss = torch.sum(cpts_loss) / torch.sum(cpts_presence)
                cpts_loss2 = torch.sum(cpts_loss2) / torch.sum(cpts_presence)
            else:
                cpts_loss = torch.sum(cpts_loss)
                cpts_loss2 = torch.sum(cpts_loss2)

            seg_loss      = seg_loss*loss_weight[0]
            seg_loss2     = seg_loss2*loss_weight[1]
            cpts_loss     = cpts_loss*loss_weight[2]
            cpts_loss2 = cpts_loss2*loss_weight[3]
            loss          = seg_loss + seg_loss2 + cpts_loss + cpts_loss2

            ave_loss.update(loss.item())
            ave_seg_loss.update(seg_loss.item())
            ave_bound_loss.update(seg_loss2.item())

            ave_Wing_loss.update(cpts_loss.item())
            ave_FL_loss.update(cpts_loss2.item())

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    accuracy = tp.sum()/pos.sum()*100
    recall = (tp/np.maximum(1.0, pos))*100
    precision = (tp/np.maximum(1.0, res))*100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))*100
    mean_IoU = IoU_array[-2:].mean()
    val_total_loss = ave_loss.average()
    val_segmentation_loss = ave_seg_loss.average()
    val_boundary_loss = ave_bound_loss.average()
    val_Wing_loss = ave_Wing_loss.average()
    val_FL_loss = ave_FL_loss.average()

    mean_distance = total_distance/total_num_Present
    mpck20 = total_num_inThresh/total_num_Present*100
    

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('Loss/valid_loss', val_total_loss, global_steps)
    writer.add_scalar('Seg_loss/valid_segmentation_loss',
                      val_segmentation_loss, global_steps)
    writer.add_scalar('Boundary_loss/valid_boundary_loss',
                      val_boundary_loss, global_steps)
    writer.add_scalar('Landmark_loss/valid_landmark_loss',
                      val_Wing_loss, global_steps)
    writer.add_scalar('Landmark_loss2/valid_landmark_loss2',
                      val_FL_loss, global_steps)
    writer.add_scalar('Mean_distance/valid_mDistance',
                      mean_distance, global_steps)
    writer.add_scalar('mIoU/valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('MPCK20/valid_MPCK20', mpck20, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    
    msg = 'Valid_total_Loss: {:.6f}, Valid_seg_loss: {:.6f}, Valid_bound_loss:{:.6f}, Valid_Wing_loss: {:.6f}, Valid_FL_loss: {:.6f}'.format(
            val_total_loss, val_segmentation_loss, val_boundary_loss, val_Wing_loss, val_FL_loss)
    logging.info(msg)
    
    return val_total_loss, mean_IoU, IoU_array, accuracy, recall, precision, mean_distance, mpck20

def test(testloader, model, sv_dir='', sv_pred=True, device=None):
    model.eval()
    total_num_points = 0
        
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, label, cpts_gt, cpts_presence, name = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)
            cpts_gt = cpts_gt.to(device)
            cpts_presence = cpts_presence.to(device)
            total_num_points +=cpts_presence.size(0)*cpts_presence.size(1)

            seg_pre, cpts_pre = model(image)
            pred = F.interpolate(input=seg_pre, size=(
                size[-2], size[-1]), mode='bilinear')

            cpts_pre = torch.reshape(
                cpts_pre, (cpts_pre.size(0), cpts_gt.size(1), cpts_gt.size(2)))

            cpts_pre= cpts_pre*cpts_presence

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            pred = pred.cpu().numpy().copy()
            ori_image = image.cpu().numpy().copy()
            ground_truth_landmarks = cpts_gt.cpu().numpy().copy()
            predicted_landmarks = cpts_pre.cpu().numpy().copy()
            label = label.cpu().numpy().copy()
            pred = np.asarray(np.argmax(pred, axis=1), dtype=np.uint8)
            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                for i in range(pred.shape[0]):
                    ori_image = (
                        (ori_image[0].transpose(1, 2, 0) * std) + mean)*255
                    result_image = ori_image.copy()
                    # Define colors
                    transparent = (0, 0, 0, 0)  # Transparent
                    cls_colors = [(77, 77, 255, 200), (255, 255, 77, 200), 
                                  (180, 77, 224, 255), (77, 255, 77, 255),
                                  (122, 233, 222, 255), (255, 77, 255, 255)]
                    
                    # Overlay ground truth mask1_contour
                    ground_truth_mask = np.array(label[0], dtype=np.uint8)
                    mask = (ground_truth_mask == 1).astype(np.uint8)
                    _, binary= cv2.threshold(mask, 0.5, 255, 0)
                    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(result_image, contours, -1, (77, 77, 255),  3) 
                    # Overlay ground truth mask2_contour
                    mask = (ground_truth_mask == 2).astype(np.uint8)
                    _, binary= cv2.threshold(mask, 0.5, 255, 0)
                    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(result_image, contours, -1, (255, 255, 77),  3) 
                    
                    # conver from numpy to PIL
                    result_image = Image.fromarray(np.uint8(result_image))
                    draw = ImageDraw.Draw(result_image) 
                    
                    # Overlay ground truth mask1 and 2, comment this section if only want to plot contour
                    ground_truth_mask = np.array(label[0], dtype=np.uint8)
                    mask_image = Image.new(
                        "RGBA", result_image.size, transparent)
                    for class_id in [1, 2]:
                        mask = (ground_truth_mask == class_id).astype(
                            np.uint8) * 200
                        mask_image.paste(
                            cls_colors[class_id-1], (0, 0), Image.fromarray(mask).convert("L"))
                    result_image.paste(mask_image, (0, 0), mask_image)
                                        
                    
                    # Overlay predicted mask1             
                    predicted_mask = np.array(pred[0], dtype=np.uint8)
                    cls1_mask = Image.new("RGBA", result_image.size, transparent)
                    mask = (predicted_mask == 1).astype(np.uint8) * 200
                    cls1_mask.paste(cls_colors[0], (0, 0), Image.fromarray(mask).convert("L")) 
                    result_image.paste(cls1_mask, (0, 0), cls1_mask)
                    # Overlay predicted mask2
                    cls2_mask = Image.new("RGBA", result_image.size, transparent)
                    mask = (predicted_mask == 2).astype(np.uint8) * 200
                    cls2_mask.paste(cls_colors[1], (0, 0), Image.fromarray(mask).convert("L")) 
                    result_image.paste(cls2_mask, (0, 0), cls2_mask)

                    # Overlay ground truth landmarks as circles
                    j = 2
                    for landmark in ground_truth_landmarks[0]:
                        x, y = landmark
                        if x != 0 and y != 0:
                            draw.ellipse(
                                [(x*1280 - 20, y*736 - 20), (x*1280 + 20, y*736 + 20)], fill=cls_colors[j])
                        j += 1
                    
                    # # Overlay predicted landmarks as crosses
                    j =2
                    for landmark in predicted_landmarks[0]:
                        x, y = landmark
                        if x != 0 and y != 0:
                            draw.line([(x*1280 - 15, y*736 - 15), (x*1280 + 15,
                                      y*736 + 15)], fill=cls_colors[j], width=6)
                            draw.line([(x*1280 + 15, y*736 - 15), (x*1280 - 15,
                                      y*736 + 15)], fill=cls_colors[j], width=6)
                        j += 1
                result_image.save(os.path.join(sv_path, name[i]+'.png'))
