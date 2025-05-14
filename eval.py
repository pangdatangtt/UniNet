import glob
import math
import os
import re
import time
import torch
import numpy as np
from skimage.measure import regionprops
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, accuracy_score, f1_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from scipy.ndimage import gaussian_filter
from sklearn import manifold, metrics
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib

from utils import t2np, rescale
import sys
from functools import partial
from multiprocessing import Pool
from skimage.measure import label, regionprops

from video_dataset import Label_loader
from UniNet_lib.mechanism import weighted_decision_mechanism


def evaluation_indusAD(c, model, dataloader, device):
    model.train_or_eval(type='eval')
    n = model.n
    is_similarity = c.weighted_decision_mechanism
    gt_list_px = []
    gt_list_sp = []
    output_list = [list() for _ in range(n*3)]
    weights_cnt = 0

    start_time = time.time()
    with torch.no_grad():
        for idx, (sample, label, gt) in enumerate(dataloader):

            gt_list_sp.extend(t2np(label))
            gt_list_px.extend(t2np(gt))
            weights_cnt += 1

            img = sample[0].to(device) if c.dataset == "MVTec 3D-AD" else sample.to(device)
            t_tf, de_features = model(img)

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)  # B*64*64
                # print(output, output.shape)
                output_list[l].append(output)
        fps = len(dataloader.dataset) / (time.time() - start_time)
        print("fps:", fps, len(dataloader.dataset))

        # postprocess
        anomaly_score, anomaly_map = weighted_decision_mechanism(weights_cnt, output_list, c.alpha, c.beta)
        # anomaly_score = gaussian_filter(anomaly_score, sigma=4) if is_similarity else \
        #     [np.max(gaussian_filter(anomaly_score, sigma=4)[i, :, :, :].numpy()) for i in range(anomaly_score.shape[0])]

        # anomaly_score_add = gaussian_filter(anomaly_score_add, sigma=4)

        gt_label = np.asarray(gt_list_sp, dtype=np.bool_)
        gt_mask = np.squeeze(np.asarray(gt_list_px, dtype=np.bool_), axis=1)

        auroc_px = round(roc_auc_score(gt_mask.flatten(), anomaly_map.flatten()) * 100, 1)
        auroc_sp = round(roc_auc_score(gt_label, anomaly_score) * 100, 1)

        pro = round(eval_seg_pro(gt_mask, anomaly_map), 1)

    return auroc_px, auroc_sp, pro


def evaluation_vad(c, model, dataloader, device):
    model.train_or_eval(type='eval')
    n = model.n
    gt_list_sp = []
    pr_list_sp = []
    output_list = [list() for _ in range(n*3)]
    weights_cnt = 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(dataloader):

            img, label = img.to(device), label.to(device)
            t_tf, de_features, _ = model(img)

            label[label > 0.5] = 1
            gt_list_sp.extend(t2np(label))
            weights_cnt += 1

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)  # B*64*64
                # print(output, output.shape)
                output_list[l].append(output)

        if c.weighted_decision_mechanism:
            anomaly_score, _ = weighted_decision_mechanism(weights_cnt, output_list, c.alpha, c.beta)
            # weights_list = [0.03] * weights_cnt
            # anomaly_map = gaussian_filter(anomaly_score, sigma=4)
            pr_list_sp = anomaly_score

        thresh = return_best_thr(gt_list_sp, pr_list_sp)
        acc = accuracy_score(gt_list_sp, pr_list_sp >= thresh) * 100
        f1 = f1_score(gt_list_sp, pr_list_sp >= thresh) * 100
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4) * 100

    return auroc_sp, f1, acc


def eval_seg_pro(gt_mask, anomaly_score_map, max_step=800):
    expect_fpr = 0.3    # default 30%

    max_th = anomaly_score_map.max()
    min_th = anomaly_score_map.min()
    delta = (max_th - min_th) / max_step
    threds = np.arange(min_th, max_th, delta).tolist()

    pool = Pool(8)
    ret = pool.map(partial(single_process, anomaly_score_map, gt_mask), threds)
    pool.close()
    pros_mean = []
    fprs = []
    for pro_mean, fpr in ret:
        pros_mean.append(pro_mean)
        fprs.append(fpr)
    pros_mean = np.array(pros_mean)
    fprs = np.array(fprs)
    # expect_fpr = sum(fprs) / len(fprs)
    idx = fprs < expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    loc_pro_auc = auc(fprs_selected, pros_mean_selected) * 100

    return loc_pro_auc


def single_process(anomaly_score_map, gt_mask, thred):
    binary_score_maps = np.zeros_like(anomaly_score_map, dtype=np.bool_)
    binary_score_maps[anomaly_score_map <= thred] = 0
    binary_score_maps[anomaly_score_map >  thred] = 1
    pro = []
    for binary_map, mask in zip(binary_score_maps, gt_mask):    # for i th image
        for region in regionprops(label(mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            tp_pixels = binary_map[axes0_ids, axes1_ids].sum()
            pro.append(tp_pixels / region.area)

    pros_mean = np.array(pro).mean()
    inverse_masks = 1 - gt_mask
    fpr = np.logical_and(inverse_masks, binary_score_maps).sum() / inverse_masks.sum()
    return pros_mean, fpr


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def evaluation_batch(c, model, dataloader, device, _class_=None, reg_calib=False, max_ratio=0):
    model.train_or_eval(type='eval')
    gt_list_sp = []
    output_list = [list() for i in range(6)]
    weights_cnt = 0
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, cls in dataloader:
            img = img.to(device)
            gt_list_sp.extend(t2np(label))
            t_tf, de_features = model(img)
            weights_cnt += 1

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)  # B*64*64
                # print(output, output.shape)
                output_list[l].append(output)

        anomaly_score, _ = weighted_decision_mechanism(weights_cnt, output_list, c.alpha, c.beta)


        # anomaly_score = gaussian_filter(anomaly_score, sigma=4)
        gt_list_sp = np.asarray(gt_list_sp, dtype=np.bool_)
        # pr_list_sp.extend(sp_score)

        auroc_sp = round(roc_auc_score(gt_list_sp, anomaly_score), 4)
        ap_sp = round(average_precision_score(gt_list_sp, anomaly_score), 4)
        f1_sp = f1_score_max(gt_list_sp, anomaly_score)

    return auroc_sp, ap_sp, f1_sp


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


# for medical AD
def evaluation_mediAD(c, model, dataloader, device, _class_=None, reduction='max'):
    model.train_or_eval(type='eval')
    n = model.n
    weights_cnt = 0
    output_list = [list() for _ in range(n*3)]
    gt_list_sp = []
    pr_list_sp = []
    with torch.no_grad():
        for img, label, _ in dataloader:
            img = img.to(device)
            t_tf, de_features = model(img)

            gt_list_sp.extend(t2np(label))
            weights_cnt += 1

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)  # B*64*64
                # print(output, output.shape)
                output_list[l].append(output)

        if c.weighted_decision_mechanism:
            anomaly_score, _ = weighted_decision_mechanism(weights_cnt, output_list, c.alpha, c.beta)
            # anomaly_map = gaussian_filter(anomaly_score, sigma=4)
            pr_list_sp = anomaly_score

        thresh = return_best_thr(gt_list_sp, pr_list_sp)
        acc = accuracy_score(gt_list_sp, pr_list_sp >= thresh) * 100
        f1 = f1_score(gt_list_sp, pr_list_sp >= thresh) * 100
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4) * 100
    return auroc_sp, f1, acc


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def evaluation_polypseg(c, model, test_dataset, num1, trainsize=256):
    model.train_or_eval(type='eval')
    DSC = 0.0
    IOU = 0.0
    n = model.n

    for i in range(num1):
        image, gt, name = test_dataset.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        t1, de_features, recon = model(image)

        res = F.interpolate((recon[0][0]+recon[0][-1])#+ recon[1][0]+recon[1][-1])
                            , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

        iou = (intersection.sum() + smooth) / \
              (input.sum() + target.sum() - intersection.sum() + smooth)
        IOU = IOU + iou

    return DSC / num1, IOU / num1


def extract_numbers(file_name):
    numbers = re.findall(r'(\d+)', file_name)
    return tuple(map(int, numbers))


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr+1e-8)))


def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    max_ele = np.max(psnr_list)
    min_ele = np.min(psnr_list)
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], max_ele, min_ele))

    return anomaly_score_list


def evaluation_video(c, model, test_folder, dataloader, device):
    from collections import OrderedDict
    # labels_list = np.load('./data/frame_labels_' + args.dataset_type + '.npy')
    test_folders = os.listdir(test_folder)
    test_folders = sorted(test_folders, key=extract_numbers)
    test_folders = [os.path.join(test_folder, aa) for aa in test_folders]
    test_length = len(test_folders)
    gt_loader = Label_loader(c, test_folders)  # Get gt labels.
    gt = gt_loader()
    labels_list = np.load('video/ped2/frame_labels_ped2.npy')

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    label_length = 0
    list1 = {}
    list2 = {}
    list3 = {}
    list4 = {}
    list5 = {}

    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        label_length += videos[video_name]['length']
        list1[video_name] = []
        list2[video_name] = []
        list3[video_name] = []
        list4[video_name] = []
        list5[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    model.train_or_eval(type='eval')
    n = model.n
    weights_cnt = 0
    output_list = [list() for _ in range(n * 3)]
    recon_list = []
    recon_list1 = []
    test_length_list = []
    test_length_list.append(label_length)
    with torch.no_grad():
        for k, (imgs, _) in enumerate(dataloader):
            if k == label_length:# - 4 * (video_num + 1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1]]['length']
                test_length_list.append(videos[videos_list[video_num].split('/')[-1]]['length'])

            imgs = (imgs).cuda()
            # imgs_ = projectionlayer(imgs)
            t1, de_features, pred = model(imgs)
            weights_cnt += 1

            for l, (t, s) in enumerate(zip(t1, de_features)):
                output = 1 - F.cosine_similarity(t, s)  # B*64*64
                # print(output, output.shape)
                output_list[l].append(output)

            # recon_list.append((grad_loss(pred[0], imgs).mean() + grad_loss(pred[-1], imgs).mean()).detach().cpu().numpy())
            recon_list1.append((F.mse_loss(pred[0], imgs[:, -3:]) + F.mse_loss(pred[-1], imgs[:, -3:])).detach().cpu().numpy())

            # anomaly_map, _ = cal_anomaly_map_rd(t1, de_features, 256, 'a')
            # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # score = np.mean(anomaly_map)

            # latest_losses = {'mse1': F.mse_loss(pred[0], imgs) + F.mse_loss(pred[-1], imgs),
            #                  # 'grad1': grad_loss(pred[0], imgs[:, -3:]).mean() + grad_loss(pred[-1], imgs[:, -3:]).mean(),
            #                  'score': score
            #                  }

            # list1[videos_list[video_num].split('/')[-1]].append(float(latest_losses['mse1']))
            # list2[videos_list[video_num].split('/')[-1]].append(float(latest_losses['mse2']))
            # list3[videos_list[video_num].split('/')[-1]].append(float(latest_losses['grad1']))
            # list4[videos_list[video_num].split('/')[-1]].append(float(latest_losses['grad2']))
            # list5[videos_list[video_num].split('/')[-1]].append(float(latest_losses['score']))

        anomaly_score, _ = weighted_decision_mechanism(weights_cnt, output_list, c.alpha, c.beta)
        anomaly_map = anomaly_score#gaussian_filter(anomaly_score, sigma=4)

        anomaly_list1 = []
        anomaly_list2 = []
        anomaly_list3 = []
        anomaly_list4 = []
        anomaly_list5 = []

        for video in sorted(videos_list):
            break
            video_name = video.split('/')[-1]
            anomaly_list1 += anomaly_score_list_inv(list1[video_name])
            # anomaly_list2 += anomaly_score_list_inv(list2[video_name])
            # anomaly_list3 += anomaly_score_list_inv(list3[video_name])
            # anomaly_list4 += anomaly_score_list_inv(list4[video_name])
            anomaly_list5 += anomaly_score_list_inv(list5[video_name])

        def conf_avg(x, size=11, n_conf=5):
            a = x.copy()
            b = []
            weight = np.array([1, 1, 1, 1, 1.2, 1.6, 1.2, 1, 1, 1, 1])

            for i in range(x.shape[0] - size + 1):
                a_ = a[i:i + size].copy()
                u = a_.mean()
                dif = abs(a_ - u)
                sot = np.argsort(dif)[:n_conf]
                mask = np.zeros_like(dif)
                mask[sot] = 1
                weight_ = weight * mask
                b.append(np.sum(a_ * weight_) / weight_.sum())
            for _ in range(size // 2):
                b.append(b[-1])
                b.insert(0, 1)
            return b

        # anomaly_list1 = conf_avg(np.array(anomaly_list1))
        # anomaly_list2 = conf_avg(np.array(anomaly_list2))
        # anomaly_list3 = conf_avg(np.array(anomaly_list3))
        # anomaly_list4 = conf_avg(np.array(anomaly_list4))
        # anomaly_list5 = conf_avg(np.array(anomaly_list5))
        # recon_list = conf_avg(np.array(recon_list))
        # recon_list1 = conf_avg(np.array(recon_list1))
        # anomaly_map = conf_avg(np.array(anomaly_map))


        hyp_alpha = [1, 1]

        # comb = np.array(anomaly_list1) * hyp_alpha[0] + np.array(anomaly_list5) * hyp_alpha[1]
        # accuracy = roc_auc_score(y_true=1 - labels_list, y_score=comb)

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        start = 0
        end = test_length_list[0]
        for i in range(test_length):
            score = []
            for j in range(start, end):
                score.append(anomaly_map[j][0] * 1 + recon_list1[j] * 1) #+ recon_list1[j])

            scores = np.concatenate((scores, score), axis=0)

            label = gt[i][:len(score)]
            labels = np.concatenate((labels, label), axis=0)

            if i+1 < test_length:
                start = start + end
                end = end + test_length_list[i+1]
        fpr, tpr, _ = metrics.roc_curve(labels, scores)
        accuracy = metrics.auc(fpr, tpr)

        return accuracy * 100



