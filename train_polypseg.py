import copy
import time

import torch
import numpy as np
import os

from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet_polyp import de_wide_resnet50_2
from datasets import loading_dataset
import torch.backends.cudnn as cudnn
from eval import evaluation_polypseg
from torch.nn import functional as F
from utils import setup_seed, count_parameters, save_weights, to_device, get_logger
from UniNet_lib.model_polyseg import UniNet
from UniNet_lib.DFS import DomainRelated_Feature_Selection


def train_polyp(c):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    dataset_name = c.dataset
    ckpt_path = os.path.join("./ckpts", dataset_name)

    # loading dataset
    train_dataloader, test_dataloader, num1 = loading_dataset(c, dataset_name)

    # model
    Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
    Source_teacher.layer4 = None
    Source_teacher.fc = None
    student = de_wide_resnet50_2(pretrained=False)
    DFS = DomainRelated_Feature_Selection()
    [Source_teacher, bn, student, DFS] = to_device([Source_teacher, bn, student, DFS], device)
    Target_teacher = copy.deepcopy(Source_teacher)

    params = list(student.parameters()) + list(bn.parameters()) + list(DFS.parameters())
    optimizer = torch.optim.AdamW(params, lr=c.lr_s, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer1 = torch.optim.AdamW(list(Target_teacher.parameters()), lr=c.lr_t,
                                   betas=(0.9, 0.999), weight_decay=1e-5)
    model = UniNet(c, Source_teacher, Target_teacher, bn, student, DFS=DFS)

    best_dice, best_iou = 0.0, 0.0
    size_rates = [0.75, 1, 1.25]

    for epoch in range(c.epochs):
        model.train_or_eval(type='train')
        loss_list = []
        for i, pack in enumerate(train_dataloader, start=1):
            for rate in size_rates:
                # ---- data prepare ----
                images, gts = pack
                images = (images).cuda()
                gts = (gts).cuda()
                trainsize = int(round(c.image_size * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # images, gts = pack
                images = (images).cuda()
                gts = (gts).cuda()

                # set stop_gradient=True if the performance is not good
                loss = model(images, gts, stop_gradient=False)
                optimizer.zero_grad()
                optimizer1.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(params, 0.5)
                optimizer.step()
                optimizer1.step()
                if rate == 1:
                    loss_list.append(loss.item())
                    # lr_scheduler.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, c.epochs, np.mean(loss_list)))
        if (epoch + 1) % 10 == 0:
            modules_list = [model.t.t_t, model.bn.bn, model.s.s1, DFS]

            dice, iou = evaluation_polypseg(c, model, test_dataloader, num1, c.image_size)
            best_dice = dice if best_dice < dice else best_dice
            best_iou = iou if best_iou < iou else best_iou
            print("last mean dice:", dice, "best mean dice:", best_dice)
            print("last mean iou:", iou, "best mean iou:", best_iou)
            save_weights(modules_list, ckpt_path, "BEST_DICE") if c.is_saved else None

        return
