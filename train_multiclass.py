import copy
import time

import torch
import numpy as np
import os

from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2
import torch.backends.cudnn as cudnn
from eval import evaluation_batch
from torch.nn import functional as F
from utils import setup_seed, count_parameters, save_weights, to_device, get_logger
from UniNet_lib.model import UniNet
from UniNet_lib.DFS import DomainRelated_Feature_Selection
from datasets import loading_dataset


def train_mc(c):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    dataset_name = c.dataset
    ckpt_path = os.path.join("./ckpts", "{}".format(dataset_name), "multiclass")

    # loading dataset
    train_dataloader, test_dataloader_list, class_list, lr = loading_dataset(c, dataset_name)

    # model
    Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
    Source_teacher.layer4 = None
    Source_teacher.fc = None
    student = de_wide_resnet50_2(pretrained=False)
    DFS = DomainRelated_Feature_Selection()
    [Source_teacher, bn, student, DFS] = to_device([Source_teacher, bn, student, DFS], device)
    Target_teacher = copy.deepcopy(Source_teacher)

    # params = list(student.parameters()) + list(bn.parameters()) + list(DFS.parameters())
    optimizer = torch.optim.AdamW([{'params': student.parameters()},
                                   {'params': bn.parameters()},
                                   {'params': DFS.parameters()},
                                   {'params': Target_teacher.parameters(), 'lr': lr['lr_t']}],  # 5e-5
                                  lr=lr['lr_s'], betas=(0.9, 0.999), weight_decay=1e-5,  # 2e-3
                                  eps=1e-10, amsgrad=True)
    model = UniNet(c, Source_teacher, Target_teacher, bn, student, DFS=DFS)

    total_iters = 5000
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_iters * 0.8)], gamma=0.2)
    it = 0
    best = 0.0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train_or_eval(type='train')
        loss_list = []
        for sample in train_dataloader:
            img = sample[0].to(device)
            loss = model(img, max=False)
            optimizer.zero_grad()
            # optimizer1.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            # optimizer1.step()
            loss_list.append(loss.item())
            lr_scheduler.step()
            if (it + 1) % 1000 == 0:
                modules_list = [model.t.t_t, model.bn.bn, model.s.s1, DFS]
                auroc_sp_list, ap_sp_list, f1_list = [], [], []

                for item, test_dataloader in zip(class_list, test_dataloader_list):
                    auroc_sp, ap_sp, f1 = evaluation_batch(c, model, test_dataloader, device)
                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_list.append(f1)

                    print('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, F1:{:.4f}'.format(item,
                                                                              auroc_sp,
                                                                              ap_sp, f1))
                print('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, F1:{:.4f}'.format(np.mean(auroc_sp_list),
                                                                            np.mean(ap_sp_list),
                                                                            np.mean(f1_list)))
                if np.mean(auroc_sp_list) > best:
                    best = np.mean(auroc_sp_list)
                    save_weights(modules_list, ckpt_path, 'BEST_I_ROC') if c.is_saved else None
                model.train_or_eval(type='train')
                if dataset_name == 'MVTec AD':
                    model.t.t_t.eval()
                    model.s.eval()
                    model.bn.eval()
                    model.dfs.eval()
            it += 1
            if it == total_iters:
                break
        print('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
        print()
