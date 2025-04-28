import copy
import time

import torch
import numpy as np
import os

from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2
from datasets import loading_dataset
import torch.backends.cudnn as cudnn
from eval import evaluation_vad
from torch.nn import functional as F
from utils import setup_seed, count_parameters, save_weights, to_device, get_logger
from UniNet_lib.model_classfication import UniNet
from UniNet_lib.DFS import DomainRelated_Feature_Selection


def train(c):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    dataset_name = c.dataset
    ckpt_path = os.path.join("./ckpts", dataset_name)

    # loading dataset
    train_dataloader, test_dataloader = loading_dataset(c, dataset_name)

    # model
    Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
    Source_teacher.layer4 = None
    Source_teacher.fc = None
    student = de_wide_resnet50_2(pretrained=False)
    DFS = DomainRelated_Feature_Selection()
    [Source_teacher, bn, student, DFS] = to_device([Source_teacher, bn, student, DFS], device)
    Target_teacher = copy.deepcopy(Source_teacher)

    params = list(student.parameters()) + list(bn.parameters()) + list(DFS.parameters())
    optimizer = torch.optim.AdamW(params, lr=c.lr_s, betas=(0.9, 0.999),
                                  weight_decay=1e-5)
    optimizer1 = torch.optim.AdamW(list(Target_teacher.parameters()), lr=c.lr_t,
                                   betas=(0.9, 0.999), weight_decay=1e-5)
    model = UniNet(c, Source_teacher, Target_teacher, bn, student, DFS=DFS)

    it = 0
    total_iters = 1000
    b_auroc, b_f1, b_acc = 0.0, 0.0, 0.0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train_or_eval(type='train')
        loss_list = []
        n = 0
        for sample in train_dataloader:
            img, label = sample[0].to(device), sample[1].to(device)
            loss = model(img, label, stop_gradient=True, max=False)
            optimizer.zero_grad()
            optimizer1.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            optimizer1.step()
            loss_list.append(loss.item())
            if (it + 1) % 75 == 0:
                print('iters: {}/{}, loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                if (it + 1) % (75 * 3) == 0:
                    auroc_sp, f1, acc = evaluation_vad(c, model, test_dataloader, device)
                    print('Sample Auroc: {:.2f}, f1: {:.2f}, acc: {:.2f}'.format(auroc_sp, f1, acc))
                    modules_list = [model.t.t_t, model.bn.bn, model.s.s1, DFS]
                    if b_auroc < auroc_sp:
                        b_auroc = auroc_sp
                        save_weights(modules_list, ckpt_path, "BEST_I_ROC")
                    if b_f1 < f1:
                        b_f1 = f1
                    if b_acc < acc:
                        b_acc = acc
                    model.train_or_eval(type='train')
            it += 1
            if it == total_iters:
                break
        # scheduler.step()
        # auroc_sp = evaluation_vad(c, epoch, model, test_dataloader, device)
        # print('Sample Auroc: {:.2f}'.format(auroc_sp))

    return
