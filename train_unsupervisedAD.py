import copy
import time

import torch
import numpy as np
import os

from UniNet_lib.resnet import wide_resnet50_2
from UniNet_lib.de_resnet import de_wide_resnet50_2
from datasets import loading_dataset
import torch.backends.cudnn as cudnn
from eval import evaluation_indusAD, evaluation_mediAD, evaluation_video
from torch.nn import functional as F
from utils import setup_seed, count_parameters, save_weights, to_device, get_logger
from UniNet_lib.model import UniNet, EarlyStopping
from UniNet_lib.DFS import DomainRelated_Feature_Selection


def train(c):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    dataset_name = c.dataset
    if c._class_ in [dataset_name]:
        ckpt_path = os.path.join("./ckpts", dataset_name)
    else:
        ckpt_path = os.path.join("./ckpts", dataset_name, f"{c._class_}")

    # ---------------------------------loading dataset-----------------------------------------------
    train_dataloader, test_dataloader = loading_dataset(c, dataset_name)

    # ---------------------------------loading model-------------------------------------------------
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
    optimizer1 = torch.optim.AdamW(list(Target_teacher.parameters()), lr=1e-4 if c._class_ == 'transistor' else c.lr_t,
                                   betas=(0.9, 0.999), weight_decay=1e-5)
    model = UniNet(c, Source_teacher, Target_teacher, bn, student, DFS=DFS)

    # total_params = count_parameters(model)
    # print("Number of parameter: %.2fM" % (total_params/1e6))

    auroc_sp, auroc_px, aupro_px, max_IRoc, max_PRoc, max_PPro = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_iters = 2000 if dataset_name == "ISIC2018" else 1000
    it = 0
    early_stopping = EarlyStopping(patience=3, verbose=False)

    # ---------------------------------------------training-----------------------------------------------
    for epoch in range(c.epochs):
        model.train_or_eval(type='train')
        loss_list = []
        for sample in train_dataloader:
            img = sample[0][0].to(device) if dataset_name == "MVTec 3D-AD" else sample[0].to(device)
            loss = model(img, stop_gradient=dataset_name in ["APTOS", "ISIC2018", "OCT2017"])
            optimizer.zero_grad()
            optimizer1.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            optimizer1.step()
            loss_list.append(loss.item())

            # ------------------------------------eval medical AD-------------------------------------------
            if dataset_name in ["APTOS", "ISIC2018", "OCT2017"]:
                if (it + 1) % 250 == 0:
                    print('iters: {}/{}, loss:{:.4f}'.format(it + 1, total_iters, np.mean(loss_list)))

                    modules_list = [model.t.t_t, model.bn.bn, model.s.s1, DFS]
                    auroc, f1, acc = evaluation_mediAD(c, model, test_dataloader, device)
                    print('Auroc: {:.2f}, f1: {:.2f}, acc: {:.2f}'.format(auroc, f1, acc))
                    if max_IRoc < auroc:
                        max_IRoc = auroc
                        save_weights(modules_list, ckpt_path, "BEST_I_ROC") if c.is_saved else None
                    model.train_or_eval(type='train')
                it += 1
                if it > total_iters:
                    return

        # ------------------------------------eval industrial and video-------------------------------------

        if dataset_name in ['MVTec AD', 'BTAD', 'MVTec 3D-AD', "VisA", 'ped2']:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, c.epochs, np.mean(loss_list)))

        modules_list = [model.t.t_t, model.bn.bn, model.s.s1, DFS]
        best_iroc = False
        if (epoch + 1) % 10 == 0 and c.domain in ['industrial', 'video']:

            if dataset_name in ['MVTec AD', 'BTAD', 'MVTec 3D-AD', "VisA"]:
                # evaluation
                auroc_px, auroc_sp, aupro_px = evaluation_indusAD(c, model, test_dataloader, device)
                print('Sample Auroc: {:.1f}, Pixel Auroc: {:.1f}, Pixel Aupro: {:.1f}'.format(auroc_sp, auroc_px,
                                                                                              aupro_px))
                if max_IRoc < auroc_sp:
                    max_IRoc = auroc_sp
                    # save_weights(modules_list, ckpt_path, "BEST_I_ROC") if c.is_saved else None
                    best_iroc = True
                if max_PRoc < auroc_px:
                    max_PRoc = auroc_px
                    save_weights(modules_list, ckpt_path, "BEST_P_ROC") if c.is_saved else None
                if (best_iroc and max_PPro == aupro_px) or max_PPro < aupro_px:
                    max_PPro = aupro_px
                    print('saved')
                    save_weights(modules_list, ckpt_path, "BEST_P_PRO") if c.is_saved else None
                print(f"MAX I_ROC: {max_IRoc:.1f}, MAX P_ROC: {max_PRoc:.1f}, MAX P_PRO: {max_PPro:.1f}")
                early_stopping(aupro_px)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            else:
                test_folder = 'video/ped2/testing/frames'
                auroc = evaluation_video(c, model, test_folder, test_dataloader, device)
                print('Auroc: {:.2f}'.format(auroc))
