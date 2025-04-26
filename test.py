import copy
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve, roc_auc_score

from UniNet_lib.DFS import DomainRelated_Feature_Selection
from UniNet_lib.mechanism import weighted_decision_mechanism
from UniNet_lib.model import UniNet
from UniNet_lib.de_resnet import de_wide_resnet50_2
from eval import evaluation_indusAD, evaluation_batch, evaluation_mediAD, evaluation_polypseg
from UniNet_lib.resnet import wide_resnet50_2
from utils import load_weights, t2np, to_device
from torch.nn import functional as F
from datasets import loading_dataset


def test(c, suffix='BEST_P_PRO'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    dataset_name = c.dataset
    ckpt_path = None
    if c._class_ in [dataset_name]:
        ckpt_path = os.path.join("./ckpts", dataset_name)
    else:
        if c.setting == 'oc':
            ckpt_path = os.path.join("./ckpts", dataset_name, f"{c._class_}")
        elif c.setting == 'mc':
            ckpt_path = os.path.join("./ckpts", "{}".format(dataset_name), "multiclass")
        else:
            pass

    # --------------------------------------loading dataset------------------------------------------
    dataset_info = loading_dataset(c, dataset_name)
    test_dataloader = dataset_info[1]

    # model
    Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
    Source_teacher.layer4 = None
    Source_teacher.fc = None
    student = de_wide_resnet50_2(pretrained=False)
    DFS = DomainRelated_Feature_Selection()
    [Source_teacher, bn, student, DFS] = to_device([Source_teacher, bn, student, DFS], device)
    Target_teacher = copy.deepcopy(Source_teacher)

    new_state = load_weights([Target_teacher, bn, student, DFS], ckpt_path, suffix)
    Target_teacher = new_state['tt']
    bn = new_state['bn']
    student = new_state['st']
    DFS = new_state['dfs']

    model = UniNet(c, Source_teacher.cuda().eval(), Target_teacher, bn, student, DFS)

    if c.domain == 'industrial':
        if c.setting == 'oc':
            auroc_px, auroc_sp, pro = evaluation_indusAD(c, model, test_dataloader, device)
            return auroc_sp, auroc_px, pro

        else:   # multiclass
            auroc_sp_list, ap_sp_list, f1_list = [], [], []
            # test_dataloader: List
            for test_loader in test_dataloader:
                auroc_sp, ap_sp, f1 = evaluation_batch(c, model, test_loader, device)
                auroc_sp_list.append(auroc_sp * 100)
                ap_sp_list.append(ap_sp * 100)
                f1_list.append(f1 * 100)
            return auroc_sp_list, ap_sp_list, f1_list, dataset_info[-2]

    if c.domain == 'medical':
        if dataset_name in ["APTOS", "ISIC2018", "OCT2017"]:
            auroc_sp, f1, acc = evaluation_mediAD(c, model, test_dataloader, device)
            return auroc_sp, acc, f1

        elif dataset_name in ["Kvasir-SEG", "CVC-ClinicDB", "CVC-ColonDB"]:
            mice, miou = evaluation_polypseg(c, model, test_dataloader, dataset_info[-1])
            return mice, miou
