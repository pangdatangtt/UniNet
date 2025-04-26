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


if __name__ == "__main__":
    from utils import setup_seed, get_logger
    from main import parsing_args
    setup_seed(1203)
    c = parsing_args()
    c.dataset = 'MVTec AD'
    if not c.weighted_decision_mechanism:
        c.default = c.alpha = c.beta = c.gamma = "w/o"

    UnsupervisedAD = ['MVTec AD', 'BTAD', 'MVTec 3D-AD', "VisA", "APTOS", "ISIC2018", "OCT2017", 'ped2']
    SupervisedAD = ["Kvasir-SEG", "CVC-ClinicDB", "CVC-ColonDB", "VAD"]

    mvtec_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                  'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    # mvtec_list = ['tile']

    mvtec3d_list = ["bagel", "carrot", "cookie", "dowel", "foam",
                    "peach", "potato", "tire", "rope", "cable_gland"]
    # mvtec3d_list = ["carrot"]

    btad_list = ["01", "02", "03"]

    visa_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
                 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

    dataset_name = c.dataset
    if dataset_name in UnsupervisedAD:
        if dataset_name == 'MVTec AD':
            dataset = mvtec_list
        elif dataset_name == "MVTec 3D-AD":
            dataset = mvtec3d_list
        elif dataset_name == 'BTAD':
            dataset = btad_list
        elif dataset_name == 'VisA':
            dataset = visa_list
        elif dataset_name in ["APTOS", "ISIC2018", "OCT2017", "ped2"]:
            dataset = [dataset_name]
        else:
            raise KeyError(f"Dataset '{dataset_name}' not found.")

    elif dataset_name in SupervisedAD:
        dataset = [dataset_name]

    else:
        raise KeyError(f"Dataset '{dataset_name}' not found.")

    from tabulate import tabulate

    results = {}
    table_ls = []
    image_auroc_list = [[], []]
    pixel_auroc_list = [[], []]
    pixel_aupro_list = [[], []]
    for idx, i in enumerate(dataset):
        c._class_ = i
        # from utils_cvpr import epochs_
        #
        # c.epochs = epochs_(i)
        # if dataset_name == 'MVTec 3D-AD':
            # c = preprocess_for_3d(c)

        args_dict = vars(c)
        args_info = f"class:{i}, "
        for key, value in args_dict.items():
            if key in ['_class_']:
                continue
            args_info += ", ".join([f"{key}:{value}, "])
        print('testing on {} dataset'.format(dataset_name)) if idx == 0 else None
        print(args_info)

        auroc_sp, auroc_px, aupro_px = test(c)
        table_ls.append(['{}'.format(i), str(np.round(auroc_sp, decimals=1)),
                         str(np.round(auroc_px, decimals=1)),
                         str(np.round(aupro_px, decimals=1))])
        image_auroc_list[0].append(auroc_sp)
        pixel_auroc_list[0].append(auroc_px)
        pixel_aupro_list[0].append(aupro_px)
        results = tabulate(table_ls, headers=['object', 'image_auroc', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    table_ls.append(['mean', str(np.round(np.mean(image_auroc_list[0]), decimals=1)),
                     str(np.round(np.mean(pixel_auroc_list[0]), decimals=1)),
                     str(np.round(np.mean(pixel_aupro_list[0]), decimals=1))])
    results = tabulate(table_ls, headers=['object', 'image_auroc', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    print(results)
