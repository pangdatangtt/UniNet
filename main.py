import numpy as np
import os

from train_unsupervisedAD import train
from datasets import preprocess_for_3d, UnsupervisedAD, SupervisedAD, \
    mvtec_list, btad_list, mvtec3d_list, visa_list
import argparse
from utils import setup_seed, get_logger
from test import test


def parsing_args():
    parser = argparse.ArgumentParser(description='UniNet')

    parser.add_argument('--domain', default='industrial', type=str,
                        choices=['industrial', 'medical', 'video', 'natural'], help="choose experimental domain.")
    parser.add_argument('--setting', default='mc', type=str, choices=['oc', 'mc', 'cd'],
                        help="choose experimental settings, including one-class, multi-class, cross-dataset.")
    parser.add_argument('--dataset', default='VisA', type=str,
                        choices=['MVTec AD', 'BTAD', 'MVTec 3D-AD', "VisA", "VAD", "APTOS", "ISIC2018", "OCT2017",
                                 "Kvasir-SEG", "CVC-ClinicDB", "CVC-ColonDB", "Ped2"],
                        help="choose experimental dataset.")
    parser.add_argument('--epochs', default=100, type=int, help="epochs.")
    parser.add_argument('--batch_size', default=8, type=int, help="batch sizes.")
    parser.add_argument('--image_size', default=256, type=int, help="image size.")
    parser.add_argument('--center_crop', default=256, type=int, help="crop image size.")
    parser.add_argument('--lr_s', default=5e-3, type=float, help="lr for student.")  # 5e-3
    parser.add_argument('--lr_t', default=1e-6, type=float, help="lr for teacher.")  # 1e-6
    parser.add_argument('--T', default=2, type=float, help="temperature for contrastive learning.")

    parser.add_argument('--weighted_decision_mechanism', action='store_true', default=True,
                        help='whether to employ weight-guided similarity to calculate anomaly map.')
    parser.add_argument('--default', default=0.3, type=float, help='the default value of weights.')
    parser.add_argument('--alpha', default=0.01, type=float, help='hyperparameters for weights.')
    parser.add_argument('--beta', default=0.00003, type=float, help='hyperparameters for weights.')

    parser.add_argument("--train_and_test_all", action='store_true', default=True,
                        help="for medical domains.")
    parser.add_argument("--is_saved", action='store_true', default=True, help="whether to save model weights.")
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--load_ckpts', action='store_true', default=True, help="loading ckpts for testing")

    args = parser.parse_args()

    # for k, v in vars(args).items():
    #     setattr(c, k, v)

    return args


if __name__ == '__main__':
    setup_seed(1203)
    c = parsing_args()
    if not c.weighted_decision_mechanism:
        c.default = c.alpha = c.beta = c.gamma = "w/o"

    domain = c.domain
    dataset_name = c.dataset
    logger = get_logger(dataset_name, os.path.join(c.save_dir, dataset_name))
    # print_fn = logger.info

    dataset = None
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

    elif dataset_name in SupervisedAD:
        dataset = [dataset_name]

    else:
        raise KeyError(f"Dataset '{dataset_name}' can not be found.")

    from tabulate import tabulate

    results = {}
    table_ls = []

    # ---------------------------------------------------------------------------------------------------------
    # --------------------------------------unsupervised industrial AD-----------------------------------------
    # ---------------------------------------------------------------------------------------------------------
    if dataset_name in UnsupervisedAD and domain == 'industrial':
        image_auroc_list = []
        pixel_auroc_list = []
        pixel_aupro_list = []

        # -----------------------------train-------------------------------------
        if not c.load_ckpts:
            for idx, i in enumerate(dataset):
                c._class_ = i
                if dataset_name == 'MVTec 3D-AD':
                    c = preprocess_for_3d(c)

                args_dict = vars(c)
                args_info = f"class:{i}, " if c.setting == 'oc' else f""
                for key, value in args_dict.items():
                    if key in ['_class_']:
                        continue
                    args_info += ", ".join([f"{key}:{value}, "])

                if c.setting == 'oc':
                    print('training on {} dataset (separate-class)'.format(dataset_name)) if idx == 0 else None
                    print(args_info)
                    train(c)
                elif c.setting == 'mc':
                    print('training on {} dataset (multiclass)'.format(dataset_name)) if idx == 0 else None
                    print(args_info)
                    from train_multiclass import train_mc
                    train_mc(c)
                    break
                else:
                    pass
                print()
            print("training over!")

        # -----------------------------test-------------------------------------
        if c.setting == 'oc':
            for idx, i in enumerate(dataset):
                print('testing on {} dataset (separate-class)'.format(dataset_name)) if idx == 0 else None
                c._class_ = i
                print(f"testing class:{i}")
                auroc_sp, auroc_px, aupro_px = test(c, 'BEST_P_PRO')
                print('')
                table_ls.append(['{}'.format(i), str(np.round(auroc_sp, decimals=1)),
                                 str(np.round(auroc_px, decimals=1)),
                                 str(np.round(aupro_px, decimals=1))])
                image_auroc_list.append(auroc_sp)
                pixel_auroc_list.append(auroc_px)
                pixel_aupro_list.append(aupro_px)
                results = tabulate(table_ls, headers=['object', 'image_auroc', 'pixel_auroc', 'pixel_aupro'],
                                   tablefmt="pipe")
            table_ls.append(['mean', str(np.round(np.mean(image_auroc_list), decimals=2)),
                             str(np.round(np.mean(pixel_auroc_list), decimals=2)),
                             str(np.round(np.mean(pixel_aupro_list), decimals=2))])
            results = tabulate(table_ls, headers=['object', 'image_auroc', 'pixel_auroc', 'pixel_aupro'],
                               tablefmt="pipe")
            print(results)

        elif c.setting == 'mc':
            print('testing on {} dataset (multiclass)'.format(dataset_name))

            c._class_ = ''
            auroc_list, acc_list, f1_list, class_list = test(c, 'BEST_I_ROC')
            print('')

            for i, x, y, z in zip(class_list, auroc_list, acc_list, f1_list):
                table_ls.append(['{}'.format(i), str(np.round(x, decimals=1)),
                                 str(np.round(y, decimals=1)),
                                 str(np.round(z, decimals=1))])
                results = tabulate(table_ls, headers=['object', 'image_auroc', 'acc', 'f1'], tablefmt="pipe")
            table_ls.append(['mean', str(np.round(np.mean(auroc_list), decimals=2)),
                             str(np.round(np.mean(acc_list), decimals=2)),
                             str(np.round(np.mean(f1_list), decimals=2))])
            results = tabulate(table_ls, headers=['object', 'image_auroc', 'acc', 'f1'],
                               tablefmt="pipe")
            print(results)

    # ---------------------------------------------------------------------------------------------------
    # -------------------------------------unsupervised medical AD---------------------------------------
    # ---------------------------------------------------------------------------------------------------
    if dataset_name in UnsupervisedAD and domain == 'medical':
        if c.train_and_test_all:
            dataset = ["APTOS", "ISIC2018", "OCT2017"]

        image_auroc_list = []
        acc_list = []
        f1_list = []
        # -----------------------------train-------------------------------------
        if not c.load_ckpts:
            for idx, i in enumerate(dataset):
                c._class_ = i
                c.dataset = i

                args_dict = vars(c)
                args_info = ""
                for key, value in args_dict.items():
                    if key in ['_class_']:
                        continue
                    args_info += ", ".join([f"{key}:{value}, "])
                print('training on {} dataset'.format(i))
                print(args_info)
                train(c)
                print()
            print("training over!")

        # -----------------------------test-------------------------------------
        for idx, i in enumerate(dataset):
            print('testing on {} dataset'.format(i))
            c._class_ = i
            c.dataset = i
            auroc_sp, acc, f1 = test(c, 'BEST_I_ROC')
            print('')
            table_ls.append(['{}'.format(i), str(np.round(auroc_sp, decimals=1)),
                             str(np.round(acc, decimals=1)),
                             str(np.round(f1, decimals=1))])
            image_auroc_list.append(auroc_sp)
            acc_list.append(acc)
            f1_list.append(f1)
            results = tabulate(table_ls, headers=['object', 'image_auroc', 'acc', 'f1'], tablefmt="pipe")
        table_ls.append(['mean', str(np.round(np.mean(image_auroc_list), decimals=2)),
                         str(np.round(np.mean(acc_list), decimals=2)),
                         str(np.round(np.mean(f1_list), decimals=2))])
        results = tabulate(table_ls, headers=['object', 'image_auroc', 'acc', 'f1'], tablefmt="pipe")
        print(results)

    # ----------------------------------------------------------------------------------------------------------
    # --------------------------------supervised medical image segmentation-------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    if dataset_name in SupervisedAD and domain == 'medical':
        if c.train_and_test_all:
            dataset = ["Kvasir-SEG", "CVC-ClinicDB", "CVC-ColonDB"]

        mice_list = []
        miou_list = []
        # -----------------------------train-------------------------------------
        if not c.load_ckpts:
            for idx, i in enumerate(dataset):
                c._class_ = i
                c.dataset = i

                args_dict = vars(c)
                args_info = ""
                for key, value in args_dict.items():
                    if key in ['_class_']:
                        continue
                    args_info += ", ".join([f"{key}:{value}, "])
                print('training on {} dataset'.format(i))
                print(args_info)

                from train_polypseg import train_polyp
                train_polyp(c)
                print()
            print("training over!")

        # -----------------------------test-------------------------------------
        for idx, i in enumerate(dataset):
            print('testing on {} dataset'.format(i))
            c._class_ = i
            c.dataset = i

            mice, miou = test(c, 'BEST_DICE')

            table_ls.append(['{}'.format(i), str(np.round(mice, decimals=1)), str(np.round(miou, decimals=1))])
            mice_list.append(mice)
            miou_list.append(miou)
            results = tabulate(table_ls, headers=['object', 'mice', 'miou'], tablefmt="pipe")
        table_ls.append(['mean', str(np.round(np.mean(mice_list), decimals=2)),
                         str(np.round(np.mean(miou_list), decimals=2))])
        results = tabulate(table_ls, headers=['object', 'mice', 'miou'], tablefmt="pipe")
        print(results)

    # -----------------------------------------------------------------------------------------------
    # ------------------------------------unsupervised video AD--------------------------------------
    # -----------------------------------------------------------------------------------------------
    if dataset_name in SupervisedAD and domain == 'industrial':
        image_auroc_list = []
        acc_list = []
        f1_list = []
        for idx, i in enumerate(dataset):
            c._class_ = i

            args_dict = vars(c)
            args_info = ""
            for key, value in args_dict.items():
                if key in ['_class_']:
                    continue
                args_info += ", ".join([f"{key}:{value}, "])
            print('training on {} dataset'.format(dataset_name)) if idx == 0 else None
            print(args_info)

            from train_vad import train
            auroc_sp, acc, f1 = train(c)
            table_ls.append(['mean', str(np.round(auroc_sp, decimals=1)),
                             str(np.round(acc, decimals=1)),
                             str(np.round(f1, decimals=1))])
            image_auroc_list.append(auroc_sp)
            acc_list.append(acc)
            f1_list.append(f1)
            results = tabulate(table_ls, headers=['object', 'image_auroc', 'acc', 'f1'], tablefmt="pipe")
        print(results)
