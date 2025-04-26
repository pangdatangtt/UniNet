import logging
import os
import random
from os.path import join

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
import tifffile as tiff
from functools import partial


def to_device(all_models, device):
    to_models = []
    for i in all_models:
        i.to(device)
        to_models.append(i)
    return to_models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def save_weights(modules_list: list, ckpt_path, suffix):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    state = {"tt": None,
             "bn": None,
             "st": None,
             "dfs": None
             }
    for (module, (key, value)) in zip(modules_list, state.items()):
        if module is None:
            continue
        module.cpu()
        state[str(key)] = module.state_dict()
        module.cuda()

    torch.save(state, join(ckpt_path, f"{suffix}.pth"))
    print("modules have been saved to {}".format(join(ckpt_path, f"{suffix}.pth")))


def load_weights(modules_list: list, ckpt_path, suffix):
    print("Loading weights from {}".format(join(ckpt_path, f"{suffix}.pth")))

    state_dict = torch.load(join(ckpt_path, f"{suffix}.pth"))
    new_state = {"tt": None,
                 "bn": None,
                 "st": None,
                 "dfs": None
                 }
    for (module, (key, value)) in zip(modules_list, new_state.items()):
        if module is None:
            continue
        module.load_state_dict(state_dict[str(key)])
        module.eval()
        module.cuda()
        new_state[str(key)] = module

    return new_state  # dict{TeacherTwo, bn, decoder}


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


class utils_3D:

    def organized_pc_to_unorganized_pc(organized_pc):
        return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

    def read_tiff_organized_pc(path):
        tiff_img = tiff.imread(path)
        return tiff_img

    def resize_organized_pc(organized_pc, img_size, tensor_out=True):  # 224
        torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0)
        torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=img_size,
                                                                     mode='nearest')
        if tensor_out:
            return torch_resized_organized_pc.squeeze(dim=0)
        else:
            return torch_resized_organized_pc.squeeze().permute(1, 2, 0).numpy()

    def organized_pc_to_depth_map(organized_pc):
        return organized_pc[:, :, 2]
