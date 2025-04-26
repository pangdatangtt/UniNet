import torch
import torch.nn as nn
from .Loss import losses
import numpy as np


class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_pro = None
        self.early_stop = False
        self.pro_min = 0.0

    def __call__(self, pro):
        current_pro = pro

        if self.best_pro is None:
            self.best_pro = current_pro
        elif current_pro < self.best_pro + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_pro = current_pro
            self.save_checkpoint(current_pro)
            self.counter = 0

    def save_checkpoint(self, pro):
        """保存当前最佳模型"""
        if self.verbose:
            print(f'PRO increases ({self.pro_min:.1f} --> {pro:.1f}). Saving model...')
        self.pro_min = pro


class UniNet(nn.Module):
    def __init__(self, c, Source_teacher, Target_teacher, bottleneck, student, DFS=None):
        super().__init__()
        self._class_ = c._class_
        self.T = c.T
        self.n = 1 if Target_teacher is None else 2
        self.t = Teachers(Source_teacher=Source_teacher, Target_teacher=Target_teacher)
        self.bn = BN(bottleneck)
        self.s = Student(student=student)
        self.dfs = DFS

    def train_or_eval(self, type='train'):
        self.type = type
        self.t.train_eval(type)
        self.bn.train_eval(type)
        self.s.train_eval(type)

        return self

    def feature_selection(self, b, a, max):
        if self._class_ in ['transistor']:
            return a

        if self.dfs is not None:
            selected_features = self.dfs(a, b, learnable=True, conv=False, max=max)
        else:
            from .DFS import domain_related_feature_selection
            selected_features = domain_related_feature_selection(a, b, max=max)
        return selected_features

    def loss_computation(self, b, a, margin=1, mask=None, stop_gradient=False):
        T = 0.1 if self._class_ in ['transistor', 'pill', 'cable', 'bottle', "grid", 'foam'] else self.T
        loss = losses(b, a, T, margin, mask=mask, stop_gradient=stop_gradient)

        return loss

    def forward(self, x, max=True, mask=None, stop_gradient=False):
        Sou_Tar_features, bnins = self.t(x)
        bnsout = self.bn(bnins)
        stu_features = self.s(bnsout)

        stu_features = [d.chunk(dim=0, chunks=2) for d in stu_features]
        stu_features = [stu_features[0][0], stu_features[1][0], stu_features[2][0],
                        stu_features[0][1], stu_features[1][1], stu_features[2][1]]

        if self.type == 'train':
            stu_features_ = self.feature_selection(Sou_Tar_features, stu_features, max)
            loss = self.loss_computation(Sou_Tar_features, stu_features_, mask=mask, stop_gradient=stop_gradient)

            return loss
        else:
            return Sou_Tar_features, stu_features


class Teachers(nn.Module):
    def __init__(self, Source_teacher, Target_teacher):
        super().__init__()
        self.t_s = Source_teacher
        self.t_t = Target_teacher

    def train_eval(self, type='train'):
        self.type = type
        self.t_s.eval()
        if self.t_t is not None:
            if type == "train":
                self.t_t.train()
            else:
                self.t_t.eval()

        return self

    def forward(self, x):
        with torch.no_grad():
            Sou_features = self.t_s(x)

        if self.t_t is None:
            return Sou_features
        else:
            Tar_features = self.t_t(x)
            bnins = [torch.cat([a, b], dim=0) for a, b in zip(Tar_features, Sou_features)]  # 512, 1024, 2048

            return Sou_features + Tar_features, bnins


class BN(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
        self.bn = bottleneck

    def train_eval(self, type='train'):
        if type == 'train':
            self.bn.train()
        else:
            self.bn.eval()

        return self

    def forward(self, x):
        bns = self.bn(x)

        return bns


class Student(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.s1 = student

    def train_eval(self, type='train'):
        if type == 'train':
            self.s1.train()
        else:
            self.s1.eval()

        return self

    def forward(self, bn_outs, skips=None):
        de_features = self.s1(bn_outs)

        return de_features
