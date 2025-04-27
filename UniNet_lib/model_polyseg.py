import torch
import torch.nn as nn
from .Loss import losses, structure_loss
import torch.nn.functional as F


class UniNet(nn.Module):
    def __init__(self, c, Source_teacher, Target_teacher, bottleneck, student, DFS=None):
        super().__init__()
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

    def feature_selection(self, b, a, max=True):
        if self.dfs is not None:
            selected_features = self.dfs(a, b, learnable=True, conv=False, max=max)
        else:
            from .DFS import domain_related_feature_selection
            selected_features = domain_related_feature_selection(a, b)
        return selected_features

    def loss_computation(self, b, a, pred_list, margin=1, mask=None, stop_gradient=False):
        loss = losses(b, a, self.T, margin, mask=None, stop_gradient=stop_gradient) + \
               structure_loss(pred_list[0][0], mask) + structure_loss(pred_list[0][-1], mask)

        return loss

    def forward(self, x, mask=None, max=True, stop_gradient=False):
        Sou_Tar_features, bnins = self.t(x)
        bnsout = self.bn(bnins)
        stu_features, stu_pred = self.s(bnsout, [bnins[-1], bnins[-2]])

        stu_features = [d.chunk(dim=0, chunks=2) for d in stu_features]
        stu_features = [stu_features[0][0], stu_features[1][0], stu_features[2][0],
                        stu_features[0][1], stu_features[1][1], stu_features[2][1]]
        stu_pred1 = F.interpolate(stu_pred[0], scale_factor=4, mode='bilinear')
        # stu_pred2 = F.interpolate(stu_pred[1], scale_factor=8, mode='bilinear')
        # stu_pred3 = F.interpolate(stu_pred[-1], scale_factor=16, mode='bilinear')

        stu_pred1 = stu_pred1.chunk(dim=0, chunks=2)
        # stu_pred2 = stu_pred2.chunk(dim=0, chunks=2)
        # stu_pred3 = stu_pred3.chunk(dim=0, chunks=2)

        if self.type == 'train':
            stu_features_ = self.feature_selection(Sou_Tar_features, stu_features, max)
            loss = self.loss_computation(Sou_Tar_features, stu_features_, [stu_pred1], mask=mask, 
                                         stop_gradient=stop_gradient)
            return loss
        else:
            return Sou_Tar_features, stu_features, [stu_pred1]


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
        de_features = self.s1(bn_outs, skips)

        return de_features
