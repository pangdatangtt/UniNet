import torch
import torch.nn.functional as F


def losses(b, a, T, margin, λ=0.7, mask=None, stop_gradient=False):
    """
    b: List of teacher features
    a: List of student features
    mask: Binary mask, where 0 for normal and 1 for abnormal
    T: Temperature coefficient
    margin: Hyperparameter for controlling the boundary
    λ: Hyperparameter for balancing loss
    """

    loss = 0.0
    margin_loss_n = 0.0
    margin_loss_a = 0.0
    contra_loss = 0.0
    for i in range(len(a)):
        s_ = a[i]
        t_ = b[i].detach() if stop_gradient else b[i]

        n, c, h, w = s_.shape

        s = s_.view(n, c, -1).transpose(1, 2)  # (N, H*W, C)
        t = t_.view(n, c, -1).transpose(1, 2)  # (N, H*W, C)

        s_norm = F.normalize(s, p=2, dim=2)
        t_norm = F.normalize(t, p=2, dim=2)

        cos_loss = 1 - F.cosine_similarity(s_norm, t_norm, dim=2)
        cos_loss = cos_loss.mean()

        simi = torch.matmul(s_norm, t_norm.transpose(1, 2)) / T
        simi = torch.exp(simi)
        simi_sum = simi.sum(dim=2, keepdim=True)
        simi = simi / (simi_sum + 1e-8)
        diag_sim = torch.diagonal(simi, dim1=1, dim2=2)

        # unsupervised and only normal (or abnormal)
        if mask is None:
            contra_loss = -torch.log(diag_sim + 1e-8).mean()
            margin_loss_n = F.relu(margin - diag_sim).mean()

        # supervised
        else:
            # gt label
            if len(mask.size()) < 3:
                normal_mask = (mask == 0)
                abnormal_mask = (mask == 1)
            # gt mask
            else:
                mask_ = F.interpolate(mask, size=(h, w), mode='nearest').squeeze(1)
                mask_flat = mask.view(mask_.size(0), -1)

                normal_mask = (mask_flat == 0)
                abnormal_mask = (mask_flat == 1)

            if normal_mask.sum() > 0:
                diag_sim_normal = diag_sim[normal_mask]
                contra_loss = -torch.log(diag_sim_normal + 1e-8).mean()
                margin_loss_n = F.relu(margin - diag_sim_normal).mean()
            if abnormal_mask.sum() > 0:
                diag_sim_abnormal = diag_sim[abnormal_mask]
                margin_loss_a = F.relu(diag_sim_abnormal - margin / 2).mean()

        margin_loss = margin_loss_n + margin_loss_a

        loss += cos_loss * λ + contra_loss * (1 - λ) + margin_loss

    return loss


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

if __name__ == '__main__':
    a = torch.tensor([[0], [0], [0]])
    print(len(a.size()))
