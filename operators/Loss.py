import torch
import torch.nn as nn
import torch.nn.functional as F
from .iou_utils import bbox_overlaps_iou, bbox_overlaps_ciou, bbox_overlaps_diou, bbox_overlaps_giou, \
    decode


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha, config):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device = config.device

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            0, n_class, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out).clamp(min=1e-4, max=1-1e-4)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        y = (t == class_ids).float()

        loss = -y * alpha * term1 - (1 - y) * (1 - alpha) * term2

        return loss.sum()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.05, dim=-1, reduce=True):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.reduce = reduce

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        if self.reduce:
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        else:
            return torch.sum(-true_dist * pred, dim=self.dim)


class IouLoss(nn.Module):

    def __init__(self, pred_mode='Corner', reduce='mean', variances=None, losstype='Diou'):
        super(IouLoss, self).__init__()
        self.reduce = reduce
        self.pred_mode = pred_mode  # predicted locs should be top-left and bottom-right corner conventions
        self.variances = variances
        self.loss = losstype

    def forward(self, loc_p, loc_t, prior_data=None, weights=None):
        num = loc_p.shape[0]

        if self.pred_mode == 'Center':
            assert prior_data is not None
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        if self.loss == 'Iou':
            loss = 1.0 - bbox_overlaps_iou(decoded_boxes, loc_t)
        else:
            if self.loss == 'Giou':
                loss = 1.0 - bbox_overlaps_giou(decoded_boxes, loc_t)
            else:
                if self.loss == 'Diou':
                    loss = 1.0 - bbox_overlaps_diou(decoded_boxes, loc_t)
                else:
                    loss = 1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t)

        if weights is not None and weights.sum() > 1e-6:
            return (loss * weights).sum() / weights.sum()
        else:
            if self.reduce == 'mean':
                loss = loss.sum() / num
            else:
                loss = loss.sum()

            return loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss"""

    def __init__(self, beta=1.0 / 9.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, weights=None):
        num = pred.size(0)

        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        l1_loss = torch.where(x >= self.beta, l1, l2)

        if weights is not None and weights.sum() > 1e-6:
            assert pred.size(0) == target.size(0) == weights.size(0)
            return (l1_loss * weights).sum() / weights.sum()
        else:
            if self.reduction == 'mean':
                return l1_loss.sum() / num
            else:
                return l1_loss.sum()
