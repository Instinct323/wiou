import math

import torch
from torch import nn


class IouLoss:
    ''' n: Number of batches per training epoch
        t: The epoch when mAP's ascension slowed significantly
        monotonous: {
            None: origin
            True: monotonic FM
            False: non-monotonic FM
        }'''

    def __init__(self, n, t, ltype='WIoU', monotonous=False, alpha=1.9, delta=3):
        # The momentum of running mean
        time_to_real = n * t
        self.momentum = 1 - pow(0.05, 1 / time_to_real)

        self.ltype = ltype
        self.monotonous = monotonous

        self.alpha = alpha
        self.delta = delta

        self.iou_mean = nn.Parameter(torch.tensor(1.), requires_grad=False)
        self._is_train = isinstance(self.monotonous, bool)

    def __setitem__(self, key, value):
        self._fget[key] = value

    def __getattr__(self, item):
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]

    def __call__(self, pred, target, ret_iou=False, **kwargs):
        # pred, target: x0,y0,x1,y1
        self.pred, self.target = pred, target
        self._fget = {
            # x,y,w,h
            'pred_xy': lambda: (self.pred[..., :2] + self.pred[..., 2: 4]) / 2,
            'pred_wh': lambda: self.pred[..., 2: 4] - self.pred[..., :2],
            'target_xy': lambda: (self.target[..., :2] + self.target[..., 2: 4]) / 2,
            'target_wh': lambda: self.target[..., 2: 4] - self.target[..., :2],
            # x0,y0,x1,y1
            'min_coord': lambda: torch.minimum(self.pred[..., :4], self.target[..., :4]),
            'max_coord': lambda: torch.maximum(self.pred[..., :4], self.target[..., :4]),
            # The overlapping region
            'wh_inter': lambda: torch.relu(self.min_coord[..., 2: 4] - self.max_coord[..., :2]),
            's_inter': lambda: torch.prod(self.wh_inter, dim=-1),
            # The area covered
            's_union': lambda: torch.prod(self.pred_wh, dim=-1) +
                               torch.prod(self.target_wh, dim=-1) - self.s_inter,
            # The smallest enclosing box
            'wh_box': lambda: self.max_coord[..., 2: 4] - self.min_coord[..., :2],
            's_box': lambda: torch.prod(self.wh_box, dim=-1),
            'l2_box': lambda: torch.square(self.wh_box).sum(dim=-1),
            # The central points' connection of the bounding boxes
            'd_center': lambda: self.pred_xy - self.target_xy,
            'l2_center': lambda: torch.square(self.d_center).sum(dim=-1),
            # IoU
            'iou': lambda: 1 - self.s_inter / self.s_union
        }
        if self._is_train:
            self.iou_mean.mul_(1 - self.momentum)
            self.iou_mean.add_(self.momentum * self.iou.detach().mean().item())

        loss = self._scaled_loss(getattr(self, self.ltype)(**kwargs))
        return (loss, self.iou) if ret_iou else loss

    def train(self):
        self._is_train = True

    def eval(self):
        self._is_train = False

    def _scaled_loss(self, loss):
        if isinstance(self.monotonous, bool):
            beta = self.iou.detach() / self.iou_mean
            if self.monotonous:
                loss *= beta.sqrt()
            else:
                divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
                loss *= beta / divisor
        return loss

    def IoU(self):
        return self.iou

    def WIoU(self):
        dist = torch.exp(self.l2_center / self.l2_box.detach())
        return dist * self.iou

    def EIoU(self):
        penalty = self.l2_center / self.l2_box \
                  + torch.square(self.d_center / self.wh_box).sum(dim=-1)
        return self.iou + penalty

    def GIoU(self):
        return self.iou + (self.s_box - self.s_union) / self.s_box

    def DIoU(self):
        return self.iou + self.l2_center / self.l2_box

    def CIoU(self, eps=1e-4):
        v = 4 / math.pi ** 2 * \
            (torch.atan(self.pred_wh[..., 0] / (self.pred_wh[..., 1] + eps)) -
             torch.atan(self.target_wh[..., 0] / (self.target_wh[..., 1] + eps))) ** 2
        alpha = v / (self.iou + v)
        return self.iou + self.l2_center / self.l2_box + alpha.detach() * v

    def SIoU(self, theta=4):
        # Angle Cost
        angle = torch.arcsin(torch.abs(self.d_center).min(dim=-1)[0] / (self.l2_center.sqrt() + 1e-4))
        angle = torch.sin(2 * angle) - 2
        # Dist Cost
        dist = angle[..., None] * torch.square(self.d_center / self.wh_box)
        dist = 2 - torch.exp(dist[..., 0]) - torch.exp(dist[..., 1])
        # Shape Cost
        d_shape = torch.abs(self.pred_wh - self.target_wh)
        big_shape = torch.maximum(self.pred_wh, self.target_wh)
        w_shape = 1 - torch.exp(- d_shape[..., 0] / big_shape[..., 0])
        h_shape = 1 - torch.exp(- d_shape[..., 1] / big_shape[..., 1])
        shape = w_shape ** theta + h_shape ** theta
        return self.iou + (dist + shape) / 2

    def __repr__(self):
        return f'{self.ltype}(' \
               f'iou_mean={self.iou_mean.item():.3f}, ' \
               f'm={self.momentum:.3e}, ' \
               f'train={self._is_train})'


if __name__ == '__main__':
    def xywh2xyxy(labels, i=0):
        labels = labels.clone()
        labels[..., i:i + 2] -= labels[..., i + 2:i + 4] / 2
        labels[..., i + 2:i + 4] += labels[..., i:i + 2]
        return labels


    torch.manual_seed(0)
    iouloss = IouLoss(890, 34, ltype='IoU')

    for i in range(10000):
        pred, tar = xywh2xyxy(torch.rand([2, 3, 1, 4], requires_grad=True))
        iouloss(pred, tar)
    print(iouloss.iou_mean)
