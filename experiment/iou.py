import math
import time
from pathlib import Path

import matplotlib.patches as pch
import matplotlib.pyplot as plt
import torch
from optimize import minimize

red = 'orangered'
orange = 'darkorange'
yellow = 'gold'
green = 'greenyellow'
cyan = 'aqua'
blue = 'deepskyblue'
purple = 'mediumpurple'
pink = 'violet'

plt.rcParams['figure.dpi'] = 150

COLORS = [purple, blue, green, yellow, orange]


class IoU_Cal:
    ''' pred, target: x0,y0,x1,y1
        monotonous: {
            None: origin
            True: monotonic FM
            False: non-monotonic FM
        }
        momentum: The momentum of running mean'''
    iou_mean = 1.
    monotonous = False
    momentum = 1 - pow(0.5, exp=1 / 10)
    _is_train = True

    def __init__(self, pred, target):
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
        self._update(self)

    def __setitem__(self, key, value):
        self._fget[key] = value

    def __getattr__(self, item):
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]

    @classmethod
    def train(cls):
        cls._is_train = True

    @classmethod
    def eval(cls):
        cls._is_train = False

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls.momentum) * cls.iou_mean + \
                                         cls.momentum * self.iou.detach().mean().item()

    def _scaled_loss(self, loss, alpha=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            beta = self.iou.detach() / self.iou_mean
            if self.monotonous:
                loss *= beta.sqrt()
            else:
                divisor = delta * torch.pow(alpha, beta - delta)
                loss *= beta / divisor
        return loss

    @classmethod
    def IoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        return self.iou

    @classmethod
    def WIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        dist = torch.exp(self.l2_center / self.l2_box.detach())
        return self._scaled_loss(dist * self.iou)

    @classmethod
    def EIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        penalty = self.l2_center / self.l2_box.detach() \
                  + torch.square(self.d_center / self.wh_box.detach()).sum(dim=-1)
        return self._scaled_loss(self.iou + penalty)

    @classmethod
    def GIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        return self._scaled_loss(self.iou + (self.s_box - self.s_union) / self.s_box)

    @classmethod
    def DIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        return self._scaled_loss(self.iou + self.l2_center / self.l2_box)

    @classmethod
    def CIoU(cls, pred, target, eps=1e-4, self=None):
        self = self if self else cls(pred, target)
        v = 4 / math.pi ** 2 * \
            (torch.atan(self.pred_wh[..., 0] / (self.pred_wh[..., 1] + eps)) -
             torch.atan(self.target_wh[..., 0] / (self.target_wh[..., 1] + eps))) ** 2
        alpha = v / (self.iou + v)
        return self._scaled_loss(self.iou + self.l2_center / self.l2_box + alpha.detach() * v)

    @classmethod
    def SIoU(cls, pred, target, theta=4, self=None):
        self = self if self else cls(pred, target)
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
        return self._scaled_loss(self.iou + (dist + shape) / 2)


def run_once(func):
    def handler(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as reason:
                print(reason)
                time.sleep(20)
                continue

    return handler


def xywh_to_ltrb(attr):
    attr[..., :2] -= attr[..., 2: 4] / 2
    attr[..., 2: 4] += attr[..., :2]
    return attr


def scatter_circle(n, radius, dot=[0., 0.], alpha=3):
    ''' Generate scatter uniformly in a circular area'''
    rho = torch.log(torch.rand(n) * (math.exp(alpha) - 1) + 1) / alpha * radius
    theta = torch.rand(n) * math.tau
    x = torch.cos(theta) * rho + dot[0]
    y = torch.sin(theta) * rho + dot[1]
    return x, y


@run_once
def simulate_exp(loss_fcn, lr=.01, max_iter=120,
                 plot_points=False,
                 file_mode='Project/%s.jpg',
                 n_points=None,
                 major_cases=False,
                 target_boxes_area=1 / 32,
                 anchor_boxes_areas=[1 / 32, 1 / 24, 3 / 64, 1 / 16, 1 / 12, 3 / 32, 1 / 8],
                 aspect_ratios=[1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]):
    ''' loss_fcn: BBR losses used in simulation experiment
        plot_points: Display the anchor point distribution map
        file_mode: The storage path of the heat map
        n_points: The number of randomly generated anchors
        major_cases: Only the main cases in the regression process are addressed
        target_boxes_areas: The area of the target box
        anchor_boxes_areas: Area of anchor boxes
        aspect_ratios: Aspect ratio of bounding boxes'''
    IoU = IoU_Cal.IoU
    IoU_Cal.iou_mean = 1.
    aspect_ratios = torch.tensor(aspect_ratios)
    anchor_boxes_areas = torch.tensor(anchor_boxes_areas)
    # The distribution pattern of the regression cases
    points_radius = 0.1 if major_cases else 0.5
    max_iter = max_iter // 2 if major_cases else max_iter
    n_points = n_points if n_points else int(2e4 * points_radius ** 2)
    # The coordinates need to be transformed to [0, 1]
    x, y = scatter_circle(n_points, radius=points_radius, dot=[.5, .5])
    # 7*7 anchor boxes are generated at each anchor point
    width = (anchor_boxes_areas[:, None] / aspect_ratios).sqrt()
    height = aspect_ratios * width
    width, height = map(torch.flatten, [width, height])
    # Splice and get all anchor boxes
    xy = torch.stack([x, y], dim=-1)
    wh = torch.stack([width, height], dim=-1)
    anchor = torch.cat([xy[:, None].repeat(1, len(width), 1),
                        wh[None].repeat(len(x), 1, 1)], dim=-1)[..., None, :]
    # Get the target box
    target_w = (target_boxes_area / aspect_ratios).sqrt()
    target_h = target_w * aspect_ratios
    target = torch.cat([torch.full([len(aspect_ratios), 2], 0.5),
                        target_w[:, None], target_h[:, None]], dim=-1)
    anchor, target = map(xywh_to_ltrb, [anchor, target])
    anchor = anchor.repeat(1, 1, len(aspect_ratios), 1)
    # Draw the anchor point distribution map
    if plot_points:
        fig = plt.subplot()
        plt.scatter(x, y, s=0.3, color=blue)
        for axis in 'xy': getattr(plt, f'{axis}lim')([-0.05, 1.05])
        for l, t, r, b in target:
            rect = pch.Rectangle((l, t), (r - l), (b - t), alpha=0.2, facecolor=purple)
            fig.add_patch(rect)
        plt.show()
    # Construct the loss function and solve it using the function <minimize>
    result, _, log = minimize(anchor.detach(), lambda x: loss_fcn(x, target).mean(), lr=lr,
                              eval_fcn=lambda x: IoU(x.detach(), target).mean(),
                              max_iter=max_iter, prefix=loss_fcn.__name__)
    loss = IoU(result, target).mean(dim=(1, 2))
    loss_fcn = loss_fcn.__name__
    print(f'{loss_fcn}: Mean IoU = {1 - loss.mean():.3f}, Min IoU = {1 - loss.max():.3f}')
    # Draw the heat map of the IoU loss
    # fig = plt.subplot(projection='3d')
    # plt.title(loss_fcn)
    # fig.set_xlabel('x')
    # fig.set_ylabel('y')
    # fig.set_zlabel('IoU')
    # fig.view_init(40, 30)
    # fig.scatter(x, y, loss, cmap=plt.get_cmap('rainbow'), c=(loss - loss.min()) / (loss.max() - loss.min()))
    # Save the drawn image
    return {loss_fcn: log}


def plot_loss(fcn_list, **simlate_kwargs):
    ''' Draw the IoU loss curve
        fcn_list: List of loss functions participating in the test
        simlate_kwargs: The keyword argument of function <simulate_exp>'''
    assert len(COLORS) >= len(fcn_list), 'Insufficient amount of color provided'
    log_dict = {}
    for fcn in fcn_list:
        log_dict.update(simulate_exp(fcn, **simlate_kwargs))
    fig = plt.subplot()
    for key in 'right', 'top':
        fig.spines[key].set_color('None')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    for color, fcn in zip(COLORS, log_dict):
        log = log_dict[fcn]
        x = torch.arange(1, len(log) + 1)
        plt.plot(x, log, label=fcn, color=color)
    plt.legend(frameon=False)
    plt.show()


def visualize_track(fcn_and_epoch: dict, lr=.01, colors=COLORS):
    ''' Visual bounding box regression
        fcn_and_epoch: {fcn: epoch ...}'''
    assert len(colors) >= len(fcn_and_epoch), 'Insufficient amount of color provided'
    IoU = IoU_Cal.IoU
    anchor = xywh_to_ltrb(torch.tensor([[.7, .7, .2, .4],
                                        [.5, .8, .6, .1]]))
    target = xywh_to_ltrb(torch.tensor([[.2, .2, .05, .1],
                                        [.5, .1, .05, .05]]))
    # Fixed the format of key-value pairs
    for fcn in fcn_and_epoch:
        epoch = fcn_and_epoch[fcn]
        if isinstance(epoch, int): fcn_and_epoch[fcn] = [epoch] * 2
        assert len(fcn_and_epoch[fcn]) == 2
    # The BBR is simulated using a gradient descent algorithm
    for i in range(2):
        fig = plt.subplot(1, 2, i + 1)
        for f in [plt.xlim, plt.ylim]: f([0, 1])
        for f in [plt.xticks, plt.yticks]: f([])
        # for loc in ['top', 'bottom', 'left', 'right']: fig.spines[loc].set_color('None')
        # Draw anchor boxes and target boxes
        anc = pch.Rectangle(anchor[i][:2], *(anchor[i][2:] - anchor[i][:2]),
                            edgecolor=orange, fill=False, label='Inital')
        anc.set_zorder(1)
        tar = pch.Rectangle(target[i][:2], *(target[i][2:] - target[i][:2]),
                            edgecolor=red, fill=False, label='GT')
        tar.set_zorder(1)
        for p in [anc, tar]: fig.add_patch(p)
        # Draws the anchor box in the optimization
        for j, (color, fcn) in enumerate(zip(colors, fcn_and_epoch)):
            epoch = fcn_and_epoch[fcn][i]
            result = minimize(anchor[i].clone(), lambda x: fcn(x, target[i]), lr=lr,
                              eval_fcn=lambda x: IoU(x.detach(), target[i]),
                              max_iter=epoch, patience=None,
                              prefix=fcn.__name__, title=not any([i, j]))[0]
            res = pch.Rectangle(result[:2], *(result[2:] - result[:2]),
                                facecolor=color, alpha=0.5, label=f'{fcn.__name__} {epoch} epochs')
            res.set_zorder(-j)
            fig.add_patch(res)
        plt.legend(frameon=False)
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [4.0, 3.0]
    f = IoU_Cal
    f.monotonous = None

    # 0: Plot the bounding box loss in the simulation experiment IoU curves
    # 1: Visualize regression cases of simulation experiments as well as IoU curves
    # 2: Visualize the trajectory of regression cases under the effect of WIoU loss and SIoU loss
    command = [lambda: plot_loss([f.WIoU, f.SIoU, f.CIoU, f.DIoU, f.GIoU], n_points=500, major_cases=False),
               lambda: simulate_exp(f.WIoU, file_mode='', plot_points=True, major_cases=True),
               lambda: visualize_track({f.WIoU: 100, f.SIoU: 120, f.CIoU: 250, f.DIoU: 250},
                                       colors=[purple, blue, green, pink])]

    command[0]()
