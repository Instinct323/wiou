import math
from pathlib import Path

import matplotlib.patches as pch
import matplotlib.pyplot as plt
import torch

from optimize import minimize

red = 'orangered'
orange = 'orange'
yellow = 'yellow'
green = 'greenyellow'
cyan = 'aqua'
blue = 'deepskyblue'
purple = 'mediumpurple'
pink = 'violet'

plt.rcParams['figure.dpi'] = 150

COLORS = [purple, blue, cyan, pink, green, yellow, orange]


def xywh_to_ltrb(attr):
    attr[..., :2] -= attr[..., 2: 4] / 2
    attr[..., 2: 4] += attr[..., :2]
    return attr


def ltrb_to_xywh(attr):
    attr[..., 2: 4] -= attr[..., :2]
    attr[..., :2] += attr[..., 2: 4] / 2
    return attr


class IoU_Cal:
    ''' pred, target: x0,y0,x1,y1 坐标
        _momentum: exp=1 / n_batch'''
    iou_mean = 1.
    _momentum = 1 - pow(0.6, exp=1 / 40)
    _is_train = True

    def __init__(self, pred, target):
        self.pred, self.target = pred, target
        self._fget = {
            # x,y,w,h 坐标
            'pred_xy': lambda: (self.pred[..., :2] + self.pred[..., 2: 4]) / 2,
            'pred_wh': lambda: self.pred[..., 2: 4] - self.pred[..., :2],
            'target_xy': lambda: (self.target[..., :2] + self.target[..., 2: 4]) / 2,
            'target_wh': lambda: self.target[..., 2: 4] - self.target[..., :2],
            # x0,y0,x1,y1 坐标
            'min_coord': lambda: torch.minimum(self.pred[..., :4], self.target[..., :4]),
            'max_coord': lambda: torch.maximum(self.pred[..., :4], self.target[..., :4]),
            # 重叠区域
            'wh_inter': lambda: self.min_coord[..., 2: 4] - self.max_coord[..., :2],
            's_inter': lambda: torch.prod(torch.relu(self.wh_inter), dim=-1),
            # 覆盖区域
            's_union': lambda: torch.prod(self.pred_wh, dim=-1) +
                               torch.prod(self.target_wh, dim=-1) - self.s_inter,
            # 最小包围框
            'wh_box': lambda: self.max_coord[..., 2: 4] - self.min_coord[..., :2],
            's_box': lambda: torch.prod(self.wh_box, dim=-1),
            'l2_box': lambda: torch.square(self.wh_box).sum(dim=-1),
            # 框的中心连线
            'd_center': lambda: self.pred_xy - self.target_xy,
            'l2_center': lambda: torch.square(self.d_center).sum(dim=-1),
            # IoU
            'iou': lambda: 1 - self.s_inter / self.s_union
        }
        self._update(self)

    def __setitem__(self, key, value):
        self._fget[key] = value

    def __getattr__(self, item):
        # 获取不存在的属性时的行为
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
        ''' 更新 IoU 损失的均值'''
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def IoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        return self.iou

    @classmethod
    def WIoU_v1(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        punish = torch.exp(torch.relu(- self.wh_inter).sum(dim=-1))
        return punish * self.iou

    @classmethod
    def WIoU_v2(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        dist = torch.exp(self.l2_center / self.l2_box.detach())
        return dist * self.iou

    @classmethod
    def WIoU_v3(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        dist = torch.exp(self.l2_center / self.l2_box.detach())
        return dist * self.iou * (self.iou.detach() / cls.iou_mean)

    @classmethod
    def WIoU_v4(cls, pred, target, gamma=1.4, delta=5, self=None):
        self = self if self else cls(pred, target)
        beta = self.iou.detach() / cls.iou_mean
        alpha = delta * torch.pow(gamma, beta - delta)
        dist = torch.exp(self.l2_center / self.l2_box.detach())
        return dist * self.iou * (beta / alpha)

    @classmethod
    def WIoU(cls, pred, target, self=None):
        return cls.WIoU_v4(pred, target, self=self)

    @classmethod
    def GIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        return self.iou + (self.s_box - self.s_union) / self.s_box

    @classmethod
    def DIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        return self.iou + self.l2_center / self.l2_box

    @classmethod
    def CIoU(cls, pred, target, eps=1e-4, self=None):
        self = self if self else cls(pred, target)
        # 计算纵横比一致性
        v = 4 / math.pi ** 2 * \
            (torch.atan(self.pred_wh[..., 0] / (self.pred_wh[..., 1] + eps)) -
             torch.atan(self.target_wh[..., 0] / (self.target_wh[..., 1] + eps))) ** 2
        alpha = v / (self.iou + v)
        return self.iou + self.l2_center / self.l2_box + alpha.detach() * v

    @classmethod
    def EIoU(cls, pred, target, self=None):
        self = self if self else cls(pred, target)
        # factor = self.iou.sqrt()
        return self.iou + self.l2_center / self.l2_box + \
               torch.square(self.d_center / self.wh_box).sum(dim=-1)

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
        return self.iou + (dist + shape) / 2


def simulate_exp(loss_fcn, lr=.01, max_iter=120,
                 plot_points=False,
                 file_mode='Project/%s.jpg',
                 n_points=None,
                 major_cases=False,
                 target_boxes_area=1 / 32,
                 anchor_boxes_areas=[1 / 32, 1 / 24, 3 / 64, 1 / 16, 1 / 12, 3 / 32, 1 / 8],
                 aspect_ratios=[1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]):
    ''' loss_fcn: 仿真实验使用的 IoU 损失函数
        plot_points: 展示生成的随机散点
        file_mode: IoU 损失热力图的存储路径
        n_points: 生成随机散点的个数
        major_cases: 只针对回归过程中的主要情况
        target_boxes_areas: 目标框的面积
        anchor_boxes_areas: 初始锚框的面积
        aspect_ratios: 初始锚框的纵横比 / 目标框的纵横比'''
    IoU = IoU_Cal.IoU
    IoU_Cal.iou_mean = 1.
    aspect_ratios = torch.tensor(aspect_ratios)
    anchor_boxes_areas = torch.tensor(anchor_boxes_areas)
    # 回归案例的模式
    points_radius = 0.1 if major_cases else 0.5
    max_iter = max_iter // 2 if major_cases else max_iter
    n_points = n_points if n_points else int(2e4 * points_radius ** 2)
    # 在 r=3 的圆内生成均匀分布的点, 坐标需要变换到 [0, 1]
    rho = torch.log(torch.rand(n_points) * (math.pow(math.e, 3) - 1) + 1) / 3 * points_radius
    theta = torch.rand(n_points) * math.tau
    x = torch.cos(theta) * rho + 0.5
    y = torch.sin(theta) * rho + 0.5
    # 在每一个锚点生成 7×7 个锚箱
    width = (anchor_boxes_areas[:, None] / aspect_ratios).sqrt()
    height = aspect_ratios * width
    width, height = map(torch.flatten, [width, height])
    # 拼接得到全体锚箱
    xy = torch.stack([x, y], dim=-1)
    wh = torch.stack([width, height], dim=-1)
    anchor = torch.cat([xy[:, None].repeat(1, len(width), 1),
                        wh[None].repeat(len(x), 1, 1)], dim=-1)[..., None, :]
    # 获取目标框
    target_w = (target_boxes_area / aspect_ratios).sqrt()
    target_h = target_w * aspect_ratios
    target = torch.cat([torch.ones([len(aspect_ratios), 2]) / 2,
                        target_w[:, None], target_h[:, None]], dim=-1)
    anchor, target = map(xywh_to_ltrb, [anchor, target])
    # 绘制散点图
    if plot_points:
        fig = plt.subplot()
        plt.scatter(x, y, s=0.3, color=blue)
        for axis in 'xy': getattr(plt, f'{axis}lim')([-0.05, 1.05])
        for l, t, r, b in target:
            rect = pch.Rectangle((l, t), (r - l), (b - t), alpha=0.2, facecolor=purple)
            fig.add_patch(rect)
        plt.show()
    # 构建损失函数, 并进行迭代求解
    anchor = anchor.repeat(1, 1, len(aspect_ratios), 1)
    result, _, log = minimize(anchor, lambda x: loss_fcn(x, target).mean(), lr=lr,
                              eval_fcn=lambda x: IoU(x.detach(), target).mean(),
                              max_iter=max_iter, prefix=loss_fcn.__name__)
    loss = IoU(result, target).mean(dim=(1, 2))
    loss_fcn = loss_fcn.__name__
    print(f'{loss_fcn}: Mean IoU = {1 - loss.mean():.3f}, Min IoU = {1 - loss.max():.3f}')
    # 绘制 loss 热力图
    fig = plt.subplot(projection='3d')
    plt.title(loss_fcn)
    fig.set_xlabel('x')
    fig.set_ylabel('y')
    fig.set_zlabel('IoU')
    fig.view_init(40, 30)
    fig.scatter(x, y, loss, cmap=plt.get_cmap('rainbow'), c=(loss - loss.min()) / (loss.max() - loss.min()))
    # 保存绘制的图像
    if file_mode:
        folder = Path(file_mode).parent
        folder.mkdir(exist_ok=True)
        plt.savefig(file_mode % loss_fcn)
        plt.close()
    else:
        plt.show()
    return {loss_fcn: log}


def plot_loss(fcn_list, **simlate_kwargs):
    ''' 绘制 IoU 损失下降曲线
        fcn_list: 参与测试的损失函数列表
        simlate_kwargs: simulate_exp 函数的关键字参数'''
    assert len(COLORS) >= len(fcn_list), 'Insufficient amount of color provided'
    log_dict = {}
    for fcn in fcn_list:
        log_dict.update(simulate_exp(fcn, **simlate_kwargs))
    # 绘制损失下降曲线
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    for color, fcn in zip(COLORS, log_dict):
        log = log_dict[fcn]
        x = torch.arange(1, len(log) + 1)
        plt.plot(x, log, label=fcn, color=color)
    plt.legend(frameon=False)
    plt.show()


def visualize_track(fcn_and_epoch: dict, lr=.01):
    ''' 可视化锚盒回归过程
        fcn_and_epoch: {fcn: epoch ...}'''
    assert len(COLORS) >= len(fcn_and_epoch), 'Insufficient amount of color provided'
    IoU = IoU_Cal.IoU
    anchor = xywh_to_ltrb(torch.tensor([[.7, .7, .2, .4],
                                        [.5, .8, .6, .1]]))
    target = xywh_to_ltrb(torch.tensor([[.2, .2, .05, .1],
                                        [.5, .1, .05, .05]]))
    # 将键值对进行统一
    for fcn in fcn_and_epoch:
        epoch = fcn_and_epoch[fcn]
        if isinstance(epoch, int): fcn_and_epoch[fcn] = [epoch] * 2
        assert len(fcn_and_epoch[fcn]) == 2
    # 使用梯度下降法仿真回归过程
    for i in range(2):
        fig = plt.subplot(1, 2, i + 1)
        for f in [plt.xlim, plt.ylim]: f([0, 1])
        for f in [plt.xticks, plt.yticks]: f([])
        # for loc in ['top', 'bottom', 'left', 'right']: fig.spines[loc].set_color('None')
        # 绘制锚框、真实框
        anc = pch.Rectangle(anchor[i][:2], *(anchor[i][2:] - anchor[i][:2]),
                            edgecolor=orange, fill=False, label='Inital')
        anc.set_zorder(2)
        tar = pch.Rectangle(target[i][:2], *(target[i][2:] - target[i][:2]),
                            edgecolor=red, fill=False, label='GT')
        tar.set_zorder(2)
        for p in [anc, tar]: fig.add_patch(p)
        # 绘制迭代过程中的锚框
        for j, (color, fcn) in enumerate(zip(COLORS, fcn_and_epoch)):
            epoch = fcn_and_epoch[fcn][i]
            result = minimize(anchor[i].clone(), lambda x: fcn(x, target[i]), lr=lr,
                              eval_fcn=lambda x: IoU(x.detach(), target[i]),
                              max_iter=epoch, patience=None,
                              prefix=fcn.__name__, title=not any([i, j]))[0]
            res = pch.Rectangle(result[:2], *(result[2:] - result[:2]),
                                facecolor=color, alpha=0.5, label=f'{fcn.__name__} {epoch} epoch')
            res.set_zorder(1 - j)
            fig.add_patch(res)
        plt.legend(frameon=False)
        plt.tight_layout()
    plt.show()


def plot_gain(gamma=[2.5, 1.9, 1.6, 1.4], delta=[2, 3, 4, 5],
              colors=[pink, blue, yellow, orange]):
    ''' 绘制动态聚焦机制'''
    fig = plt.subplot()
    for key in 'right', 'top':
        fig.spines[key].set_color('None')
    for key in 'left', 'bottom':
        fig.spines[key].set_position(('data', 0))
    # 边界框离群程度
    beta = torch.linspace(0, 5, 100)
    for g, d, c in zip(gamma, delta, colors):
        alpha = d * torch.pow(g, beta - d)
        plt.plot(beta, beta / alpha, color=c, label=f'α={g}, δ={d}')
    # 设置坐标轴
    plt.xlabel('β')
    plt.ylabel('r')
    plt.ylim([0, 1.8])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    f = IoU_Cal

    # 0: Plot the bounding box loss of the unfocused mechanism in the simulation experiment IoU curves
    # 1: Plot the IoU curves of WIoU loss versus SIoU loss in the simulation experiment
    # 2: Visualize regression cases of simulation experiments as well as IoU curves
    # 3: Visualize the trajectory of regression cases under the effect of WIoU loss and SIoU loss
    # 4: Plot the relationship between the gradient multiplier r and the degree of outlier β of the prediction box
    command = [lambda: plot_loss([f.WIoU_v2, f.WIoU_v1, f.SIoU, f.EIoU,
                                  f.CIoU, f.DIoU, f.GIoU], n_points=None, major_cases=False),
               lambda: plot_loss([f.WIoU_v4, f.WIoU_v2, f.SIoU], n_points=500, major_cases=False),
               lambda: simulate_exp(f.WIoU_v4, file_mode='', plot_points=True, major_cases=True),
               lambda: visualize_track({f.CIoU: 250, f.DIoU: 250}),
               lambda: plot_gain()]

    command[-1]()