import math
import time

import matplotlib.patches as pch
import matplotlib.pyplot as plt
import torch

from iou import IouLoss
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
                 n_points=None,
                 major_cases=False,
                 target_boxes_area=1 / 32,
                 anchor_boxes_areas=[1 / 32, 1 / 24, 3 / 64, 1 / 16, 1 / 12, 3 / 32, 1 / 8],
                 aspect_ratios=[1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4]):
    ''' loss_fcn: BBR losses used in simulation experiment
        plot_points: Display the anchor point distribution map
        n_points: The number of randomly generated anchors
        major_cases: Only the main cases in the regression process are addressed
        target_boxes_areas: The area of the target box
        anchor_boxes_areas: Area of anchor boxes
        aspect_ratios: Aspect ratio of bounding boxes'''
    iloss = IouLoss(1, 1, ltype='IoU', monotonous=None)

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
                              eval_fcn=lambda x: iloss(x.detach(), target).mean(),
                              max_iter=max_iter, prefix=loss_fcn.__name__)
    loss = iloss(result, target).mean(dim=(1, 2))
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
    iloss = IouLoss(1, 1, ltype='IoU', monotonous=None)
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
                              eval_fcn=lambda x: iloss(x, target[i]),
                              max_iter=epoch, patience=None,
                              prefix=fcn.__name__, title=not any([i, j]))[0]
            res = pch.Rectangle(result[:2], *(result[2:] - result[:2]),
                                facecolor=color, alpha=0.5, label=f'{fcn.__name__} {epoch} epochs')
            res.set_zorder(-j)
            fig.add_patch(res)
        plt.legend(frameon=False)
        plt.tight_layout()
    plt.show()


def plot_gain(alpha_delta_imean=[[1.9, 3, .2],
                                 [1.7, 2.7, .2],
                                 [1.6, 2.5, .2]],
              colors=[pink, blue, yellow, orange]):
    fig = plt.subplot()
    for key in 'right', 'top':
        fig.spines[key].set_color('None')
    for key in 'left', 'bottom':
        fig.spines[key].set_position(('data', 0))
    # The outlier degree of bounding box
    assert len(alpha_delta_imean) <= len(colors)
    liou = torch.linspace(0, 1, 100)
    for adi, c in zip(alpha_delta_imean, colors):
        iloss = IouLoss(ltype='IoU', monotonous=False)
        IouLoss.alpha, IouLoss.delta = adi[:2]
        iloss.iou_mean.mul_(adi[-1])
        plt.plot(liou, iloss._scaled_loss(torch.ones_like(liou), iou=liou), color=c,
                 label=r'$\overline{\mathcal{L}_{IoU}}=%.1f$' % adi[-1])
    plt.plot(liou, torch.ones_like(liou), color='gray', linestyle='--', alpha=0.7)
    # Sets the format of the axes
    plt.xlabel('IoU loss')
    plt.ylabel('gradient gain')
    plt.yticks(*[[0.5, 1, 1.5] * 2])
    plt.ylim([0, 1.5])
    plt.legend(frameon=False)
    plt.show()


if __name__ == '__main__':
    class FcnGetter:
        def __getattr__(self, item):
            return IouLoss(n=1, t=40, ltype=item, monotonous=None)


    f = FcnGetter()
    # 0: Plot the bounding box regression loss in the simulation experiment
    # 1: Visualize regression cases of simulation experiment
    # 2: Visualize the trajectory of regression cases under the effect of WIoU loss and SIoU loss
    # 3: Plot the relationship between the gradient multiplier r and the outlier degree β
    plt.rcParams['figure.figsize'] = [4.0, 3.0]
    command = [lambda: plot_loss([f.WIoU, f.SIoU, f.CIoU, f.DIoU, f.GIoU], n_points=500, major_cases=False),
               lambda: simulate_exp(f.WIoU, plot_points=True, major_cases=True),
               lambda: visualize_track({f.WIoU: 100, f.SIoU: 120, f.CIoU: 250, f.DIoU: 250},
                                       colors=[purple, blue, green, pink]),
               lambda: plot_gain()]

    # command[-1]()

    DEFAULT = [IouLoss.alpha, IouLoss.delta]
    plot_gain(alpha_delta_imean=[[*DEFAULT, .5],
                                 [*DEFAULT, .4],
                                 [*DEFAULT, .3],
                                 [*DEFAULT, .2]],
              colors=[pink, blue, yellow, orange])
