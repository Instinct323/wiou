import logging
from typing import Callable, Optional

import torch
from tqdm import tqdm

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


class minimize:
    ''' variant: 作为变量的 tensor
        loss_fcn: 以 variant 为输入, loss 为输出的函数
        lr: 学习率
        patience: 允许 loss 无进展的次数
        eval_fcn: 需要记录的损失函数
        max_iter: 最大迭代次数
        prefix: 进度条前缀
        title: 输出标题
        return: 最优变量的 tensor, 最小 loss 值, loss 日志'''

    def __new__(cls,
                variant: torch.tensor,
                loss_fcn: Callable,
                lr: float,
                eval_fcn=None,
                patience: Optional[int] = 50,
                max_iter: Optional[int] = None,
                prefix: str = 'Minimize',
                title: bool = True,
                leave: bool = True):
        assert patience or max_iter
        # 初始化变量
        variant.requires_grad = True
        cls.variant = variant
        cls.optimizer = torch.optim.Adam([variant], lr=lr)
        # 记录最优变量
        cls.min_loss, cls.best_variant, cls.log = float('inf'), None, []
        if title: LOGGER.info(('%10s' * 3) % ('', 'cur_loss', 'min_loss'))
        # 设置类变量
        cls.prefix = prefix
        cls.leave = leave
        instance = object.__new__(cls)
        instance.loss_fcn, instance.eval_fcn = loss_fcn, eval_fcn
        instance.main(patience, max_iter)
        return instance.best_variant, instance.min_loss, instance.log

    def main(self, patience, max_iter):
        # 初始化迭代参数
        pbar = tqdm(range(max_iter if max_iter else patience), leave=self.leave)
        angry = 0 if patience else None
        if not max_iter:
            # 贪心模式
            while angry != patience:
                is_better = self.update(pbar)
                angry = 0 if is_better else angry + 1
                pbar.reset() if is_better else pbar.update()
        else:
            # 指定次数
            for _ in pbar:
                is_better = self.update(pbar)
                # 懒惰模式
                if patience:
                    angry = 0 if is_better else angry + 1
                    if angry == patience: break
        pbar.close()

    def update(self, pbar):
        is_better = False
        # 计算损失值, 记入日志
        loss = self.loss_fcn(self.variant)
        loss_value = loss.item() if not self.eval_fcn else self.eval_fcn(self.variant).item()
        self.log.append(loss_value)
        # 保存更优的变量
        if loss_value < self.min_loss:
            self.min_loss, self.best_variant = loss_value, self.variant.clone().detach()
            is_better = True
        # 反向传播梯度, 更新变量
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        pbar.set_description(('%10s' + '%10.4g' * 2) % (self.prefix, loss_value, self.min_loss))
        return is_better


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 绘制原始值
    x = torch.linspace(-3, 3, 50)
    # 拟合目标: x + 0.3 x^2 - 0.5 x^3 + 4 sin(x) + 噪声
    y = x + 0.3 * x ** 2 - 0.5 * x ** 3 + 4 * torch.sin(x) + 5 * (torch.rand(len(x)) - 0.5)
    plt.scatter(x, y, c='deepskyblue', label='true')


    def cal_y(variant, x):
        # x: [bs, 1] -> [bs, 4]
        # a x + b x^2 + c x^3 + d e^x
        x = torch.stack([x, x ** 2, x ** 3, torch.exp(-x)], dim=1)
        # y: [bs, 4] × [4, ] -> [bs, ]
        y = x @ variant
        return y


    def loss(variant):
        pred_y = cal_y(variant, x)
        # 平方差损失
        return ((y - pred_y) ** 2).sum()


    best_var, min_loss, log = minimize(torch.ones(4), loss_fcn=loss, lr=1e-1, patience=50, max_iter=2000)
    print(best_var)
    # 绘制预测值
    plt.plot(x, cal_y(best_var, x), c='orange', label='pred')

    plt.legend()
    plt.show()