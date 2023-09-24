import logging
from typing import Callable, Optional

import torch
from tqdm import tqdm

logging.basicConfig(format='%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


class minimize:
    ''' x: The variable being optimized (torch.tensor)
        loss_fcn: Loss function with x as input
        lr: Learning rate
        patience: Tolerance for no progress in optimization
        eval_fcn: The loss function used for the evaluation
        max_iter: Maximum number of iterations
        prefix: Prefix of the progress bar
        title: Show the title
        return: optimal solution, min loss, log of loss'''

    def __new__(cls,
                x: torch.tensor,
                loss_fcn: Callable,
                lr: float,
                eval_fcn=None,
                patience: Optional[int] = 50,
                max_iter: Optional[int] = None,
                prefix: str = 'Minimize',
                title: bool = True,
                leave: bool = True):
        assert patience or max_iter
        # Initialize the variable
        x.requires_grad = True
        cls.x = x
        cls.optimizer = torch.optim.Adam([x], lr=lr)
        # Record the optimal solution
        cls.min_loss, cls.best_variant, cls.log = float('inf'), None, []
        if title: LOGGER.info(('%10s' * 3) % ('', 'cur_loss', 'min_loss'))
        # Set class properties
        cls.prefix = prefix
        cls.leave = leave
        instance = object.__new__(cls)
        instance.loss_fcn, instance.eval_fcn = loss_fcn, eval_fcn
        instance.main(patience, max_iter)
        return instance.best_variant, instance.min_loss, instance.log

    def main(self, patience, max_iter):
        # Initialize the iteration-dependent parameters
        pbar = tqdm(range(max_iter if max_iter else patience), leave=self.leave)
        angry = 0 if patience else None
        if not max_iter:
            # Greedy method
            while angry != patience:
                is_better = self.update(pbar)
                angry = 0 if is_better else angry + 1
                pbar.reset() if is_better else pbar.update()
        else:
            for _ in pbar:
                is_better = self.update(pbar)
                # Lazy method
                if patience:
                    angry = 0 if is_better else angry + 1
                    if angry == patience: break
        pbar.close()

    def update(self, pbar):
        is_better = False
        # Calculate the loss value and log it
        loss = self.loss_fcn(self.x)
        loss_value = loss.item() if not self.eval_fcn else self.eval_fcn(self.x.detach()).item()
        self.log.append(loss_value)
        # Save the optimal solution
        if loss_value < self.min_loss:
            self.min_loss, self.best_variant = loss_value, self.x.clone().detach()
            is_better = True
        # Back propagate the gradient and update the variable
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        pbar.set_description(('%10s' + '%10.4g' * 2) % (self.prefix, loss_value, self.min_loss))
        return is_better


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = torch.linspace(-3, 3, 50)
    # x + 0.3 x^2 - 0.5 x^3 + 4 sin(x) + noise
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
        return ((y - pred_y) ** 2).sum()


    best_var, min_loss, log = minimize(torch.ones(4), loss_fcn=loss, lr=1e-1, patience=50, max_iter=2000)
    print(best_var)
    plt.plot(x, cal_y(best_var, x), c='orange', label='pred')

    plt.legend()
    plt.show()
