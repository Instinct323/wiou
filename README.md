# Project Branches

## v1

- The code used in the experiment is detailed in the corresponding [paper](https://arxiv.org/abs/2301.10051). You can find additional information on its usage [here](https://blog.csdn.net/qq_55745968/article/details/128888122).
- Experimental data is exclusively stored in this branch.
- dataset: `MS COCO 2017` (categories: 20, train: 28474, val: 1219)

## v2 (master)

- The refactored version of branch `v1`, integrating advanced features of `nn.Module` into the `IouLoss`.
- dataset: `Fire-Smoke` (categories: 2, train: 4320, val: 1330)

# Example

After initializing your model, initialize the `IouLoss` and assign it to the model. The instance attribute `iou_mean` of `IouLoss` will be output to the `state_dict` for saving and loading during the training process.

```python
# After initializing your model, integrate the IoU loss module into the model.
m = YourDetectionModel()
m.iouloss = IouLoss(ltype='WIoU', monotonous=False)
# Ensure that `IouLoss` is assigned to the model before executing `load_state_dict`.
m.load_state_dict(torch.load('last.pt'))
```

Modify the bounding box regression loss in the loss function. Note that the confidence of the bounding box uses $IoU$ instead of $1-\mathcal{L}_{WIoU}$.

```python
m.iouloss.train()
iloss, liou = m.iouloss(xywh2xyxy(pred), xywh2xyxy(gt), ret_iou=True)
# Accumulate the regression loss.
lbox += iloss.mean()
# Adjust the objectness term to use IoU as the bounding box confidence.
tobj = 1 - liou.detach()
```

# Citation

```
@article{tong2023wise,
  title={Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism},
  author={Tong, Zanjia and Chen, Yuhang and Xu, Zewei and Yu, Rong},
  journal={arXiv preprint arXiv:2301.10051},
  year={2023}
}
```
