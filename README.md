# Project Branches

- **v1:**
  The code [(Usage)](https://blog.csdn.net/qq_55745968/article/details/128888122) used in the experiment detailed in the corresponding [paper](https://arxiv.org/abs/2301.10051).
  Experimental data is located in this branch only.
  - dataset: `MS COCO 2017` (categories: 20, train: 28474, val: 1219)
  - citation:
    ```
    @article{tong2023wise,
      title={Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism},
      author={Tong, Zanjia and Chen, Yuhang and Xu, Zewei and Yu, Rong},
      journal={arXiv preprint arXiv:2301.10051},
      year={2023}
    }
    ```
    
- **v2 (master):**
  The refactored version of branch `v1`, integrating advanced features of `nn.Module` into the `IouLoss`.
