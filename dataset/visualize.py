import os
from pathlib import Path

import cv2
import numpy as np
import random
from tqdm import tqdm


def plot_one_box(x, img, color, label=None, line_thickness=None):
    ''' 绘制边界框'''
    tl = line_thickness or max([round(0.003 * min(img.shape[:2])), 1])  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_color(n_color):
    heat = np.linspace(0, 255, n_color).round().astype(np.uint8)[None]
    return cv2.applyColorMap(heat, cv2.COLORMAP_RAINBOW)[0].tolist()


def parse_label(image_dir, label_dir, detect_dir=None,
                show=False, category=None, color=None):
    ''' image_dir: 原图像目录
        label_dir: 标签文件目录
        detect_dir: 检测结果目录
        category: 类别名称列表
        color: 类别对应颜色'''
    if (detect_dir or show) and not color:
        color = get_color(len(category))
    if detect_dir and not detect_dir.is_dir(): detect_dir.mkdir()
    # 依次取出图像
    for img_file in tqdm(list(image_dir.iterdir())):
        txt = label_dir / img_file.with_suffix('.txt').name
        if txt.is_file():
            img = cv2.imread(str(img_file))
            h, w = img.shape[:2]
            # 读取边界框
            with open(txt) as f:
                # 选定标志
                choose_flag = False
                for cls, *xywh in list(map(lambda s: list(map(eval, s.split())), f.readlines())):
                    xywh, conf = (xywh[:4], xywh[-1]) if len(xywh) == 5 else (xywh, None)
                    xywh = np.array(xywh)
                    # xywh -> xyxy
                    xywh[:2] -= xywh[2:] / 2
                    xywh[2:] += xywh[:2]
                    xywh[::2] *= w
                    xywh[1::2] *= h
                    xyxy = xywh
                    yield img_file.name, cls, xyxy, conf
                    # 绘制边界框
                    if cls in [3, 12]: choose_flag = True
                    if detect_dir or show:
                        plot_one_box(xyxy, img, color=color[cls],
                                     label=category[cls] + (f' {conf:.2f}' if conf else ''))
            # 存储图像
            if detect_dir and choose_flag: cv2.imwrite(str(detect_dir / img_file.name), img)
            if show:
                cv2.imshow(img_name, img)
                cv2.waitKey(0)


names = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant']

for i in parse_label(image_dir=Path('images/train2017'), label_dir=Path('labels/train2017'),
                     detect_dir=Path(os.getenv('Download')) / '__true__',
                     category=names): pass
