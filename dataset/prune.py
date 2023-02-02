from pathlib import Path
from tqdm import tqdm


def prune_dataset(image_dir, label_dir, cls_pool=None, min_n_boxes=1):
    ''' YOLOv7 数据集裁剪
        cls_pool: 保留的类别
        min_n_boxes: 每张图片的最少边界框数'''
    if cls_pool: cls_pool = {cls: i for i, cls in enumerate(
        range(cls_pool) if isinstance(cls_pool, int) else cls_pool)}
    # 读取图像, 这将会清除没有对应 txt 的图像
    for image_folder in filter(lambda p: p.is_dir(), image_dir.iterdir()):
        unlink_count = 0
        label_folder = label_dir / image_folder.stem
        # 创建进度条
        pbar = tqdm(list(image_folder.iterdir()))
        for image in pbar:
            label = label_folder / (image.stem + '.txt')
            temp = label.with_suffix('.tmp')
            unlink_flag = False
            # 读取标签文件
            if label.is_file():
                with open(label) as f:
                    bboxes = f.readlines()
                # 筛除标签
                if cls_pool: bboxes = list(filter(
                    lambda bbox: int(bbox.split()[0]) in cls_pool.keys(), bboxes))
                if len(bboxes) >= min_n_boxes:
                    # 写入临时标签
                    if cls_pool:
                        with open(temp, 'w') as f:
                            for bbox in bboxes:
                                attr = bbox.split()
                                attr[0] = str(cls_pool[int(attr[0])])
                                f.write(' '.join(attr) + '\n')
                else:
                    unlink_flag = True
            else:
                # 标签文件为空
                unlink_flag = True
            if unlink_flag:
                # 删除标签文件、图像
                for file in (image, label): file.unlink(missing_ok=True)
                # 统计已删除的数据量
                unlink_count += 1
            prune_rate = unlink_count / len(pbar) * 100
            pbar.set_description(f'{image_folder.stem} Pruning Rate {prune_rate:.2f} %')
    # 临时文件覆盖原文件
    temp_files = list(label_dir.glob('**/*.tmp'))
    if temp_files:
        input('Type anything to start rewriting the label')
        for temp in tqdm(temp_files, desc='Overwrite Labels'):
            txt = temp.with_suffix('.txt')
            txt.unlink(missing_ok=True)
            temp.rename(txt)


def make_index(image_dir):
    ''' 为 YOLOv7 数据集制作索引文本'''
    for folder in filter(lambda p: p.is_dir(), image_dir.iterdir()):
        txt = image_dir.parent / (folder.stem + '.txt')
        with open(txt, 'w') as f:
            for file in tqdm(list(folder.iterdir()), desc=txt.name):
                file = './' + str(file).replace('\\', '/')
                f.write(f'{file}\n')


prune_dataset(Path('images'), Path('labels'), cls_pool=range(1, 21), min_n_boxes=2)
make_index(Path('images'))
