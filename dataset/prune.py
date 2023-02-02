from pathlib import Path
from tqdm import tqdm


def prune_dataset(image_dir, label_dir, cls_pool=None, min_n_boxes=1):
    ''' Pruning of the dataset in YOLOv7 format
        cls_pool: The indexes of the category to keep
        min_n_boxes: Minimum number of bounding boxes per image'''
    if cls_pool: cls_pool = {cls: i for i, cls in enumerate(
        range(cls_pool) if isinstance(cls_pool, int) else cls_pool)}
    # Read the image and clear any image that has no corresponding txt
    for image_folder in filter(lambda p: p.is_dir(), image_dir.iterdir()):
        unlink_count = 0
        label_folder = label_dir / image_folder.stem
        # Creating a progress bar
        pbar = tqdm(list(image_folder.iterdir()))
        for image in pbar:
            label = label_folder / (image.stem + '.txt')
            temp = label.with_suffix('.tmp')
            unlink_flag = False
            # Read tag files
            if label.is_file():
                with open(label) as f:
                    bboxes = f.readlines()
                # Filter out labels
                if cls_pool: bboxes = list(filter(
                    lambda bbox: int(bbox.split()[0]) in cls_pool.keys(), bboxes))
                if len(bboxes) >= min_n_boxes:
                    # Write temporary labels
                    if cls_pool:
                        with open(temp, 'w') as f:
                            for bbox in bboxes:
                                attr = bbox.split()
                                attr[0] = str(cls_pool[int(attr[0])])
                                f.write(' '.join(attr) + '\n')
                else:
                    unlink_flag = True
            else:
                # The tag file is empty
                unlink_flag = True
            if unlink_flag:
                # Delete the tag file, the image
                for file in (image, label): file.unlink(missing_ok=True)
                # Count the amount of deleted data
                unlink_count += 1
            prune_rate = unlink_count / len(pbar) * 100
            pbar.set_description(f'{image_folder.stem} Pruning Rate {prune_rate:.2f} %')
    # Make the temporary file overwrite the original file
    temp_files = list(label_dir.glob('**/*.tmp'))
    if temp_files:
        input('Type anything to start rewriting the label')
        for temp in tqdm(temp_files, desc='Overwrite Labels'):
            txt = temp.with_suffix('.txt')
            txt.unlink(missing_ok=True)
            temp.rename(txt)


def make_index(image_dir):
    ''' Make an index file for the dataset'''
    for folder in filter(lambda p: p.is_dir(), image_dir.iterdir()):
        txt = image_dir.parent / (folder.stem + '.txt')
        with open(txt, 'w') as f:
            for file in tqdm(list(folder.iterdir()), desc=txt.name):
                file = './' + str(file).replace('\\', '/')
                f.write(f'{file}\n')


prune_dataset(Path('images'), Path('labels'), cls_pool=range(1, 21), min_n_boxes=2)
make_index(Path('images'))
