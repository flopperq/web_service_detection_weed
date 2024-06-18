import os
from pathlib import Path

import yaml
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def make_dirs(dir='new_dir/'):
    # создаем папки
    dir = Path(dir)
    for p in dir, dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True)  # создаем папку
    return dir

def convert_weedcoco_json(json_dir=''):
    save_dir = make_dirs(dir=f'{json_dir}')  # output папку
    print()

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob('*.json')):
        with open(json_file) as f:
            data = json.load(f)
        images = {'%g' % x['id']: x for x in data['images']}
        # создаем аннотации
        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)

        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
            img = images['%g' % img_id]
            h, w, f = img['height'], img['width'], img['file_name']

            bboxes = []
            for ann in anns:
                #  COCO формат границ
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:  # если w <= 0 и h <= 0
                    continue

                cls = ann['category_id']  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Запись
            label_path = (Path(json_dir) / 'labels' / Path(f).name).with_suffix('.txt')
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_path, 'a') as file:
                for i in range(len(bboxes)):
                    line = *(bboxes[i]),
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')

    # Сохранение dataset.yaml
    names = [data['categories'][i]['name'].split(': ')[1] for i in range(len(data['categories']))]
    d = {'path': json_dir,
         'train': 'images/train',
         'val': 'images/train',
         'test': 'images/train',
         'nc': len(names),
         'names': names}  # dictionary

    with open(f"{json_dir}/weedcoco.yaml", 'w') as f:
        yaml.dump(d, f, sort_keys=False)

    print('\nweedCOCO to YOLO conversion completed successfully!')
