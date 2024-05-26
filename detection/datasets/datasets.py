import os

from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json


def creat_groundndtruth_files_COCO() -> None:
    labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', \
              'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', \
              'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', \
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', \
              'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', \
              'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', \
              'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', \
              'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

    img_dir_path = rf'detection\data\COCO\val2017'
    anno_path = rf'detection\data\COCO\anno\instances_val2017.json'
    coco = COCO(anno_path)

    imgs = list(map(lambda i: int(i[:12]), os.listdir(img_dir_path)))

    for img in imgs:
        annotation_ids = coco.getAnnIds(imgIds=img)
        anns = coco.loadAnns(annotation_ids)

        text = []

        for a in anns:
            label = labels[a['category_id']].replace(' ', '_')
            l, t, w, h = list(a['bbox'])
            text.append(f'{label} {int(l)} {int(t)} {int(l + w)} {int(t + h)}' + '\n')

        file_path = rf'detection\results\COCO\gold' + rf'\{img:012}.txt'

        with open(file_path, 'w') as fp:
            fp.writelines(text)


def convert_dataset_to_coco_style_json(dir: str = rf'detection\data\VisDrone\test') -> None:
    # category_id_to_name = {
    #     0: 'ignore',
    #     1: 'pedestrian',
    #     2: 'people',
    #     3: 'bicycle',
    #     4: 'car',
    #     5: 'van',
    #     6: 'truck',
    #     7: 'tricycle',
    #     8: 'awning-tricycle',
    #     9: 'bus',
    #     10: 'motor',
    #     11: 'others',
    #     12: 'obj'
    # }

    category_id_to_name = {
        0: 'pedestrian',
        1: 'people',
        2: 'bicycle',
        3: 'car',
        4: 'van',
        5: 'truck',
        6: 'tricycle',
        7: 'awning-tricycle',
        8: 'bus',
        9: 'motor',
    }
    
    img_dir = dir + rf'\images'
    ann_dir = dir + rf'\anns'

    coco = Coco()

    for category_id, category_name in category_id_to_name.items():
        coco.add_category(CocoCategory(id=category_id, name=category_name))
    
    img_file_names = os.listdir(img_dir)

    for img_file_name in tqdm(img_file_names):
        img_path = rf'{img_dir}\{img_file_name}'
        width, height = Image.open(img_path).size

        coco_image = CocoImage(file_name=img_file_name, height=height, width=width)

        only_img_file_name = img_file_name.split('.')[0]
        ann_path = rf'{ann_dir}\{only_img_file_name}.txt'

        with open(ann_path, 'r') as fp:
            lines = fp.readlines()

        for line in lines:
            x, y, w, h, _, c, _, _ = line.split(',')

            if int(c) != 0 and int(c) != 11:
                coco_image.add_annotation(
                    CocoAnnotation(
                        bbox=[int(x), int(y), int(w), int(h)],
                        category_id=int(c),
                        category_name=category_id_to_name[int(c) - 1]
                        )
                    )

        coco.add_image(coco_image)

    save_json(data=coco.json, save_path='detection/results/VisDrone/gold_test.json')


def count_bbox_dist_visdrone():
    paths = [rf'detection\data\VisDrone\test\anns', rf'detection\data\VisDrone\train\anns', rf'detection\data\VisDrone\val\anns']

    result = [0] * 12

    for path in paths:
        file_names = os.listdir(path)

        for file_name in file_names:
            with open(rf'{path}\{file_name}', 'r') as fp:
                lines = fp.readlines()

                for line in lines:
                    result[int(line.split(',')[5])] += 1
    
    print(result)
    
    result = result[1:-1]

    print(result)
    print(sum(result))

    fig, ax = plt.subplots()

    l = ['Пешеход', 'Человек', 'Велосипед', 'Автомобиль', 'Фургон', 'Грузовик', 'Трицикл', 'Трицикл с тентом', 'Автобус', 'Мотоцикл']

    ax.bar(l, result)

    ax.bar_label(ax.containers[0], fontsize=10)
    ax.set_ylabel('Количество ограничительных рамок')
    ax.set_xlabel('Названия классов')
    ax.set_title('Распределение ограничительных рамок по классам объектов')

    plt.show()


def count_area_bbox_dist_visdrone():
    paths = [rf'detection\data\VisDrone\test\anns', rf'detection\data\VisDrone\train\anns', rf'detection\data\VisDrone\val\anns']

    result = [0] * 3

    for path in paths:
        file_names = os.listdir(path)

        for file_name in file_names:
            with open(rf'{path}\{file_name}', 'r') as fp:
                lines = fp.readlines()

                for line in lines:
                    x, y, w, h, _, c, _, _ = line.split(',')[:8]
                    
                    if int(c) != 0 and int(c) != 11:
                        area = int(w) * int(h)

                        if area < 32 ** 2:
                            result[0] += 1

                        elif area > 96 ** 2:
                            result[2] += 1

                        else:
                            result[1] += 1
    
    print(result)
    print(sum(result))
    
    fig, ax = plt.subplots()

    l = ['Площадь меньше 1024', 'Площадь между 1024 и 9216', 'Площадь больше 9216']

    ax.bar(l, result)

    ax.bar_label(ax.containers[0], fontsize=10)
    ax.set_ylabel('Количество ограничительных рамок')
    ax.set_xlabel('Значение площади ограничительной рамки')
    ax.set_title('Распределение ограничительных рамок по их площадям')

    plt.show()
