import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from pycocotools.coco import COCO

def show_image(
        title: str,
        image,
        delay: int = 0
        ) -> None:
    cv2.imshow(title, image)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def show_DACSDC_img(
        dir_name: str,
        img_name: str,
        show_true_bbox: bool = False
        ) -> None:
    image = get_DACSDC_img(dir_name, img_name, show_true_bbox)
    show_image(dir_name + img_name, image)


def get_DACSDC_img(
        dir_name: str,
        img_name: str,
        show_true_bbox: bool = False
        ):
    image_path = rf'detection\data\DAC-SDC\{dir_name}\{img_name}.jpg'
    image = cv2.imread(image_path)

    if (show_true_bbox):
        bbox = get_DACSDC_true_bbox(dir_name, img_name)
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        color = (0, 255, 0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image


def get_DACSDC_true_bbox(
        dir_name: str,
        img_name: str
        ) -> tuple[int, int, int, int] | None:
    anno_path = rf'detection\data\DAC-SDC\{dir_name}\{img_name}.xml'
    bbox_root = ET.parse(anno_path).getroot().find('object/bndbox')

    bbox = [bbox_root.find('xmin').text, bbox_root.find('ymin').text, bbox_root.find('xmax').text, bbox_root.find('ymax').text]

    return tuple(map(int, bbox))


def show_COCO_img(
        img_num: int,
        show_true_bbox: bool = False
        ) -> None:
    image = get_COCO_img(img_num, show_true_bbox)
    show_image(str(img_num), image)


def get_COCO_img(
        img_num: int,
        show_true_bbox: bool = False
        ):
    image_path = rf'detection\data\COCO\val2017\{img_num:012}.jpg'
    image = cv2.imread(image_path)

    if (show_true_bbox):
        bboxs = get_COCO_true_bboxs(img_num)

        for bbox in bboxs:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            color = (0, 255, 0)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image


def get_COCO_true_bboxs(img_num: int) -> list[tuple[int, int, int, int]]:
    anno_path = rf'detection\data\COCO\anno\instances_val2017.json'
    coco = COCO(anno_path)
    annotation_ids = coco.getAnnIds(imgIds=img_num)
    anns = coco.loadAnns(annotation_ids)

    bboxs = []

    for a in anns:
        bboxs.append(tuple(map(int, list(a['bbox']))))
    
    return bboxs


def show_VisDrone_img(
        set_name: str,
        frame_name: str,
        show_true_bbox: bool = False
        ) -> None:
    image = get_VisDrone_img(set_name, frame_name, show_true_bbox)
    show_image(frame_name, image)


def get_VisDrone_img(
        set_name: str,
        img_name: str,
        show_true_bbox: bool = False
        ):
    image_path = rf'detection\data\VisDrone\{set_name}\images\{img_name}.jpg'
    image = cv2.imread(image_path)

    if (show_true_bbox):
        bboxs = get_VisDrone_true_bboxs(set_name, img_name)

        for bbox in bboxs:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            color = (0, 255, 0)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image


def get_VisDrone_true_bboxs(
        set_name: str,
        img_name: str
        ) -> list[tuple[int, int, int, int]]:
    anno_path = rf'detection\data\VisDrone\{set_name}\anns\{img_name}.txt'

    with open(anno_path, 'r') as fp:
        anns = fp.readlines()

    bboxs = []

    for ann in anns:
        bboxs.append(tuple(map(int, ann.split(',')[:4])))
    
    return bboxs


def show_six_imgs_coco():
    img_nums = [139, 285, 724, 785, 20247, 29596]
    imgs = [cv2.cvtColor(get_COCO_img(img_num, True), cv2.COLOR_BGR2RGB) for img_num in img_nums]

    fig = plt.figure(figsize=(6, 6))

    for i in range(len(img_nums)):
        fig.add_subplot(2, 3, i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.title(str(img_nums[i]))

    plt.show()


def show_six_imgs_visdrone():
    img_nums = ['0000006_00159_d_0000001', '0000006_05208_d_0000014', '0000054_00786_d_0000001', '0000074_16210_d_0000032', '9999944_00000_d_0000002', '9999973_00000_d_0000014']
    imgs = [cv2.cvtColor(get_VisDrone_img('test', img_num, True), cv2.COLOR_BGR2RGB) for img_num in img_nums]

    fig = plt.figure(figsize=(6, 6))

    for i in range(len(img_nums)):
        fig.add_subplot(2, 3, i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')

    plt.show()


def show_train_result(sgd_csv_path, adam_csv_path):
    sgd_result = pd.read_csv(sgd_csv_path)
    adam_result = pd.read_csv(adam_csv_path)

    # fig, ax = plt.subplots()

    # ax.plot(sgd_result['    metrics/mAP50-95(B)'][50:], label='СГС с импульсом')
    # ax.plot(adam_result['    metrics/mAP50-95(B)'][50:], label='Adam')
    
    # ax.legend()
    # ax.set_title('mAP50-95')

    # plt.show()

    figure, axis = plt.subplots(1, 2)

    axis[0].plot(sgd_result['       metrics/mAP50(B)'][50:], label='СГС с импульсом')
    axis[0].plot(adam_result['       metrics/mAP50(B)'][50:], label='Adam')
    axis[0].legend()
    axis[0].set_title('mAP50')
    axis[0].set_xlabel('Номер эпохи')
    axis[0].set_ylabel('Значение метрики')

    axis[1].plot(sgd_result['    metrics/mAP50-95(B)'][50:], label='СГС с импульсом')
    axis[1].plot(adam_result['    metrics/mAP50-95(B)'][50:], label='Adam')
    axis[1].legend()
    axis[1].set_title('mAP50-95')
    axis[1].set_xlabel('Номер эпохи')
    axis[1].set_ylabel('Значение метрики')

    plt.show()

    figure, axis = plt.subplots(1, 3)

    axis[0].plot(sgd_result['           val/box_loss'][50:], label='СГС с импульсом')
    axis[0].plot(adam_result['           val/box_loss'][50:], label='Adam')
    axis[0].legend()
    axis[0].set_title('Box loss')
    axis[0].set_xlabel('Номер эпохи')
    axis[0].set_ylabel('Значение функции потерь')

    axis[1].plot(sgd_result['           val/cls_loss'][50:], label='СГС с импульсом')
    axis[1].plot(adam_result['           val/cls_loss'][50:], label='Adam')
    axis[1].legend()
    axis[1].set_title('Class loss')
    axis[1].set_xlabel('Номер эпохи')
    axis[1].set_ylabel('Значение функции потерь')

    axis[2].plot(sgd_result['           val/dfl_loss'][50:], label='СГС с импульсом')
    axis[2].plot(adam_result['           val/dfl_loss'][50:], label='Adam')
    axis[2].legend()
    axis[2].set_title('DFL loss')
    axis[2].set_xlabel('Номер эпохи')
    axis[2].set_ylabel('Значение функции потерь')

    plt.show()


def show_dect_drone():
    img1_path = rf'detection\data\exmpls\e1.PNG'
    img2_path = rf'detection\data\exmpls\e2.PNG'

    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(1, 2))

    fig.add_subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis('off')

    fig.add_subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis('off')

    plt.show()
