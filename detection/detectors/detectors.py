import os
import time
import PIL

import torch
import pprint
import json

from tqdm import tqdm
from torchvision import transforms
from torchvision.io.image import read_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights, \
                                         fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from ultralytics import YOLO

COCO_categorys = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

VisDrone_name_to_id = {
    'pedestrian': 1,
    'people': 2,
    'bicycle': 3,
    'car': 4,
    'van': 5,
    'truck': 6,
    'tricycle':7,
    'awning-tricycle': 8,
    'bus': 9,
    'motor': 10,
}


def ssd_detection(
        img_path: str,
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        score_thresh: float = 0.6
        ) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ssdlite320_mobilenet_v3_large(weights=weights, score_thresh=score_thresh)
    model.to(device)
    model.eval()

    preprocess = weights.transforms()

    img = read_image(img_path).to(device)

    with torch.no_grad():
        prediction = model([preprocess(img)])[0]

    labels = [weights.meta['categories'][i]  for i in prediction['labels']]
    scores = [f'{i:.2f}' for i in prediction['scores']]

    labels_and_scores = [i + ' ' + j  for i, j in zip(labels, scores)]

    box = draw_bounding_boxes(img, boxes=prediction['boxes'],
                              labels=labels_and_scores,
                              colors='red',
                              width=4)

    result_img = to_pil_image(box.detach())
    result_img.show()


def rcnn_detection(
        img_path: str,
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        score_thresh: float = 0.6
        ) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=score_thresh)
    model.to(device)
    model.eval()

    preprocess = weights.transforms()

    img = read_image(img_path).to(device)

    with torch.no_grad():
        prediction = model([preprocess(img)])[0]

    labels = [weights.meta['categories'][i]  for i in prediction['labels']]
    scores = [f'{i:.2f}' for i in prediction['scores']]

    labels_and_scores = [i + ' ' + j  for i, j in zip(labels, scores)]

    box = draw_bounding_boxes(img, boxes=prediction['boxes'],
                              labels=labels_and_scores,
                              colors='red',
                              width=4)

    result_img = to_pil_image(box.detach())
    result_img.show()


def yolo_detection(
        img_path: str,
        score_thresh: float = 0.6,
        weights_path: str = rf'detection\data\YOLO\yolov8n.pt',
        img_size: int = 640
        ) -> None:
    model = YOLO(weights_path)

    results = model.predict(img_path, conf=score_thresh, imgsz=img_size, verbose=False)
    results[0].show()


def count_fps_sdd(
        path_to_imgs: str = rf'detection\data\COCO\val2017',
        imgs_num: int = 0,
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        score_thresh: float = 0.6,
        device: str = 'cpu'
        ) -> None:
    device = torch.device(device)

    model = ssdlite320_mobilenet_v3_large(weights=weights, score_thresh=score_thresh)
    model.to(device)
    model.eval()

    preprocess = weights.transforms()

    img_names = os.listdir(path_to_imgs)

    if imgs_num:
        img_names = img_names[:imgs_num]

    total_time = 0.0

    for img_name in img_names:
        img = read_image(path_to_imgs + '\\' + img_name).to(device)

        batch = [preprocess(img)]
        
        with torch.no_grad():
            start = time.time()

            model(batch)

            total_time += time.time() - start

    print('=== SDD Lite ===')
    print('Device:', device)
    print('Count of images:', len(img_names))
    print('Total time:', total_time)
    print('Time per image:', total_time / len(img_names))
    print('FPS:', len(img_names) / total_time)


def count_fps_rcnn(
        path_to_imgs: str = rf'detection\data\COCO\val2017',
        imgs_num: int = 0,
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        score_thresh: float = 0.6,
        device: str = 'cpu'
        ) -> None:
    device = torch.device(device)

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, score_thresh=score_thresh)
    model.to(device)
    model.eval()

    preprocess = weights.transforms()

    img_names = os.listdir(path_to_imgs)

    if imgs_num:
        img_names = img_names[:imgs_num]

    total_time = 0.0

    for img_name in img_names:
        img = read_image(path_to_imgs + '\\' + img_name).to(device)

        batch = [preprocess(img)]
        
        with torch.no_grad():
            start = time.time()

            model(batch)
            
            total_time += time.time() - start

    print('=== Faster R-CNN ===')
    print('Device:', device)
    print('Count of images:', len(img_names))
    print('Total time:', total_time)
    print('Time per image:', total_time / len(img_names))
    print('FPS:', len(img_names) / total_time)


def count_fps_yolo(
        path_to_imgs: str = rf'detection\data\COCO\val2017',
        imgs_num: int = 0,
        path_to_weights: str = rf'detection\data\YOLO\yolov8n.pt',
        score_thresh: float = 0.6,
        img_size: int = 640,
        device: str = 'cpu'
        ) -> None:
    device = torch.device(device)

    model = YOLO(path_to_weights)
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(size=(img_size, img_size))
    ])

    img_names = os.listdir(rf'detection\data\COCO\val2017')

    if imgs_num:
        img_names = img_names[:imgs_num]

    total_time = 0.0

    for img_name in img_names:
        img = pil_to_tensor(PIL.Image.open(path_to_imgs + '\\' + img_name).convert('RGB')).float().to(device)
        img = preprocess(torch.div(img, 256))

        batch = img.unsqueeze(0)

        start = time.time()

        model.predict(batch, conf=score_thresh, imgsz=img_size, device=device, verbose=False)

        total_time += time.time() - start

    print('=== YOLOv8 ===')
    print('Device:', device)
    print('Image size:', img_size)
    print('Count of images:', len(img_names))
    print('Total time:', total_time)
    print('Time per image:', total_time / len(img_names))
    print('FPS:', len(img_names) / total_time)


def count_params() -> None:
    ssd = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    ssd_total_params = sum(p.numel() for p in ssd.parameters())

    rcnn = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    rcnn_total_params = sum(p.numel() for p in rcnn.parameters())

    yolo = YOLO(rf'detection\data\YOLO\yolov8n.pt')
    yolo_total_params = sum(p.numel() for p in yolo.parameters())

    print('Total number of parameters:')
    print('SSD Lite:', ssd_total_params)
    print('Faster R-CNN:', rcnn_total_params)
    print('YOLOv8:', yolo_total_params)


def ssd_detect_all_coco(
        conf_thresh: float = 0.001
        ) -> None:
    img_dir_path = rf'detection\data\COCO\val2017'
    imgs = os.listdir(img_dir_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT

    model = ssdlite320_mobilenet_v3_large(weights=weights, score_thresh=conf_thresh)
    model.to(device)
    model.eval()

    preprocess = weights.transforms()

    batch_size = 64
    batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]

    total_result = []

    for name_batch in batches:
        batch = [read_image(img_dir_path + fr'\{name}').to(device) for name in name_batch]

        with torch.no_grad():
            preds = model(list(map(preprocess, batch)))

        img_ids = list(map(lambda i: int(i.split('.')[0]), name_batch))

        for img_id, pred in zip(img_ids, preds):

            for l, c, b in zip(pred['labels'], pred['scores'], pred['boxes']):
                x, y, x2, y2 = b
                total_result.append({'image_id': img_id,
                                     'category_id': int(l),
                                     'bbox': [float(x), float(y), float(x2) - float(x), float(y2) - float(y)],
                                     'score': float(c)})
    
    with open(rf'detection\results\COCO\ssdlite.json', 'w') as fp:
        json.dump(total_result, fp)

    # for name_batch in batches:
    #     batch = [read_image(img_dir_path + fr'\{name}').to(device) for name in name_batch]

    #     with torch.no_grad():
    #         results = model(list(map(preprocess, batch)))

    #     names = list(map(lambda i: i.split('.')[0], name_batch))

    #     for name, result in zip(names, results):
    #         text = []

    #         for l, c, b in zip(result['labels'], result['scores'], result['boxes']):
    #             label = weights.meta['categories'][l].replace(' ', '_')

    #             text.append(f'{label} {c:.6f} {int(b[0])} {int(b[1])} {int(b[2])} {int(b[3])}' + '\n')

    #         file_path = res_dir_path + rf'\{name}.txt'

    #         with open(file_path, 'w') as fp:
    #             fp.writelines(text)


def rcnn_detect_all_coco(
        conf_thresh: float = 0.001
        ) -> None:
    img_dir_path = rf'detection\data\COCO\val2017'
    imgs = os.listdir(img_dir_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, score_thresh=conf_thresh)
    model.to(device)
    model.eval()

    preprocess = weights.transforms()

    batch_size = 64
    batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]

    total_result = []

    for name_batch in batches:
        batch = [read_image(img_dir_path + fr'\{name}').to(device) for name in name_batch]

        with torch.no_grad():
            preds = model(list(map(preprocess, batch)))

        img_ids = list(map(lambda i: int(i.split('.')[0]), name_batch))

        for img_id, pred in zip(img_ids, preds):

            for l, c, b in zip(pred['labels'], pred['scores'], pred['boxes']):
                x, y, x2, y2 = b
                total_result.append({'image_id': img_id,
                                     'category_id': int(l),
                                     'bbox': [float(x), float(y), float(x2) - float(x), float(y2) - float(y)],
                                     'score': float(c)})
    
    with open(rf'detection\results\COCO\fasterrcnn.json', 'w') as fp:
        json.dump(total_result, fp)

    # for name_batch in batches:
    #     batch = [read_image(img_dir_path + fr'\{name}').to(device) for name in name_batch]

    #     with torch.no_grad():
    #         results = model(list(map(preprocess, batch)))

    #     names = list(map(lambda i: i.split('.')[0], name_batch))

    #     for name, result in zip(names, results):
    #         text = []

    #         for l, c, b in zip(result['labels'], result['scores'], result['boxes']):
    #             label = weights.meta['categories'][l].replace(' ', '_')

    #             text.append(f'{label} {c:.6f} {int(b[0])} {int(b[1])} {int(b[2])} {int(b[3])}' + '\n')

    #         file_path = res_dir_path + rf'\{name}.txt'

    #         with open(file_path, 'w') as fp:
    #             fp.writelines(text)


def yolo_detect_all_coco(
        conf_thresh: float = 0.001,
        img_size: int = 640
    ) -> None:
    img_dir_path = rf'detection\data\COCO\val2017'
    imgs = os.listdir(img_dir_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = YOLO(rf'detection\data\YOLO\yolov8n.pt')
    model.to(device)

    batch_size = 64
    batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]

    total_result = []

    for batch in batches:
        preds = model.predict([img_dir_path + fr'\{name}' for name in batch], conf=conf_thresh, imgsz=img_size, device=device, verbose=False)

        img_ids = list(map(lambda i: int(i.split('.')[0]), batch))

        for img_id, pred in zip(img_ids, preds):

            for l, c, b in zip(pred.boxes.cls, pred.boxes.conf, pred.boxes.xyxy):
                for k, v in COCO_categorys.items():
                    if v == pred.names[int(l)]:
                        category_id = k
                        break

                x, y, x2, y2 = b
                total_result.append({'image_id': img_id,
                                     'category_id': category_id,
                                     'bbox': [float(x), float(y), float(x2) - float(x), float(y2) - float(y)],
                                     'score': float(c)})
    
    with open(rf'detection\results\COCO\yolo_{img_size}.json', 'w') as fp:
        json.dump(total_result, fp)

        # for name, result in zip(names, results):
        #     text = []

        #     for l, c, b in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
        #         label = result.names[int(l)].replace(' ', '_')

        #         text.append(f'{label} {float(c):.6f} {int(b[0])} {int(b[1])} {int(b[2])} {int(b[3])}' + '\n')

        #     file_path = res_dir_path + rf'\{name}.txt'

        #     with open(file_path, 'w') as fp:
        #         fp.writelines(text)


def train_yolo(weights_path:str = rf'detection\data\YOLO\yolov8n.pt', ep_num: int = 100, opt_name: str = 'SGD') -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = YOLO(weights_path)
    model.to(device)
    results = model.train(data='detection\data\VisDrone\VisDrone.yaml', epochs=ep_num, imgsz=640, batch=8, device=device,
                          workers=4, optimizer=opt_name)
    
    pprint.pp(results, width=20)


def yolo_detect_all_visdrone(
        path_to_weight: str,
        conf_thresh: float = 0.001,
        img_size: int = 640
    ) -> None:
    img_dir_path = rf'detection\data\VisDrone\test\images'
    imgs = os.listdir(img_dir_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = YOLO(path_to_weight)
    model.to(device)

    total_result = []
    img_id = 0

    for img_name in tqdm(imgs):
        pred = model.predict(img_dir_path + fr'\{img_name}', conf=conf_thresh, imgsz=img_size, device=device, verbose=False)[0]
        
        img_id += 1

        for l, c, b in zip(pred.boxes.cls, pred.boxes.conf, pred.boxes.xyxy):
            x, y, x2, y2 = b
            total_result.append({'image_id': img_id,
                                 'category_id': VisDrone_name_to_id[pred.names[int(l)]],
                                 'bbox': [float(x), float(y), float(x2) - float(x), float(y2) - float(y)],
                                 'score': float(c)})
    
    with open(rf'detection\results\VisDrone\yolo8_adam_drone.json', 'w') as fp:
        json.dump(total_result, fp)
