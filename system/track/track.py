import os
import time
import math
import cv2
import torch

from ultralytics import YOLO
from podm.box import Box, intersection_over_union
from tqdm import tqdm


def get_tracker(tracker_name: str):
    match tracker_name:
        case 'kcf':
            return cv2.TrackerKCF_create()
        case 'csrt':
            return cv2.TrackerCSRT_create()
        case _:
            raise Exception('Tracker name is not correct.')


def find_nearest_bbox(lbbox, dist_thresh, predicted_bboxs):
    lx, ly, lw, lh = lbbox
    max_iou = 0

    for pbbox in predicted_bboxs:
        px, py, px2, py2 = pbbox

        box1 = Box.of_box(lx, ly, lx + lw, ly + lh)
        box2 = Box.of_box(px, py, px2, py2)

        iou = intersection_over_union(box1, box2)
        if iou > max_iou:
            max_iou = iou
            bbox = [px, py, px2 - px, py2 - py]
    
    min_dist = dist_thresh

    if max_iou == 0:
        lcx = lx + lw / 2
        lcy = ly + lh / 2

        for pbbox in predicted_bboxs:
            px, py, px2, py2 = pbbox

            pcx = px + (px2 - px) / 2
            pcy = py + (py2 - py) / 2
            
            dist = math.sqrt((lcx - pcx) ** 2 + (lcy - pcy) ** 2)

            if dist < min_dist and dist < dist_thresh:
                min_dist = dist
                bbox = [px, py, px2 - px, py2 - py]

    if max_iou == 0 and min_dist == dist_thresh:
        return None

    return bbox


def get_new_img_border(margin, img_size, obj_bbox):
    x, y, w, h = obj_bbox
    cx = x + (w // 2)
    cy = y + (h // 2)

    img_w, img_h = img_size

    if (w * h) / (margin * margin) > 0.01:
        return 0, img_w - 1, 0, img_h - 1

    marg_w = margin - 1
    marg_h = margin - 1

    if marg_w > img_w:
        marg_w = img_w - 1
    
    if marg_h > img_h:
        marg_h = img_h - 1

    lw, rw, th, bh = -1, -1, -1, -1 

    if cx - (marg_w // 2) < 0:
        lw = 0
        rw = marg_w
    elif cx + (marg_w // 2) > img_w - 1:
        lw = img_w - 1 - marg_w
        rw = img_w - 1
    
    if cy - (marg_h // 2) < 0:
        th = 0
        bh = marg_h
    elif cy + (marg_h // 2) > img_h - 1:
        th = img_h - 1 - marg_h
        bh = img_h - 1

    if lw == -1:
        lw = cx - (marg_w // 2)
        rw = cx + (marg_w // 2)
    
    if th == -1:
        th = cy - (marg_h // 2)
        bh = cy + (marg_h // 2)

    return lw, rw, th, bh


def track(
        seq_name: str,
        tracker_name: str = 'kcf',
        device: str = 'cpu',
        yolo_weights: str = f'tracking_system\weights\yolo8_drone.pt',
        conf_thresh: float = 0.2,
        img_size: int = 640,
        frame_num_to_call_detector: int = 30,
        dist_thresh: int = 300,
        margin: int = 640
    ) -> None:
    seq_dir_path = rf'tracking_system\data\UAV123\seq\{seq_name}'

    device = torch.device(device)
    yolo = YOLO(yolo_weights)
    yolo.to(device)

    with open(rf'tracking_system\data\UAV123\bbox\{seq_name}.txt', 'r') as fp:
        t_init_bbox = list(map(int, fp.readline().split(',')))
    
    frame_names = os.listdir(seq_dir_path)

    init_frame = cv2.imread(rf'{seq_dir_path}\{frame_names[0]}')

    img_h, img_w, _ = init_frame.shape

    lw, rw, th, bh = get_new_img_border(margin, [img_w, img_h], t_init_bbox)

    detector_prediction = yolo.predict(init_frame[th:bh, lw:rw], conf=conf_thresh, imgsz=img_size, device=device, verbose=False)[0]
    # detector_prediction.show()

    max_iou = 0.1
    obj_class = -1

    for l, b in zip(detector_prediction.boxes.cls, detector_prediction.boxes.xyxy):
        px, py, px2, py2 = b

        px = lw + int(px)
        py = th + int(py)
        px2 = lw + int(px2)
        py2 = th + int(py2)

        box1 = Box.of_box(t_init_bbox[0], t_init_bbox[1], t_init_bbox[0] + t_init_bbox[2], t_init_bbox[1] + t_init_bbox[3])
        box2 = Box.of_box(px, py, px2, py2)
        iou = intersection_over_union(box1, box2)

        if iou > max_iou:
            max_iou = iou
            obj_class = int(l)
            init_bbox = [px, py, px2 - px, py2 - py]
    
    if obj_class == -1:
        with open(rf'tracking_system\results\{tracker_name}\{seq_name}.txt', 'w') as fp:
            fp.write('NaN\n')
        return
    
    tracker = get_tracker(tracker_name)
    tracker.init(init_frame, init_bbox)

    result_bboxs = []
    result_bboxs.append(f'{init_bbox[0]} {init_bbox[1]} {init_bbox[2]} {init_bbox[3]}\n')

    frame_counter = 0
    skip_frame = False

    for frame_name in frame_names[1:]:
        frame = cv2.imread(rf'{seq_dir_path}\{frame_name}')

        if skip_frame:
            result_bboxs.append('NaN\n')
            frame_counter += 1

            if frame_counter < frame_num_to_call_detector:
                continue
            else:
                skip_frame = False
        else:
            success, bbox = tracker.update(frame)
            frame_counter += 1

        if not success or frame_counter >= frame_num_to_call_detector:
            frame_counter = 0

            img_h, img_w, _ = init_frame.shape
            lw, rw, th, bh = get_new_img_border(margin, [img_w, img_h], lbbox)

            detector_prediction = yolo.predict(frame[th:bh, lw:rw], conf=conf_thresh, imgsz=img_size, device=device, verbose=False)[0]
            # detector_prediction.show()

            predicted_bboxs = []
            for l, b in zip(detector_prediction.boxes.cls, detector_prediction.boxes.xyxy):
                if int(l) == obj_class:
                    px, py, px2, py2 = b
                    predicted_bboxs.append([lw + int(px), th + int(py), lw + int(px2), th + int(py2)])

            bbox = find_nearest_bbox(lbbox, dist_thresh, predicted_bboxs)
            
            if bbox is None:
                result_bboxs.append('NaN\n')
                skip_frame = True
                continue

            tracker = get_tracker(tracker_name)
            tracker.init(frame, bbox)

        lbbox = bbox
        result_bboxs.append(f'{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')
    
    with open(rf'tracking_system\results\{tracker_name}\{seq_name}.txt', 'w') as fp:
        fp.writelines(result_bboxs)


def track_all(
        tracker_name: str = 'kcf',
        device: str = 'cpu',
        path_to_weights: str = f'tracking_system\weights\yolo8_drone.pt',
        conf_thresh: float = 0.2,
        img_size: int = 640,
        frame_num_to_call_detector: int = 30,
        dist_thresh: int = 300,
        margin: int = 640
    ) -> None:
    seqs_dir_path = rf'tracking_system\data\UAV123\seq'
    seqs = os.listdir(seqs_dir_path)

    for seq in tqdm(seqs):
        track(seq, tracker_name, device, path_to_weights, conf_thresh, img_size, frame_num_to_call_detector, dist_thresh, margin)


def count_fps(
        tracker_name: str = 'kcf',
        device: str = 'cpu',
        path_to_weights: str = f'tracking_system\weights\yolo8_drone.pt',
        conf_thresh: float = 0.2,
        img_size: int = 640,
        frame_num_to_call_detector: int = 30,
        dist_thresh: int = 300,
        margin: int = 640
    ) -> None:
    seq_dir_path = rf'tracking_system\data\UAV123\seq\car1'

    total_time = 0.0
    total_frames = 0

    for i in range(100):
        device = torch.device(device)
        yolo = YOLO(path_to_weights)
        yolo.to(device)

        with open(rf'tracking_system\data\UAV123\bbox\car1.txt', 'r') as fp:
            t_init_bbox = list(map(int, fp.readline().split(',')))

        frame_names = os.listdir(seq_dir_path)[:100]

        init_frame = cv2.imread(rf'{seq_dir_path}\{frame_names[0]}')

        start = time.time()

        img_h, img_w, _ = init_frame.shape
        lw, rw, th, bh = get_new_img_border(margin, [img_w, img_h], t_init_bbox)
        detector_prediction = yolo.predict(init_frame[th:bh, lw:rw], conf=conf_thresh, imgsz=img_size, device=device, verbose=False)[0]

        max_iou = 0
        obj_class = -1

        for l, b in zip(detector_prediction.boxes.cls, detector_prediction.boxes.xyxy):
            px, py, px2, py2 = b

            px = lw + int(px)
            py = th + int(py)
            px2 = lw + int(px2)
            py2 = th + int(py2)

            box1 = Box.of_box(t_init_bbox[0], t_init_bbox[1], t_init_bbox[0] + t_init_bbox[2], t_init_bbox[1] + t_init_bbox[3])
            box2 = Box.of_box(px, py, px2, py2)
            iou = intersection_over_union(box1, box2)

            if iou > max_iou:
                max_iou = iou
                obj_class = int(l)
                init_bbox = [px, py, px2 - px, py2 - py]

        tracker = get_tracker(tracker_name)
        tracker.init(init_frame, init_bbox)

        frame_counter = 0
        skip_frame = False

        total_time += time.time() - start

        for frame_name in frame_names[1:]:
            frame = cv2.imread(rf'{seq_dir_path}\{frame_name}')

            start = time.time()

            if skip_frame:
                frame_counter += 1

                if frame_counter < frame_num_to_call_detector:
                    continue
                else:
                    skip_frame = False
            else:
                success, bbox = tracker.update(frame)
                frame_counter += 1

            if not success or frame_counter >= frame_num_to_call_detector:
                frame_counter = 0

                img_h, img_w, _ = init_frame.shape
                lw, rw, th, bh = get_new_img_border(margin, [img_w, img_h], lbbox)

                detector_prediction = yolo.predict(frame[th:bh, lw:rw], conf=conf_thresh, imgsz=img_size, device=device, verbose=False)[0]

                predicted_bboxs = []
                for l, b in zip(detector_prediction.boxes.cls, detector_prediction.boxes.xyxy):
                    if int(l) == obj_class:
                        px, py, px2, py2 = b
                        predicted_bboxs.append([lw + int(px), th + int(py), lw + int(px2), th + int(py2)])

                bbox = find_nearest_bbox(lbbox, dist_thresh, predicted_bboxs)

                if bbox is None:
                    skip_frame = True
                    continue

                tracker = get_tracker(tracker_name)
                tracker.init(frame, bbox)

            lbbox = bbox

            total_time += time.time() - start
        total_frames += len(frame_names)

    print(f'All time {total_time} fames {total_frames} fps {total_frames / total_time}')
