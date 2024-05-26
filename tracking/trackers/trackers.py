import os
import time
import cv2

from tqdm import tqdm
from podm.box import Box, intersection_over_union


def get_tracker(tracker_name: str):
    match tracker_name:
        case 'kcf':
            return cv2.TrackerKCF_create()
        case 'csrt':
            return cv2.TrackerCSRT_create()
        case _:
            raise Exception('Tracker name is not correct.')


def track_all(tracker_name: str) -> None:
    seqs_dir_path = rf'tracking\data\UAV123\seq'
    seqs = os.listdir(seqs_dir_path)

    result_bboxs = []

    for seq in tqdm(seqs):
        result_bboxs.clear()

        seq_dir_path = rf'{seqs_dir_path}\{seq}'

        with open(rf'tracking\data\UAV123\bbox\{seq}.txt', 'r') as fp:
            init_bbox = list(map(int, fp.readline().split(',')))

        frame_names = os.listdir(seq_dir_path)
        init_frame = cv2.imread(rf'{seq_dir_path}\{frame_names[0]}')

        tracker = get_tracker(tracker_name)
        x, y, w, h = init_bbox
        tracker.init(init_frame, [x, y, w, h])

        result_bboxs.append(rf'{x} {y} {w} {h}' + '\n')

        for frame_name in frame_names[1:]:
            frame = cv2.imread(rf'{seq_dir_path}\{frame_name}')

            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = bbox
                result_bboxs.append(rf'{x} {y} {w} {h}' + '\n')
            else:
                break
        
        with open(rf'tracking\results\{tracker_name}\{seq}.txt', 'w') as fp:
            fp.writelines(result_bboxs)


def count_fps(tracker_name: str) -> None:
    seq_dir_path = rf'tracking\data\UAV123\seq\car1'

    total_time = 0.0
    total_frames = 0

    for i in range(100):
        with open(rf'tracking\data\UAV123\bbox\car1.txt', 'r') as fp:
            init_bbox = list(map(int, fp.readline().split(',')))

        frame_names = os.listdir(seq_dir_path)[:100]
        init_frame = cv2.imread(rf'{seq_dir_path}\{frame_names[0]}')
        
        start = time.time()

        tracker = get_tracker(tracker_name)
        x, y, w, h = init_bbox
        tracker.init(init_frame, [x, y, w, h])

        total_time += time.time() - start

        for frame_name in frame_names[1:]:
            frame = cv2.imread(rf'{seq_dir_path}\{frame_name}')

            start = time.time()
            success, _ = tracker.update(frame)

            total_time += time.time() - start

        total_frames += len(frame_names)
    
    print(f'All time {total_time} fames {total_frames} fps {total_frames / total_time}')


def count_failrate(tracker_name: str) -> None:
    seqs_dir_path = rf'tracking\data\UAV123\seq'
    seqs = os.listdir(seqs_dir_path)

    true_bboxs = []
    total_fails = 0
    total_frames = 0

    for seq in tqdm(seqs):
        true_bboxs.clear()
        seq_dir_path = rf'{seqs_dir_path}\{seq}'

        with open(rf'tracking\data\UAV123\bbox\{seq}.txt', 'r') as fp:
            for line in fp.readlines():
                if 'NaN' in line:
                    true_bboxs.append(None)
                else:
                    true_bboxs.append(list(map(int, line.split(','))))

        frame_names = os.listdir(seq_dir_path)

        need_to_init = True

        for i, frame_name in enumerate(frame_names):
            frame = cv2.imread(rf'{seq_dir_path}\{frame_name}')

            if not true_bboxs[i]:
                need_to_init = True
                continue

            if need_to_init:
                x, y, w, h = true_bboxs[i]

                tracker = get_tracker(tracker_name)
                tracker.init(frame, [x, y, w, h])

                need_to_init = False
                continue

            success, box = tracker.update(frame)

            if success:
                x, y, w, h = box
                tx, ty, tw, th = true_bboxs[i]
                box1 = Box.of_box(x, y, x + w, y + h)
                box2 = Box.of_box(tx, ty, tx + tw, ty + th)
                iou = intersection_over_union(box1, box2)

            if not (success and iou > 0.2):
                x, y, w, h = true_bboxs[i]

                tracker = get_tracker(tracker_name)
                tracker.init(frame, [x, y, w, h])

                total_fails += 1
        
        total_frames += len(frame_names)
    
    print(f'fails {total_fails} frames {total_frames} fails per frame {total_fails / total_frames}')
