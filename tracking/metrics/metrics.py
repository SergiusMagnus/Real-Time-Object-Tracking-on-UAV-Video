import os
import math
import numpy as np

from tqdm import tqdm
from podm.box import Box, intersection_over_union

skip_seqs = ['bike2', 'car1_s', 'car2_s', 'car3_s', 'car4_s', 'car13', 'person1_s', 'person2_s', 'person3_s', 'person18', 'person20', 'person21',
             'truck1', 'truck4', 'wakeboard4', 'wakeboard10']

def calc_center_error(predict_dir: str, gold_dir: str, skip: bool = False) -> None:
    seqs = os.listdir(predict_dir)

    threshes = []
    precisions = []

    for thresh in tqdm(range(0, 51, 5)):
        success = 0
        total = 0

        t_bboxs = []
        p_bboxs = []

        for seq in seqs:
            if skip and seq.split('.')[0] in skip_seqs:
                continue

            t_bboxs.clear()
            p_bboxs.clear()

            with open(rf'{gold_dir}\{seq}', 'r') as fp:
                for line in fp.readlines():
                    if 'NaN' in line:
                        t_bboxs.append(None)
                    else:
                        t_bboxs.append(list(map(int, line.split(','))))
            
            with open(rf'{predict_dir}\{seq}', 'r') as fp:
                for line in fp.readlines():
                    if 'NaN' in line:
                        p_bboxs.append(None)
                    else:
                        p_bboxs.append(list(map(int, line.split(' '))))

            for p, t in zip(p_bboxs, t_bboxs):
                if t:
                    tx, ty, tw, th = t
                    px, py, pw, ph = p

                    tcx = tx + tw / 2
                    tcy = ty + th / 2

                    pcx = px + pw / 2
                    pcy = py + ph / 2

                    dist = math.sqrt((tcx - pcx) ** 2 + (tcy - pcy) ** 2)

                    if dist <= thresh:
                        success += 1
            
            total += len(t_bboxs)

        threshes.append(thresh)
        precisions.append(success / total)
    
    print('threshes:', threshes)
    print('precisions:', precisions)
    print('auc:', np.trapz(np.array(precisions), dx=0.1))


def calc_region_overlap(predict_dir: str, gold_dir: str, skip: bool = False) -> None:
    seqs = os.listdir(predict_dir)

    threshes = []
    successes = []

    for thresh in tqdm(range(0, 11)):
        success = 0
        total = 0

        t_bboxs = []
        p_bboxs = []

        for seq in seqs:
            if skip and seq.split('.')[0] in skip_seqs:
                continue
            
            t_bboxs.clear()
            p_bboxs.clear()

            with open(rf'{gold_dir}\{seq}', 'r') as fp:
                for line in fp.readlines():
                    if 'NaN' in line:
                        t_bboxs.append(None)
                    else:
                        t_bboxs.append(list(map(int, line.split(','))))
            
            with open(rf'{predict_dir}\{seq}', 'r') as fp:
                for line in fp.readlines():
                    if 'NaN' in line:
                        p_bboxs.append(None)
                    else:
                        p_bboxs.append(list(map(int, line.split(' '))))
            
            for p, t in zip(p_bboxs, t_bboxs):
                if t:
                    tx, ty, tw, th = t
                    px, py, pw, ph = p

                    box1 = Box.of_box(tx, ty, tx + tw, ty + th)
                    box2 = Box.of_box(px, py, px + pw, py + ph)
                    iou = intersection_over_union(box1, box2)

                    if iou >= thresh / 10:
                        success += 1
                
            total += len(t_bboxs)

        threshes.append(thresh)
        successes.append(success / total)
    
    print('threshes:', threshes)
    print('successes:', successes)
    print('auc:', np.trapz(np.array(successes), dx=0.1))
