from podm import coco_decoder
from podm.metrics import get_pascal_voc_metrics, get_bounding_boxes
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calc_coco_metrics(gold_dir: str, predict_dir: str) -> None:
    coco_gld = COCO(gold_dir)
    coco_rst = coco_gld.loadRes(predict_dir)
    cocoEval = COCOeval(coco_gld, coco_rst, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def calc_pascal_metrics(gold_dir: str, predict_dir: str) -> None:
    with open(gold_dir) as fp:
        gold_dataset = coco_decoder.load_true_object_detection_dataset(fp)

    with open(predict_dir) as fp:
        pred_dataset = coco_decoder.load_pred_object_detection_dataset(fp, gold_dataset)
    
    gt_BoundingBoxes = get_bounding_boxes(gold_dataset)
    pd_BoundingBoxes = get_bounding_boxes(pred_dataset)
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, 0.5)

    for cls, metric in results.items():
        label = metric.label
        print('class', cls)
        print('ap', metric.ap)
        print('precision', metric.precision)
        print('interpolated_recall', metric.interpolated_recall)
        print('interpolated_precision', metric.interpolated_precision)
        print('tp', metric.tp)
        print('fp', metric.fp)
        print('num_groundtruth', metric.num_groundtruth)
        print('num_detection', metric.num_detection)
