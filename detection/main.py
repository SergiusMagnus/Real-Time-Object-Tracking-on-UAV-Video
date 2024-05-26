from detectors import ssd_detection, rcnn_detection, yolo_detection, count_fps_sdd, count_fps_rcnn, count_fps_yolo, count_params, \
    ssd_detect_all_coco, rcnn_detect_all_coco, yolo_detect_all_coco, train_yolo, yolo_detect_all_visdrone
from visualization import show_DACSDC_img, show_COCO_img, show_VisDrone_img, show_six_imgs_coco, show_six_imgs_visdrone, show_train_result, show_dect_drone

from datasets import creat_groundndtruth_files_COCO, convert_dataset_to_coco_style_json, count_bbox_dist_visdrone, count_area_bbox_dist_visdrone

from metrics import calc_coco_metrics

from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

if __name__ == '__main__':
    # show_DACSDC_img(dir_name, frame_name, True)
    # show_COCO_img(285, True)
    # show_VisDrone_img('test', '9999986_00000_d_0000007', True)

    # img_path = rf'detection\data\COCO\val2017\000000002153.jpg'
    # img_path = rf'detection\data\COCO\val2017\000000001268.jpg'
    # img_path = rf'detection\data\COCO\val2017\000000000285.jpg'
    # img_path = rf'detection\data\DAC-SDC\car13\0001.jpg'
    # img_path = rf'detection\data\VisDrone\val\images\0000023_01233_d_0000011.jpg'

    # ssd_detection(img_path)
    # rcnn_detection(img_path, score_thresh=0.6)
    # yolo_detection(img_path, score_thresh=0.25)

    # count_fps_sdd(imgs_num=2000)
    # count_fps_rcnn(imgs_num=2000)
    # count_fps_yolo(imgs_num=2000, img_size=960)

    # model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1, score_thresh=0.6)
    # weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    # print(SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.meta['categories'])
    # print(len(SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.meta['categories']))
    
    # creat_groundndtruth_files_COCO()

    # ssd_detect_all_coco()
    # rcnn_detect_all_coco()
    # yolo_detect_all_coco()

    # calc_coco_metrics(rf'detection\results\COCO\instances_val2017.json', rf'detection\results\COCO\ssdlite.json')
    # calc_coco_metrics(rf'detection\results\COCO\instances_val2017.json', rf'detection\results\COCO\fasterrcnn.json')
    # calc_coco_metrics(rf'detection\results\COCO\instances_val2017.json', rf'detection\results\COCO\yolo_640.json')

    # train_yolo(ep_num=150, opt_name='Adam')

    # img_path = rf'detection\data\VisDrone\test\images\0000006_01111_d_0000003.jpg'
    # img_path = rf'detection\data\VisDrone\test\images\9999986_00000_d_0000055.jpg'
    # img_path = rf'detection\data\VisDrone\test\images\9999986_00000_d_0000007.jpg'
    # img_path = rf'detection\data\DAC-SDC\truck1\0001.jpg'
    img_path = rf'detection\data\UAV123\seq\person20\000001.jpg'
    yolo_detection(img_path, weights_path=rf'detection\weights\yolo8_sgd_drone.pt', score_thresh=0.2)

    # convert_dataset_to_coco_style_json(rf'detection\data\VisDrone\test')

    # yolo_detect_all_visdrone(rf'detection\weights\yolo8_adam_drone.pt')
    # calc_coco_metrics(rf'detection\results\VisDrone\gold_test.json', rf'detection\results\VisDrone\yolo8_sgd_drone.json')
    # calc_coco_metrics(rf'detection\results\VisDrone\gold_test.json', rf'detection\results\VisDrone\yolo8_adam_drone.json')

    # count_params()

    # show_six_imgs_coco()
    # show_six_imgs_visdrone()

    # count_bbox_dist_visdrone()
    # count_area_bbox_dist_visdrone()

    # show_train_result(rf'detection\data\train_results\sgd_results.csv', rf'detection\data\train_results\adam_results.csv')

    # show_dect_drone()

    pass
