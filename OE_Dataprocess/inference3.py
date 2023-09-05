# Copyright (c) Facebook, Inc. and its affiliates.

# 推理最大proposal与gt的覆盖程度

import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json

# Detectron imports
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.layers import batched_nms
from detectron2.structures import BoxMode, Boxes, Instances, pairwise_iou
from detectron2.data import MetadataCatalog

from demo.predictor import VisualizationDemo

import matplotlib.pyplot as plt

# constants
WINDOW_NAME = "COCO detections"

voc_cat_mapping_dict = {}
for i in range(0, 20):
    voc_cat_mapping_dict[i] = i + 1

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        # default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        default = "../configs/PascalVOC-Detection/rpn_R_50_C4_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input_dir",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    # parser.add_argument(
    #     "--output_dir",
    #     help="A file or directory to save output visualizations. "
    #     "If not given, will show output in an OpenCV window.",
    # )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def instances_to_token(instances):
    num_instance = len(instances)
    if num_instance == 0:
        return []
    classes = instances["instances"].pred_classes.cpu().tolist()
    token = ""
    for item in classes:
        token = token+"{},".format(item)
    return token

def instances_to_json(instances, img_id, cat_mapping_dict=None):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances): detectron2 instances
        img_id (int): the image id
        cat_mapping_dict (dict): dictionary to map between raw category id from net and dataset id. very important if
        performing inference on different dataset than that used for training.

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.cpu().tolist()
    classes = instances.pred_classes.cpu().tolist()
    # inter_feat = instances.inter_feat.cpu().tolist()
    # if instances.has('logistic_score'):
    #    logistic_score = instances.logistic_score.cpu().tolist()
    # import ipdb; ipdb.set_trace()

    classes = [
        cat_mapping_dict[class_i] if class_i in cat_mapping_dict.keys() else -
        1 for class_i in classes]
    # breakpoint()
    # pred_cls_probs = instances.pred_cls_probs.cpu().tolist()

    # if instances.has("pred_boxes_covariance"):
    #     pred_boxes_covariance = covar_xyxy_to_xywh(
    #         instances.pred_boxes_covariance).cpu().tolist()
    # else:
    #     pred_boxes_covariance = []

    results = []
    for k in range(num_instance):
        if classes[k] != -1:
            # if instances.has('logistic_score'):
            #     result = {
            #         "image_id": img_id,
            #         "category_id": classes[k],
            #         "bbox": boxes[k],
            #         "score": scores[k],
            #         "inter_feat": inter_feat[k],
            #         "logistic_score": logistic_score[k],
            #         "cls_prob": pred_cls_probs[k],
            #         "bbox_covar": pred_boxes_covariance[k]
            #     }
            # else:
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
                #"inter_feat": inter_feat[k],
                # "cls_prob": pred_cls_probs[k]
                # "bbox_covar": pred_boxes_covariance[k]
            }

            results.append(result)
    return results

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get("voc_2012_test")

    demo = VisualizationDemo(cfg)

    # image_ids_path = '/home/leiyvtian/project/RAL/data/Subset_OE_data_VOCsize/seed1254698/oe.txt'
    # with open(image_ids_path,'r') as f:
    #     _image_ids = f.readlines()
    # image_ids = []
    # for item in _image_ids:
    #     image_ids.append(int((item.strip('\n')).split('_')[2])-1)
    #
    # image_level_label_path = '/home/leiyvtian/datasets/ImageNet/data/ImageNet2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
    # with open(image_level_label_path,'r') as f:
    #     _image_level_label = f.readlines()
    # image_level_label = []
    # for item in _image_level_label:
    #     image_level_label.append(item.strip('\n'))
    # image_level_label = np.array(image_level_label)
    # image_level_label_randomseed0 = image_level_label[image_ids].tolist()
    # # 统计出现频率
    # freq = {}
    # for i in range(1, 1001, 1):
    #     freq['{}'.format(i)] = 0
    # for i in image_level_label_randomseed0:
    #     freq['{}'.format(i)] += 1

    # # plt.rcParams["font.sans-serif"] = ['SimHei']
    # # plt.rcParams["axes.unicode_minus"] = False
    #
    # for i in range(1, 1001, 1):
    #     plt.bar(i, freq[str(i)])
    #
    # plt.title("OE distribution")
    # plt.xlabel("classes")
    # plt.ylabel("number")
    #
    # plt.show()
    j =1
    final_output_list = []
    if args.input_dir:
        # f_data = open("/home/leiyvtian/RAL/data/ImageNet1k.txt", "w")
        # f_label = open("./scene/{}_label.txt".format(cat),'w')
        # with tqdm.tqdm(len(os.listdir(args.input_dir))) as pbar:
        # with tqdm.tqdm(10) as pbar:
            for path in tqdm.tqdm(os.listdir(args.input_dir)):
                # use PIL, to be consistent with evaluation
                img = read_image(os.path.join(args.input_dir,path), format="BGR")
                start_time = time.time()
                predictions, vis = demo.run_on_image(img)
                boxes = predictions['proposals'].proposal_boxes
                scores = predictions['proposals'].objectness_logits
                sizes = boxes.area()
                ind = sizes[:-1].argmax().item() if len(sizes)>1 else 0
                max_size = sizes[ind]
                max_box = boxes[ind].tensor.cpu().numpy()
                max_box = BoxMode.convert(max_box, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                max_box = max_box.tolist()
                max_score = scores[ind]
                max_score = max_score.cpu().tolist()
                final_output_list.extend(
                    [{
                        "image_id": path.split('.')[0],
                        "category_id": 21,
                        "bbox": max_box[0],
                        "score":max_score
                        # "score": scores[k],
                        # "inter_feat": inter_feat[k],
                        # "cls_prob": pred_cls_probs[k]
                        # "bbox_covar": pred_boxes_covariance[k]
                    }])
                # vis.show()
                # if j%10 ==0:
                vis.save('./data128/{}'.format(path))
                j = j +1
                    # instances_to_json(
                    #     predictions['instances'],
                    #     path.split('.')[0],
                    #     voc_cat_mapping_dict))
                # f_label.write()
                # pbar.update(1)
        # f_data.close()

            inference_output_dir = '/home/leiyvtian/datasets/ImageNet1k_Val_OE'
            with open(os.path.join(inference_output_dir, 'coco_instances_results_rpn_max_1000proposal.json'), 'w') as fp:
                json.dump(final_output_list, fp, indent=4, separators=(',', ': '))






