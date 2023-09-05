# Copyright (c) Facebook, Inc. and its affiliates.

# 分析随机抽取的OE数据集的统计特性
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

from predictor import VisualizationDemo

import matplotlib.pyplot as plt

# constants
WINDOW_NAME = "COCO detections"


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

# python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
#   --input input1.jpg input2.jpg \
#   [--other-options]
#   --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        # default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        default = "../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
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

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get("coco_2017_val")

    demo = VisualizationDemo(cfg)

    image_ids_path = '/home/leiyvtian/project/RAL/data/Subset_OE_data_VOCsize/seed1254698/oe.txt'
    with open(image_ids_path,'r') as f:
        _image_ids = f.readlines()
    image_ids = []
    for item in _image_ids:
        image_ids.append(int((item.strip('\n')).split('_')[2])-1)

    image_level_label_path = '/home/leiyvtian/datasets/ImageNet/data/ImageNet2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
    with open(image_level_label_path,'r') as f:
        _image_level_label = f.readlines()
    image_level_label = []
    for item in _image_level_label:
        image_level_label.append(item.strip('\n'))
    image_level_label = np.array(image_level_label)
    image_level_label_randomseed0 = image_level_label[image_ids].tolist()
    # 统计出现频率
    freq = {}
    for i in range(1, 1001, 1):
        freq['{}'.format(i)] = 0
    for i in image_level_label_randomseed0:
        freq['{}'.format(i)] += 1

    # plt.rcParams["font.sans-serif"] = ['SimHei']
    # plt.rcParams["axes.unicode_minus"] = False

    for i in range(1, 1001, 1):
        plt.bar(i, freq[str(i)])

    plt.title("OE distribution")
    plt.xlabel("classes")
    plt.ylabel("number")

    plt.show()


    if args.input_dir:
        for cat in os.listdir(args.input_dir):
            f_data = open("./scene21/{}_data.txt".format(cat), "w")
            # f_label = open("./scene/{}_label.txt".format(cat),'w')
            with tqdm.tqdm(len(os.listdir(os.path.join(args.input_dir,cat)))) as pbar:
                for path in tqdm.tqdm(os.listdir(os.path.join(args.input_dir,cat))):
                    # use PIL, to be consistent with evaluation
                    img = read_image(os.path.join(args.input_dir,cat,path), format="BGR")
                    start_time = time.time()
                    predictions, visualized_output = demo.run_on_image(img)


                    f_data.write(instances_to_token(
                            predictions)+'\n')
                    # f_label.write()
                    pbar.update(1)
            f_data.close()





