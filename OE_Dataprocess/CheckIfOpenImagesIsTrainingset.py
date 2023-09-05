import numpy as np
import os
import json

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.matcher import Matcher
import torch
import matplotlib.pyplot as plt
# Coco evaluator tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd

OpenImagesPath = "/home/leiyvtian/datasets/OpenImages/ood_classes_rm_overlap/COCO-Format/val_coco_format.json"
OpenImagesAPI= COCO(OpenImagesPath)
# a = OpenImagesAPI.imgtoanns['1c4b37dc276ea673']

print(len(OpenImagesAPI.imgs))

bbox_file = '/home/leiyvtian/datasets/OpenImages/ood_classes_rm_overlap/train-annotations-bbox.csv'
bbox_data = pd.read_csv(bbox_file)

cls_file = '/home/leiyvtian/datasets/OpenImages/ood_classes_rm_overlap/class-descriptions-boxable.csv'
cls_data = pd.read_csv(cls_file,header = None)


ImgeID_bbox_data = bbox_data.values[: , 0]
cls_cls_data = cls_data.values[:,0]
cat_cls_data = cls_data.values[:,1]
# ImgeID_bbox_data.where(ImgeID_bbox_data)
LabelName_bbox_data = bbox_data.values[:, 2]

# img_index = np.array(np.where(ImgeID_bbox_data == "1c4b37dc276ea673")).flatten()
# label = LabelName_bbox_data[img_index]
# data = ImgeID_bbox_data[img_index[0]]

openimage_cat = []
cat2img = {}
cat2if = {}
for img_id in OpenImagesAPI.imgs:
    # img_id = img['id']
    img_index = np.array(np.where(ImgeID_bbox_data==img_id)).flatten()
    # label = LabelName_bbox_data[img_index]
    # label_unique = np.unique(label)
    # for item in label_unique:
    #     cls_index = np.array(np.where(cls_cls_data==item)).flatten()
    #     cats = cat_cls_data[cls_index]
    #     cats_unique = np.unique(cats).tolist()
    #     openimage_cat.extend(cats_unique)
    #     # cat2img[cats[0]] = [img_id] if not cat2img.__contains__(cats[0]) else cat2img[cats[0]].extend(img_id)
    #     if not cat2img.__contains__(cats[0]) :
    #         cat2img[cats[0]] =[img_id]
    #     else:
    #         # l = cat2img[cats[0]]
    #         # l.append(img_id)
    #         cat2img[cats[0]].append(img_id)
    if len(img_index)==0:
        print(False)