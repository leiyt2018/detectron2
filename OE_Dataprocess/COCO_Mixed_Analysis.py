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

voc_cat = ['person','bicycle','car','motorcycle','airplane','bus','train','boat','bird','cat','dog',
           'horse','sheep','cow','bottle','chair','couch','potted plant','dinning table','tv']

# Coco evaluator tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
coco_mixed_id_json_path = '/home/leiyvtian/datasets/coco/annotations/instances_val2017_mixed_ID.json'
coco_mixed_ood_json_path = '/home/leiyvtian/datasets/coco/annotations/instances_val2017_mixed_OOD.json'
coco_ood_json_path = '/home/leiyvtian/datasets/coco/annotations/instances_val2017_coco_ood.json'
coco_val_json_path ='/home/leiyvtian/datasets/coco/annotations/instances_val2017.json'


coco_mixed_id_api = COCO(coco_mixed_id_json_path)
coco_mixed_ood_api = COCO(coco_mixed_ood_json_path)
coco_val_api = COCO(coco_val_json_path)
coco_ood_api = COCO(coco_ood_json_path)

print(len(coco_mixed_id_api.imgs))
print(len(coco_mixed_ood_api.imgs))
print(len(coco_val_api.imgs))

delta = []
ann_ood = 0
ann_id = 0
ann_val = 0
ann_extend_ood = 0
ann_extend_ood_val = 0
non_extend_label = 0
for key, value in coco_mixed_id_api.imgs.items():
    if key in coco_mixed_ood_api.imgToAnns:
        ann_ood += len(coco_mixed_ood_api.imgToAnns[key])
    if key in coco_mixed_id_api.imgToAnns:
        ann_id += len(coco_mixed_id_api.imgToAnns[key])
    if key in coco_val_api.imgToAnns:
        ann_val += len(coco_val_api.imgToAnns[key])
for key , value in coco_ood_api.imgs.items():
    ann_extend_ood += len(coco_ood_api.imgToAnns[key])
    ann_extend_ood_val +=len(coco_val_api.imgToAnns[key])

with open('./remained_no_nontest_ood.txt','w') as f:
    for key , value in coco_ood_api.imgs.items():
        if len(coco_ood_api.imgToAnns[key]) == len(coco_val_api.imgToAnns[key]):
            non_extend_label+=1
            f.write(str(key))
            f.write('\n')
f.close()




print('ann_val-ann_id = {}'.format(ann_val-ann_id))
print('ann_val-ann_id + ann_ood = {}'.format(ann_val - ann_id + ann_ood))
print('ann_extend_ood:{}'.format(ann_extend_ood))
print('ann_extend_ood_val:{}'.format(ann_extend_ood_val))
print('ann_extend_ood_non_extend_label:{}'.format(non_extend_label))

print('coco_ood:{}'.format(len(coco_ood_api.imgToAnns)))
print('coco_mix:{}'.format(len(coco_mixed_id_api.imgToAnns)))
print('coco_mix_ood:{}'.format(len(coco_mixed_ood_api.imgToAnns)))

# delta = []
# for key, value in gt_coco_api.imgs.items():
#     ann = gt_coco_val_api.imgToAnns[key]
#     coco_label_num = len(ann)
#     unsniffer_label_num = len(gt_coco_api.imgToAnns[key])
#     delta.append( unsniffer_label_num - coco_label_num)
#
# total_num = 0
# for item in delta :
#     total_num += item
# mean_num_delta = total_num /len(delta)
# print(mean_num_delta)
# print(total_num)
# print(len(delta))


