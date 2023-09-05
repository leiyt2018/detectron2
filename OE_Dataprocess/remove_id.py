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

ImageNetVal_Result_file = '/home/leiyvtian/datasets/ImageNet1k_Val_OE/coco_instances_results_rpn_max_proposal.json'

# f = open(ImageNetVal_Result_file, 'r')
# content = f.read()
# a = json.loads(content)
# for item in a:
#     item['bbox'] = item['bbox'][0]
# f.close()
# inference_output_dir = '/home/leiyvtian/datasets/ImageNet1k_Val_OE'
# with open(os.path.join(inference_output_dir, 'coco_instances_results_rpn_max_proposal1.json'), 'w') as fp:
#     json.dump(a, fp, indent=4, separators=(',', ': '))
OE_dataset_file = '/home/leiyvtian/datasets/ImageNet1k_Val_OE/oe_val.txt'
# gt_coco_api = COCO(ImageNetVal_json_file)
# print(len(gt_coco_api.imgs))
with open(ImageNetVal_Result_file, 'r') as f:
    dataset = json.load(f)

print(len(dataset))

with open(OE_dataset_file, 'r') as l:
    _OE_ImageID = l.readlines()
OE_ImageID = []
OE_dataset_Results = []
for line in _OE_ImageID:
    OE_ImageID.append(line.strip('\n'))
for data in dataset:
    if data['image_id'] in OE_ImageID:
        OE_dataset_Results.append(data)
print(len(OE_dataset_Results))

with open(os.path.join('/home/leiyvtian/datasets/ImageNet1k_Val_OE', 'coco_instances_results_rpn_max_128proposal_oe'), 'w') as fp:
    json.dump(OE_dataset_Results, fp, indent=4, separators=(',', ': '))