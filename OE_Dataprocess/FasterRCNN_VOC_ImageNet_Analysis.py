import numpy as np
import os
import json

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Coco evaluator tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ImageNetVal_json_file = '/home/leiyvtian/datasets/ImageNet1k_Val_OE/coco_instances_results.json'
OE_dataset_file = '/home/leiyvtian/datasets/ImageNet1k_Val_OE/oe_val.txt'
# gt_coco_api = COCO(ImageNetVal_json_file)
# print(len(gt_coco_api.imgs))
with open(ImageNetVal_json_file, 'r') as f:
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

with open(os.path.join('/home/leiyvtian/datasets/ImageNet1k_Val_OE', 'coco_instances_results_OE.json'), 'w') as fp:
    json.dump(OE_dataset_Results, fp, indent=4, separators=(',', ': '))


# dicts = {}
# for item in dataset:
