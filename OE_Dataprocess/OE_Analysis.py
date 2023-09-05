import numpy as np
import os
import json
import shutil

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Coco evaluator tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

file_path = '/home/leiyvtian/RAL/Data/seed0/oe_v2_10000.txt'
src_file = '/home/leiyvtian/datasets/ImageNet1k_Val_OE/JPEGImages/'
trg_file = '/home/leiyvtian/RAL/Data/seed0/oe_v2_10000/'
with open(file_path,'r') as f:
    image_ids = f.readlines()
    image_ids = [img.strip('\n') for img in image_ids]
f.close()

for img_id in image_ids:
    shutil.copy(src_file+img_id+'.JPEG', trg_file+img_id+'.JPEG')