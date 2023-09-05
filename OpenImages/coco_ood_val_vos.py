import os
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
coco_ood_path = "E:/Datasets/ObjectDetection/coco/annotations/instances_val2017_ood_wrt_bdd_rm_overlap.json"
# coco_val_path = "E:/Datasets/ObjectDetection/coco/annotations/instances_val2017.json"
coco_api = COCO(coco_ood_path)
# print(len(coco_api['imgs']))
# coco_voc = [1,2,3,4,5,6,7,9,16,17,18,19,20,21,44,62,63,64,67,72]
# ex_cat = [32,34,39,40,43,59,80,84,89,90]
coco_voc = []
ex_cat =[]
ignore_id = [12,26,29,30,45,66,68,69,71,83]
cat2num={}
id2catname = {}
for key, value in coco_api.cats.items():
    id2catname[value['id']] = value['name']

for i in range(1,91):
    cat2num[i]=0
for _, item in coco_api.imgToAnns.items():
    ist_cat =[l['category_id'] for l in item]
    for i in range(1,91):
        if i in ist_cat:
            cat2num[i] += 1
print(cat2num)

import matplotlib.pyplot as plt           #导入库

# date = date.set_index('日期')             #把日期列设为索引
# date.index = pd.to_datetime(date.index)   #把索引转为时间格式
plt.bar([id2catname[i] for i in range(1,91) if i not in coco_voc and i not in ex_cat and i not in ignore_id], [cat2num[i] for i in range(1,91) if i not in coco_voc and i not in ex_cat and i not in ignore_id])       #以日期为横轴，收盘价为纵轴绘制条形图
plt.xticks(rotation=80)
plt.show()
# plt.bar([i for i in range(1,61)], [cat2num[i] for i in range(1,91) if i not in coco_voc and i not in ex_cat and i not in ignore_id])       #以日期为横轴，收盘价为纵轴绘制条形图
# plt.show()
num = [cat2num[i] for i in range(1,91) if i not in coco_voc and i not in ex_cat and i not in ignore_id]
max_value = max(num)
min_value = min(num)
_num = [(item-min_value)/(max_value-min_value) for item in num]
print(_num)
print(np.std(_num))
print(np.mean(_num))
