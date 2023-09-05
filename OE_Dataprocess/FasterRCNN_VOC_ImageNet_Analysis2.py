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

ImageNetVal_Result_file = '/home/leiyvtian/datasets/ImageNet1k_Val_OE/coco_instances_results_OE.json'
ImageNetVal_GT_file = '/home/leiyvtian/datasets/ImageNet1k_Val_OE/ImageNet1k_Val_OE.json'
# OE_dataset_file = '/home/leiyvtian/datasets/ImageNet1k_Val_OE/oe_val.txt'
# # gt_coco_api = COCO(ImageNetVal_json_file)
# # print(len(gt_coco_api.imgs))
# with open(ImageNetVal_json_file, 'r') as f:
#     dataset = json.load(f)
#
# print(len(dataset))
#
# with open(OE_dataset_file, 'r') as l:
#     _OE_ImageID = l.readlines()
# OE_ImageID = []
# OE_dataset_Results = []
# for line in _OE_ImageID:
#     OE_ImageID.append(line.strip('\n'))
# for data in dataset:
#     if data['image_id'] in OE_ImageID:
#         OE_dataset_Results.append(data)
# print(len(OE_dataset_Results))
#
# with open(os.path.join('/home/leiyvtian/datasets/ImageNet1k_Val_OE', 'coco_instances_results_OE.json'), 'w') as fp:
#     json.dump(OE_dataset_Results, fp, indent=4, separators=(',', ': '))


# dicts = {}
# for item in dataset:
# Evaluate detection results
gt_coco_api = COCO(ImageNetVal_GT_file)
res_coco_api = gt_coco_api.loadRes(ImageNetVal_Result_file) # 从开始编号
# match_matrix = pairwise_iou(gt_coco_api.)
print(len(gt_coco_api.imgs))

# for key, res_values in res_coco_api.imgToAnns.items():
#     res_classes = []
#     res_boxes = []
#     for v in res_values:
#         res_classes.append(v['category_id'])
#         res_boxes.append(v['bbox'])
#     gt_classes = []
#     gt_boxes = []
#     gt_values = gt_coco_api.imgToAnns[key]
#     for v in gt_values:
#         gt_classes.append(v['category_id'])
#         gt_boxes.append(v['bbox'])
iou_threshold = [0.3, 0.7]
iou_labels = [0, -1, 1]
matcher = Matcher(iou_threshold,iou_labels)

image_level_label_path = '/home/leiyvtian/datasets/ImageNet/data/ImageNet2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
with open(image_level_label_path, 'r') as f:
    _image_level_label = f.readlines()
image_level_label = []
for item in _image_level_label:
    image_level_label.append(item.strip('\n'))
image_level_label = np.array(image_level_label)
# image_level_label_randomseed0 = image_level_label[image_ids].tolist()
# 统计出现频率
freq = {}
for i in range(1, 1001, 1):
    freq['{}'.format(i)] = 0
freq_cls = {}
for i in range(1,21,1):
    freq_cls['{}'.format(i)] = 0
# for i in image_level_label_randomseed0:
#     freq['{}'.format(i)] += 1
count = 0
for key, gt_values in gt_coco_api.imgToAnns.items():
    # print(key)
    gt_classes = []
    gt_boxes = []
    for v in gt_values:
        gt_classes.append(v['category_id'])
        gt_boxes.append(v['bbox'])


    res_classes = []
    res_boxes = []
    res_values = res_coco_api.imgToAnns[key]
    for v in res_values:
        res_classes.append(v['category_id'])
        res_boxes.append(v['bbox'])
    res_boxes = Boxes(res_boxes)
    gt_boxes = Boxes(gt_boxes)

    pairwise_iou_matrix = pairwise_iou(gt_boxes,res_boxes)
    matched_idxs, matched_labels = matcher(pairwise_iou_matrix)
    matched_labels = matched_labels.long()

    if len(matched_labels) != 0:

        if_detected = len(torch.nonzero(torch.where(matched_labels==1,matched_labels,0))) != 0
        if if_detected:
            count +=1
            image_index = int(gt_values[0]['image_id'].split('_')[2]) - 1
            image_lvl_label = image_level_label[image_index]
            freq['{}'.format(image_lvl_label)] += 1

            detected_index = torch.nonzero(torch.where(matched_labels == 1, matched_labels, 0))
            # detected_matched_idxs = matched_idxs[detected_index]
            _res_classes = torch.tensor(res_classes)
            detected_cat = _res_classes[detected_index]
            detected_cat_unique = torch.unique(detected_cat)
            # for i in detected_cat_unique:
            #     freq_cls[str(i)]+=1
            for item in detected_cat_unique:
                freq_cls[str(item.item())]+=1
            # freq_cls[str(detected_cat_unique[0].item())]+=1

removed_ids_file = '/home/leiyvtian/datasets/ImageNet1k_Val_OE/removed_ids.txt'
with open(removed_ids_file, 'r') as f:
    _removed_ids = f.readlines()
removed_ids = []
for item in _removed_ids:
    removed_ids.append(item.strip('\n'))

index = 1
for i in range(1, 1001, 1):
    if str(i) not in removed_ids:
        plt.bar(index, freq[str(i)])
        index +=1
plt.title("OE distribution")
plt.xlabel("classes")
plt.ylabel("number")

plt.show()

for i in range(1, 21, 1):
    plt.bar(i, freq_cls[str(i)])

plt.title("OE class")
plt.xlabel("classes")
plt.ylabel("number")

plt.show()

with open("./stat_iou_0.5.txt",'w') as f:
    for key, value in freq.items():
        if key not in removed_ids:
            f.write("{}:{}\n".format(key,value))
f.close()
#
# with open("./stat_voc.txt",'w') as f:
#     for key, value in freq_cls.items():
#         f.write("{}:{}\n".format(key,value))
print(count/35550)





