import torch
from pycocotools.coco import  COCO
coco_val_json_path ='/home/leiyvtian/datasets/coco/annotations/instances_val2017.json'
coco_api = COCO(coco_val_json_path)
coco_voc = [1,2,3,4,5,6,7,8,9,16,17,18,19,20,21,44,62,63,64,67,72]
voc_id = []
no_voc_img_id = []
for img_id, img2anns in coco_api.imgToAnns.items():
    for item in img2anns:
        if item['category_id'] in coco_voc:
            voc_id.append(item['category_id'])
    if len(voc_id)==0:
        no_voc_img_id.append(img_id)
    voc_id = []
# print(len(coco_api))
print(len(no_voc_img_id))