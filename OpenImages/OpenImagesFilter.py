import json
import os
from pycocotools.coco import COCO

OpenImages_v1_json_path = 'E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/COCO-Format/val_coco_format.json'
OpenImages_v2_remained_id_path = 'E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/OpenImage_ours_remained_id.txt'
with open(OpenImages_v1_json_path, 'r') as f:
    data = json.load(f)
f.close()
with open(OpenImages_v2_remained_id_path,'r') as f:
    remained_image_ids = f.readlines()
    # for item in remained_image_ids:
    #     item.strip('\n')
    remained_image_ids = [item.strip('\n') for item in remained_image_ids]
f.close()

b = data
images = []
for img in data['images']:
    if img['id'] not in remained_image_ids:
        data['images'].remove(img)

with open(os.path.join("E:\Datasets\AAAI2023\OpenImagesCOCO_Ours\COCO-Format", 'val_coco_format_v2.json'), 'w') as fp:
    json.dump(data, fp, indent=4, separators=(',', ': '))
fp.close()