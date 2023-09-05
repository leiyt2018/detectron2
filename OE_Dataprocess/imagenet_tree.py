# import torch
#
# state_dict = torch.load("D:\Project\RAL2023\detectron2\model_weights\imagenet21k_miil_tree.pth")
#
# print(state_dict)
import os

import torch
from pycocotools.coco import  COCO



# val_path = "E:/Datasets/Classification/val/n01698640"
# oe_path = "oe_v3_12000.txt"
# with open(oe_path,'r') as f:
#     a = f.readlines()
#     dicts = [i.strip('\n') for i in a]
# f.close()
# n = 0
# for item in os.listdir(val_path):
#     if item.split('.')[0] in dicts:
#         n+=1
# print(n)

test_path = "E:\Datasets\AAAI2023\OpenImagesCOCO_Ours_Trainingset_Version\BalancedBenchmark\ImageSets\Layout/test.txt"
val_path = "E:\Datasets\AAAI2023\OpenImagesCOCO_Ours_Trainingset_Version\BalancedBenchmark\ImageSets\Layout/val_v2.txt"
with open(test_path, 'r') as f:
    test_set = f.readlines()
    test_set = [item.strip('\n') for item in test_set]
f.close()
with open(val_path,'r') as f:
    val_set = f.readlines()
    val_set = [item.strip('\n') for item in val_set]
f.close()
count =0
overlap_pic = []
for item in val_set :
    if item in test_set:
        overlap_pic.append(item)

file_path = "E:\Datasets\AAAI2023\OpenImagesCOCO_Ours_Trainingset_Version\BalancedBenchmark\ImageSets\Main_val_v2"
num = 0
cat2num_dict = {}
val_3_path = "E:\Datasets\AAAI2023\OpenImagesCOCO_Ours_Trainingset_Version\BalancedBenchmark\ImageSets\Layout/val_v3.txt"
val_3_list = []
for path in os.listdir(file_path):
    with open(os.path.join(file_path, path), 'r') as f:
        data = f.readlines()
        processed_data = [item.strip('\n') for item in data]
        processed_data = [item for item in processed_data if item not in overlap_pic]
        cat2num_dict[path.split('.')[0]] = len(processed_data)
        if len(processed_data) >=15 :
            num += len(processed_data)
            val_3_list.extend(processed_data)
    f.close()
print(num)
with open(val_3_path, 'w') as f:
    for item in val_3_list :
        f.write(item)
        f.write('\n')
import matplotlib.pyplot as plt  # 导入库

# # date = date.set_index('日期')             #把日期列设为索引
# # date.index = pd.to_datetime(date.index)   #把索引转为时间格式
# plt.bar([i for i in range(1, 51) if i], [cat2num[i] for i in range(1, 91) if
#                                          i not in coco_voc and i not in ex_cat and i not in ignore_id])  # 以日期为横轴，收盘价为纵轴绘制条形图
# plt.show()
keys = []
values = []
for key, value in cat2num_dict.items():
    keys.append(key)
    values.append(value)
plt.bar([i for i in range(len(keys))], values)
plt.xticks(rotation=0)
plt.show()
