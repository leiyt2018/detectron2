import pandas as pd
import numpy as np

ImageNet_file = 'E:/Datasets/AAAI2023/OE_Analysis/imagenet.xlsx'
imagenet_data = pd.read_excel(ImageNet_file)
removed_ids = []

def collect_id(row, collum):
    if row + 1 <= 1000:
        removed_ids.append(str(row + 1))
        return
    ids = str(imagenet_data.iloc[row, collum])
    strr = ids.split('[')[-1]
    strl = strr.split(']')[0]
    str2 = strl.split(',')

    if len(str2) == 1 and int(str2[0]) <= 1000:
        removed_ids.append(str2[0])
        return

    for s in str2:
        # print(int(s))
        collect_id(int(s) - 1, 5)



def get_removed_ids(OE_file, sheet_name):
    OE_data = pd.read_excel(OE_file, sheet_name=sheet_name)
    raw_removed_ids = OE_data.values[:, 3].tolist()
    raw_removed_ids = [a_ for a_ in raw_removed_ids if a_ == a_]
    # print(a) 输出为 [1, 2, 3, 4]

    # print(raw_removed_ids)
    for i in raw_removed_ids:
        # 深度优先遍历收集，全部叶节点，即id <= 1000
        # print(int(i))
        collect_id(int(i) - 1, 5)



def covert_voc_to_coco(file):
    pass



get_removed_ids('E:/Datasets/AAAI2023/OE_Analysis/OE_v3.xlsx', "PASCAL VOC")
print(len(set(removed_ids)))
get_removed_ids('E:/Datasets/AAAI2023/OE_Analysis/OE_v3.xlsx', "MS COCO")
print(len(set(removed_ids)))
get_removed_ids('E:/Datasets/AAAI2023/OE_Analysis/OE_v3.xlsx', "OpenImage")
print(len(set(removed_ids)))
get_removed_ids('E:/Datasets/AAAI2023/OE_Analysis/OE_v3.xlsx', "Leaky2")
print(len(set(removed_ids)))

ids = [int(i) for i in removed_ids]
ids.sort()
print(len(set(removed_ids)))
f = open("E:/Datasets/AAAI2023/OE_Analysis/removed_ids_v3.txt", "w")

for line in list(set(ids)):
    f.write(str(line) + '\n')
f.close()

# get_removed_ids('/home/leiyvtian/RAL/data/OE_v2.xlsx',"MS COCO")
# # print(removed_ids)
# get_removed_ids('/home/leiyvtian/RAL/data/OE_v2.xlsx',"PASCAL VOC")
# ids = [int(i) for i in removed_ids]
# ids.sort()
# print(len(removed_ids))
stat={}
for item in removed_ids:
    arr = [i for i in removed_ids if i==item]
    stat[item]=len(arr)
# print(stat)
count = 0
for key ,value in stat.items():
    if value>=2:
        count+=1
print(count)
# print(len(stat))
### 制作OE训练集，得到txt格式文件

label_path = 'E:/Datasets/Classification/ImageNet/data/ImageNet2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
with open(label_path, "r") as f:
    data = f.readlines()
label = []
for index, item in enumerate(data):
    label.append(item.strip('\n'))
# print(len(label))
# print(removed_ids)
label_image = {}
OE_data = [str(index).rjust(8,'0') for index, value in enumerate(label,1) if value not in removed_ids]
OE_data = ['ILSVRC2012_val_{}'.format(s) for s in OE_data]

# print(len(removed_ids))
f = open("E:/Datasets/AAAI2023/OE_Analysis/oe_v3.txt", "w")
for line in OE_data:
    f.write(str(line)+'\n')
f.close()