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
import pandas as pd

def OpenImages_ours_analysis():

    OpenImagesPath = "E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/COCO-Format/val_coco_format.json"
    OpenImagesAPI= COCO(OpenImagesPath)
    # a = OpenImagesAPI.imgtoanns['1c4b37dc276ea673']

    print(len(OpenImagesAPI.imgs))

    bbox_file = 'E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/test-annotations-bbox.csv'
    bbox_data = pd.read_csv(bbox_file)

    cls_file = 'E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/classes.csv'
    cls_data = pd.read_csv(cls_file,header = None)


    ImgeID_bbox_data = bbox_data.values[: , 0]
    cls_cls_data = cls_data.values[:,0]
    cat_cls_data = cls_data.values[:,1]
    # ImgeID_bbox_data.where(ImgeID_bbox_data)
    LabelName_bbox_data = bbox_data.values[:, 2]

    # img_index = np.array(np.where(ImgeID_bbox_data == "1c4b37dc276ea673")).flatten()
    # label = LabelName_bbox_data[img_index]
    # data = ImgeID_bbox_data[img_index[0]]

    openimage_cat = []
    cat2img = {}
    cat2if = {}
    for img_id in OpenImagesAPI.imgs:
        # img_id = img['id']
        img_index = np.array(np.where(ImgeID_bbox_data==img_id)).flatten()
        label = LabelName_bbox_data[img_index]
        label_unique = np.unique(label)
        for item in label_unique:
            cls_index = np.array(np.where(cls_cls_data==item)).flatten()
            cats = cat_cls_data[cls_index]
            cats_unique = np.unique(cats).tolist()
            openimage_cat.extend(cats_unique)
            # cat2img[cats[0]] = [img_id] if not cat2img.__contains__(cats[0]) else cat2img[cats[0]].extend(img_id)
            if not cat2img.__contains__(cats[0]) :
                cat2img[cats[0]] =[img_id]
            else:
                # l = cat2img[cats[0]]
                # l.append(img_id)
                cat2img[cats[0]].append(img_id)
            # cat2if[item] = len(cats)
    openimage_cat = list(set(openimage_cat))
    with open('E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/cat.txt', 'w') as f:
        for item in openimage_cat:
            f.write(item+'\n')
    f.close()

    with open('E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/cat2imgid.txt', 'w') as f:
        for key, value in cat2img.items():
            # if len(value) >=2:
            #     f.write(str(key)+':'+ str(value[0])+','+str(value[1]+'\n'))
            # else:
            #     f.write(str(key)+':'+ str(value[0]+'\n'))
            f.write(str(key)+":")
            for val in value:
                f.write(str(val)+',')
            f.write('\n')
    f.close()

    with open('E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/cat2num.txt', 'w') as f:
        dict = {}
        for key, value in cat2img.items():
            dict[key]=len(value)
        dict = sorted(dict.items(), key = lambda x:-x[1])
        for key, value in dict:
            f.write(str(key)+':'+ str(value)+'\n')

    f.close()
    removed_images = []
    for key, imgs in cat2img.items():
        if len(imgs)<=10:
            for img in imgs:
                if img not in removed_images:
                    removed_images.append(img)
    remained_images = []
    for img_id in OpenImagesAPI.imgs:
        if img_id not in removed_images:
            remained_images.append(img_id)
    print('removed_images:{}'.format(len(removed_images)))
    print('remained_images:{}'.format(len(remained_images)))

    with open('E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/OpenImage_ours_remained_id.txt','w') as f:
        for img_id in remained_images:
            f.write(img_id)
            f.write('\n')
    f.close()

    with open('E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/OpenImage_ours_removed_id.txt','w') as f:
        for img_id in removed_images:
            f.write(img_id)
            f.write('\n')
    f.close()


def OpenImages_vos_analysis():

    removed_cat = ['Radish', 'Zucchini', 'Salad', 'Armadillo', 'Tomato', 'Carrot', 'Otter', 'Snail', 'Broccoli',
                   'Fork', 'Cutting board', 'Bell pepper', 'Starfish', 'Scorpio', 'Bowl', 'Lemon', 'Sculpture',
                   'Butterfly', 'Cocktail', 'Flowerpot', 'Chopsticks', 'Hamster', 'Pasta', 'Orange', 'Sandwich', 'Taco',
                   'Reptile', 'Spider', 'Tart', 'Strawberry', 'Pretzel', 'Knife', 'Kitchen knife', 'Poster', 'Wine glass',
                   'Mouse', 'Laptop', 'Sushi', 'Cattle', 'Spoon', 'Grapefruit', 'Common fig', 'Tablet computer',
                   'Goat', 'Cabbage', 'Beer', 'Juice', 'Squash', 'Dolphin', 'Picture frame', 'Tea', 'Red panda', 'Hamburger',
                   'Waste container', 'Wine glass', 'Coffee cup', 'Saucer', 'Crab']

    file_path = 'E:/Datasets/ObjectDetection/OpenImages/OpenImages/ood_classes_rm_overlap/'
    OpenImagesPath = os.path.join(file_path, 'COCO-Format', 'val_coco_format.json')
    OpenImagesAPI = COCO(OpenImagesPath)
    # a = OpenImagesAPI.imgtoanns['1c4b37dc276ea673']

    print(len(OpenImagesAPI.imgs))

    bbox_file = os.path.join(file_path,'train-annotations-bbox.csv')
    bbox_data = pd.read_csv(bbox_file)

    cls_file = os.path.join(file_path,'class-descriptions-boxable.csv')
    cls_data = pd.read_csv(cls_file, header=None)

    ImgeID_bbox_data = bbox_data.values[:, 0]
    cls_cls_data = cls_data.values[:, 0]
    cat_cls_data = cls_data.values[:, 1]
    # ImgeID_bbox_data.where(ImgeID_bbox_data)
    LabelName_bbox_data = bbox_data.values[:, 2]

    # img_index = np.array(np.where(ImgeID_bbox_data == "1c4b37dc276ea673")).flatten()
    # label = LabelName_bbox_data[img_index]
    # data = ImgeID_bbox_data[img_index[0]]

    openimage_cat = []
    cat2img = {}
    cat2if = {}
    for img_id in OpenImagesAPI.imgs:
        # img_id = img['id']
        img_index = np.array(np.where(ImgeID_bbox_data == img_id)).flatten()
        label = LabelName_bbox_data[img_index]
        label_unique = np.unique(label)
        for item in label_unique:
            cls_index = np.array(np.where(cls_cls_data == item)).flatten()
            cats = cat_cls_data[cls_index]
            cats_unique = np.unique(cats).tolist()
            openimage_cat.extend(cats_unique)
            # cat2img[cats[0]] = [img_id] if not cat2img.__contains__(cats[0]) else cat2img[cats[0]].extend(img_id)
            if not cat2img.__contains__(cats[0]):
                cat2img[cats[0]] = [img_id]
            else:
                # l = cat2img[cats[0]]
                # l.append(img_id)
                cat2img[cats[0]].append(img_id)
            # cat2if[item] = len(cats)
    openimage_cat = list(set(openimage_cat))
    with open(os.path.join(file_path, 'COCO-Format', 'cat.txt'), 'w') as f:
        for item in openimage_cat:
            f.write(item + '\n')
    f.close()

    with open(os.path.join(file_path, 'COCO-Format', 'cat2imgid.txt'), 'w') as f:
        for key, value in cat2img.items():
            # if len(value) >=2:
            #     f.write(str(key)+':'+ str(value[0])+','+str(value[1]+'\n'))
            # else:
            #     f.write(str(key)+':'+ str(value[0]+'\n'))
            f.write(str(key) + ":")
            for val in value:
                f.write(str(val) + ',')
            f.write('\n')
    f.close()

    with open(os.path.join(file_path, 'COCO-Format', 'cat2num.txt'), 'w') as f:
        dict = {}
        for key, value in cat2img.items():
            dict[key] = len(value)
        dict = sorted(dict.items(), key=lambda x: -x[1])
        for key, value in dict:
            f.write(str(key) + ':' + str(value) + '\n')

    f.close()
    removed_images = []
    for key, imgs in cat2img.items():
        if key in removed_cat:
            for img in imgs:
                if img not in removed_images:
                    removed_images.append(img)
    remained_images = []
    for img_id in OpenImagesAPI.imgs:
        if img_id not in removed_images:
            remained_images.append(img_id)
    print('removed_images:{}'.format(len(removed_images)))
    print('remained_images:{}'.format(len(remained_images)))

    with open(os.path.join(file_path, 'COCO-Format', 'OpenImage_ours_remained_id.txt'), 'w') as f:
        for img_id in remained_images:
            f.write(img_id)
            f.write('\n')
    f.close()

    with open(os.path.join(file_path, 'COCO-Format', 'OpenImage_ours_removed_id.txt'), 'w') as f:
        for img_id in removed_images:
            f.write(img_id)
            f.write('\n')
    f.close()

# OpenImages_vos_analysis()

def remove_hand_selected_data():
    file_path = 'E:/Datasets/ObjectDetection/OpenImages/OpenImages/ood_classes_rm_overlap/'
    with open(os.path.join(file_path,'COCO-Format', 'removed_hand_select.txt'),'r') as f:
        selected_file = f.readlines()
        selected_file = [item.strip('\n') for item in selected_file]
    f.close()
    with open(os.path.join(file_path, 'COCO-Format', 'OpenImage_ours_remained_id.txt'), 'r') as f:
        remained_imageids = f.readlines()
        remained_imageids = [item.strip('\n') for item in remained_imageids]
    f.close()
    with open(os.path.join(file_path, 'COCO-Format', 'OpenImage_ours_remained_id_v2.txt'), 'w') as f:
        for img_id in remained_imageids:
            if img_id not in selected_file:
                f.write(img_id)
                f.write('\n')
    f.close()

def make_filtered_openimages():
    file_path = 'E:/Datasets/ObjectDetection/OpenImages/OpenImages/ood_classes_rm_overlap/'
    OpenImagesPath = os.path.join(file_path, 'COCO-Format', 'val_coco_format.json')
    with open(os.path.join(file_path, 'COCO-Format', 'OpenImage_ours_remained_id_v2.txt'), 'r') as f:
        remained_imageids = f.readlines()
        remained_imageids = [item.strip('\n') for item in remained_imageids]
    f.close()
    with open(OpenImagesPath,'r') as f:
        json_data = json.load(f)
    f.close()
    filtered_images = []
    for img in json_data['images']:
        if img['id'] in remained_imageids:
            filtered_images.append(img)
    json_data['images'] = filtered_images
    with open(os.path.join("E:/Datasets/ObjectDetection/OpenImages/OpenImages/ood_classes_rm_overlap/COCO-Format", "val_coco_format_v2.json"), 'w') as fp:
        json.dump(json_data, fp, indent=4, separators=(',', ': '))
    fp.close()

def new_openimages_analysis():

    file_path = 'E:/Datasets/ObjectDetection/OpenImages/OpenImages/ood_classes_rm_overlap/'
    OpenImagesPath = os.path.join(file_path, 'COCO-Format', 'val_coco_format_v2.json')
    OpenImagesAPI = COCO(OpenImagesPath)
    # a = OpenImagesAPI.imgtoanns['1c4b37dc276ea673']

    print(len(OpenImagesAPI.imgs))

    bbox_file = os.path.join(file_path, 'train-annotations-bbox.csv')
    bbox_data = pd.read_csv(bbox_file)

    cls_file = os.path.join(file_path, 'class-descriptions-boxable.csv')
    cls_data = pd.read_csv(cls_file, header=None)

    ImgeID_bbox_data = bbox_data.values[:, 0]
    cls_cls_data = cls_data.values[:, 0]
    cat_cls_data = cls_data.values[:, 1]
    # ImgeID_bbox_data.where(ImgeID_bbox_data)
    LabelName_bbox_data = bbox_data.values[:, 2]

    # img_index = np.array(np.where(ImgeID_bbox_data == "1c4b37dc276ea673")).flatten()
    # label = LabelName_bbox_data[img_index]
    # data = ImgeID_bbox_data[img_index[0]]

    openimage_cat = []
    cat2img = {}
    cat2if = {}
    for img_id in OpenImagesAPI.imgs:
        # img_id = img['id']
        img_index = np.array(np.where(ImgeID_bbox_data == img_id)).flatten()
        label = LabelName_bbox_data[img_index]
        label_unique = np.unique(label)
        for item in label_unique:
            cls_index = np.array(np.where(cls_cls_data == item)).flatten()
            cats = cat_cls_data[cls_index]
            cats_unique = np.unique(cats).tolist()
            openimage_cat.extend(cats_unique)
            # cat2img[cats[0]] = [img_id] if not cat2img.__contains__(cats[0]) else cat2img[cats[0]].extend(img_id)
            if not cat2img.__contains__(cats[0]):
                cat2img[cats[0]] = [img_id]
            else:
                # l = cat2img[cats[0]]
                # l.append(img_id)
                cat2img[cats[0]].append(img_id)
            # cat2if[item] = len(cats)
    openimage_cat = list(set(openimage_cat))
    with open(os.path.join(file_path, 'COCO-Format', 'v2', 'cat.txt'), 'w') as f:
        for item in openimage_cat:
            f.write(item + '\n')
    f.close()

    with open(os.path.join(file_path, 'COCO-Format', 'v2', 'cat2imgid.txt'), 'w') as f:
        for key, value in cat2img.items():
            # if len(value) >=2:
            #     f.write(str(key)+':'+ str(value[0])+','+str(value[1]+'\n'))
            # else:
            #     f.write(str(key)+':'+ str(value[0]+'\n'))
            f.write(str(key) + ":")
            for val in value:
                f.write(str(val) + ',')
            f.write('\n')
    f.close()

    with open(os.path.join(file_path, 'COCO-Format', 'v2', 'cat2num.txt'), 'w') as f:
        dict = {}
        for key, value in cat2img.items():
            dict[key] = len(value)
        dict = sorted(dict.items(), key=lambda x: -x[1])
        for key, value in dict:
            f.write(str(key) + ':' + str(value) + '\n')

    f.close()

# 对vos中的OpenImage数据集分析,剔除掉所属类别图片少的数据
# OpenImages_vos_analysis()
# 移除手动筛选的不合格数据
# remove_hand_selected_data()
# 生成新的json文件
# make_filtered_openimages()
new_openimages_analysis()

