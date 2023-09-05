import os
import json
import numpy as np
import pandas as pd
from pycocotools.coco import COCO


OPEN_IMAGES_TO_COCO = {#'Person': 'person',
#                        'Bicycle': 'bicycle',
#                        'Car': 'car',
#                        'Motorcycle': 'motorcycle',
#                        'Airplane': 'airplane',
#                        'Bus': 'bus',
#                        'Train': 'train',
                       'Truck': 'truck',
                       # 'Boat': 'boat',
                       'Traffic light': 'traffic light',
                       'Fire hydrant': 'fire hydrant',
                       'Stop sign': 'stop sign',
                       'Parking meter': 'parking meter',
                       ## 'Bench': 'bench',
                       # 'Bird': 'bird',
                       # 'Cat': 'cat',
                       # 'Dog': 'dog',
                       # 'Horse': 'horse',
                       # 'Sheep': 'sheep',
                       # 'Elephant': 'cow',
                       'Elephant': 'elephant',
                       'Bear': 'bear',
                       'Zebra': 'zebra',
                       'Giraffe': 'giraffe',
                       'Backpack': 'backpack',
                       'Umbrella': 'umbrella',
                       'Handbag': 'handbag',
                       ##'Tie': 'tie',
                       'Suitcase': 'suitcase',
                       'Flying disc': 'frisbee',
                       'Ski': 'skis',
                       'Snowboard': 'snowboard',
                       'Ball': 'sports ball',
                       'Kite': 'kite',
                       'Baseball bat': 'baseball bat',
                       'Baseball glove': 'baseball glove',
                       'Skateboard': 'skateboard',
                       'Surfboard': 'surfboard',
                       'Tennis racket': 'tennis racket',
                       # 'Bottle': 'bottle',
                       'Wine glass': 'wine glass',
                       'Coffee cup': 'cup',
                       'Fork': 'fork',
                       'Knife': 'knife',
                       ##'Spoon': 'spoon',
                       'Bowl': 'bowl',
                       'Banana': 'banana',
                       'Apple': 'apple',
                       'Sandwich': 'sandwich',
                       'Orange': 'orange',
                       'Broccoli': 'broccoli',
                       'Carrot': 'carrot',
                       'Hot dog': 'hot dog',
                       'Pizza': 'pizza',
                       'Doughnut': 'donut',
                       'Cake': 'cake',
                       # 'Chair': 'chair',
                       # 'Couch': 'couch',
                       # 'Houseplant': 'potted plant',
                       ##'Bed': 'bed',
                       # 'Table': 'dining table',
                       'Toilet': 'toilet',
                       # 'Television': 'tv',
                       'Laptop': 'laptop',
                       'Computer mouse': 'mouse',
                       'Remote control': 'remote',
                       'Computer keyboard': 'keyboard',
                       'Mobile phone': 'cell phone',
                       'Microwave oven': 'microwave',
                       'Oven': 'oven',
                       'Toaster': 'toaster',
                       'Sink': 'sink',
                       'Refrigerator': 'refrigerator',
                       ##'Book': 'book',
                       'Clock': 'clock',
                       'Vase': 'vase',
                       'Scissors': 'scissors',
                       'Teddy bear': 'teddy bear',
                       'Hair dryer': 'hair drier',
                       'Toothbrush': 'toothbrush'}

def OpenImages_testset_COCO_cat():
    # dataset_dir = "E:\Datasets\AAAI2023\OpenImagesCOCO_Ours\COCO-Format"
    # with open(os.path.join(dataset_dir,'val_coco_format.json'), 'r') as f:
    #     dataset_dicts = json.load(f)
    # f.close()
    # for img in dataset_dicts['images']:
    #     img_id = img['id']
    Cat = [key for key, value in OPEN_IMAGES_TO_COCO.items()]
    Instance_num_dict = {}
    for c in Cat:
        Instance_num_dict[c] = 0

    OpenImagesPath = "E:\Datasets\AAAI2023\OpenImagesCOCO_Ours\COCO-Format"
    OpenImagesAPI = COCO(os.path.join(OpenImagesPath,'val_coco_format_v2.json'))
    # a = OpenImagesAPI.imgtoanns['1c4b37dc276ea673']

    print(len(OpenImagesAPI.imgs))

    bbox_file = 'E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/test-annotations-bbox.csv'
    bbox_data = pd.read_csv(bbox_file)

    cls_file = 'E:/Datasets/AAAI2023/OpenImagesCOCO_Ours/classes.csv'
    cls_data = pd.read_csv(cls_file, header=None)

    ImgeID_bbox_data = bbox_data.values[:, 0]
    cls_cls_data = cls_data.values[:, 0]
    cat_cls_data = cls_data.values[:, 1]
    # ImgeID_bbox_data.where(ImgeID_bbox_data)
    LabelName_bbox_data = bbox_data.values[:, 2]

    # img_index = np.array(np.where(ImgeID_bbox_data == "1c4b37dc276ea673")).flatten()
    # label = LabelName_bbox_data[img_index]
    # data = ImgeID_bbox_data[img_index[0]]

    # openimage_cat = []
    # cat2img = {}
    for img_id in OpenImagesAPI.imgs:
        # img_id = img['id']
        img_index = np.array(np.where(ImgeID_bbox_data == img_id)).flatten()
        labels = LabelName_bbox_data[img_index]
        for item in labels:
            cls_index = np.array(np.where(cls_cls_data == item)).flatten()
            cat = cat_cls_data[cls_index]
            if cat[0] in Cat:
                Instance_num_dict[cat[0]] += 1
        # label_unique = np.unique(label)
        # for item in label_unique:
        #     cls_index = np.array(np.where(cls_cls_data == item)).flatten()
        #     cats = cat_cls_data[cls_index]
        #     cats_unique = np.unique(cats).tolist()
        #     openimage_cat.extend(cats_unique)
        #     # cat2img[cats[0]] = [img_id] if not cat2img.__contains__(cats[0]) else cat2img[cats[0]].extend(img_id)
        #     if not cat2img.__contains__(cats[0]):
        #         cat2img[cats[0]] = [img_id]
        #     else:
        #         # l = cat2img[cats[0]]
        #         # l.append(img_id)
        #         cat2img[cats[0]].append(img_id)
        # for item in labels:
        #     Instance_num_dict[item] += 1
        with open('./InstancesStat_ours_OpenImages_cocoCAT_v2.txt','w') as f:
            for key, value in Instance_num_dict.items():
                f.write("{}:{}".format(key,value))
                f.write('\n')
            f.write('totoal:{}'.format(sum([value for key, value in Instance_num_dict.items()])))
        f.close()


OpenImages_testset_COCO_cat()