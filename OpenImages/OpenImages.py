import fiftyone as fo
import fiftyone.zoo as foz
import os, shutil
# dataset = foz.load_zoo_dataset(
#     "open-images-v7",
#     split="validation,test",
#     max_samples=300,
#     seed=51,
#     shuffle=True,
# )
# session = fo.launch_app(dataset.view())


# import json
# bbox_labels_600_hierachy_path = "/home/leiyvtian/RAL/detectron2/OpenImages/bbox_labels_600_hierarchy.json"
#
# with open(bbox_labels_600_hierachy_path,'r') as f:
#     bbox_labels = json.load(f)
# f.close()
# print(bbox_labels)
#
# b = "/m/0jbk"

#
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
# total_num = 0
# with open('./cat2numv2.txt','w') as f:
#     for key, value in OPEN_IMAGES_TO_COCO.items():
#         # dataset = fo.zoo.load_zoo_dataset(
#         #               "open-images-v7",
#         #               split="test",
#         #               label_types=["detections"],
#         #               classes=[key],
#         #               max_samples=100,
#         #           )
#         # os.makedirs(os.path.join("E:\Datasets\ObjectDetection\OL-OODBENCHMARK",value))
#         # shutil.move('C:/Users/Administrator/fiftyone/open-images-v7/test/data', os.path.join('E:\Datasets\ObjectDetection\OL-OODBENCHMARK_OPENIMAGES_TEST',value))
#         # shutil.move(os.path.join('E:\Datasets\ObjectDetection\OL-OODBENCHMARK', value,'data'),os.path.join('E:\Datasets\ObjectDetection\OL-DATASET',value))
#         total_num += len(os.listdir(os.path.join('E:\Datasets\Predictions\OL-OODBENCHMARK_OPENIMAGES_TEST_predictions',value)))
#         f.write("{}:{}\n".format(key,len(os.listdir(os.path.join('E:\Datasets\Predictions\OL-OODBENCHMARK_OPENIMAGES_TEST_predictions',value)))))
#     f.write("total num:{}".format(total_num))
# print(total_num)
# with open('.\OpenImages_coco_cat_ours_image_ids.txt','r') as f:
#     image_ids = []
#     for item in f.readlines():
#         image_ids.append(item.strip('\n'))
# f.close()
# dataset = fo.zoo.load_zoo_dataset(
#               "open-images-v7",
#               split="test",
#               label_types=["detections"],
#               # image_ids_file = '.\OpenImages_coco_cat_ours_image_ids.txt'
#               image_ids = image_ids
#           )
# # os.makedirs(os.path.join("E:\Datasets\ObjectDetection\OL-OODBENCHMARK", 'Broccoli'))
# # shutil.move('C:/Users/Administrator/fiftyone/open-images-v7/test/data', os.path.join('E:/Datasets/AAAI2023/', 'test_image_id'))
#
# session = fo.launch_app(dataset,port=5152)
# session.wait()

dataset = fo.zoo.load_zoo_dataset(
                      "open-images-v7",
                      split="train",
                      label_types=["detections"],
                      classes=['truck'],
                      max_samples=1000,
                  )
session = fo.launch_app(dataset,port=5153)
session.wait()

