import fiftyone as fo
import fiftyone.zoo as foz

# dataset = fo.zoo.load_zoo_dataset(
#                       "open-images-v6",
#                       split="train",
#                       label_types=["detections"],
#                       # classes=['Laptop', 'Computer mouse', 'Remote control', 'Computer keyboard', 'Mobile phone'],
#                       # classes=['Backpack'],#'Handbag', 'Backpack'
#                       # classes = ["Sandwich", "Hot dog", "Pizza", "Doughnut", "Cake"],
#                       classes=['Caterpillar'],
#                       seed=51,
#                       max_samples = 2000,
#                   )
#
# session = fo.launch_app(dataset,port=5153)
# session.wait()



with open('E:\Datasets\AAAI2023\OpenImagesCOCO_Ours_Trainingset_Version\BalancedBenchmark\ImageSets\Layout/val_v3.txt','r') as f:
    image_ids = []
    for item in f.readlines():
        image_ids.append(item.strip('\n'))
f.close()
# image_ids = ["0a0b97aa8518cdb2",
# "1736dc2684d8a9f2",
# "1b8775c9339f4fc6",
# "242a871d77daf35e"]
# image_ids_stop_sign = ["2a9b52b33ae6a8d8",
# "34250993aa586a64",
# "5f1cddc0f3c6835a",
# "5d5ef12f514980ad",
# "690c26d5132340d8",
# "8db5213d126cd643"]
# image_ids_ball =["507ef56f499ac8bd",
# "6854395dd5a0a9ed",
# "69a572c7e396d7f2",
# "757e756ba723367d",
# "6c2ed6121bfcb962",
# "08162d9b963a3f5a",
# "155503e911df29ec",
# "6b57303bb25aebda",
# "1a2358430af31b48"]
dataset = fo.zoo.load_zoo_dataset(
              "open-images-v7",
              split="train",
              label_types=["detections"],
              # image_ids_file = '.\OpenImages_coco_cat_ours_image_ids.txt'
              # image_ids = ['000ce1f42b8b5d06']
              image_ids = image_ids
          )
# os.makedirs(os.path.join("E:\Datasets\ObjectDetection\OL-OODBENCHMARK", 'Broccoli'))
# shutil.move('C:/Users/Administrator/fiftyone/open-images-v7/test/data', os.path.join('E:/Datasets/AAAI2023/', 'test_image_id'))

session = fo.launch_app(dataset,port=5152)
session.wait()