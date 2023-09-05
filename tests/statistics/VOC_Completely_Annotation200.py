from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.catalog import DatasetCatalog
from torch.utils.data.sampler import SequentialSampler
import os
from detectron2.data.detection_utils import read_image, convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer
import cv2
# SPLITS = [
#     ("voc_2007_trainval", "VOC2007", "trainval"),
#     ("voc_2007_train", "VOC2007", "train"),
#     ("voc_2007_val", "VOC2007", "val"),
#     ("voc_2007_test", "VOC2007", "test"),
#     ("voc_2012_trainval", "VOC2012", "trainval"),
#     ("voc_2012_train", "VOC2012", "train"),
#     ("voc_2012_val", "VOC2012", "val"),
# ]
# for name, dirname, split in SPLITS:
#     year = 2007 if "2007" in name else 2012
#     register_pascal_voc(name, os.path.join(root, dirname), split, year)
#     MetadataCatalog.get(name).evaluator_type = "pascal_voc"
SPLITS =("voc_2007_trainval","VOC2007","trainval")
root = "/home/leiyvtian/datasets/VOCdatasets/VOC_0712_converted"
# register_pascal_voc("voc_2007_trainval",os.path.join(root,"VOC2007"),"trainval","2007")
register_coco_instances("voc0712_200",metadata=dict(),
                        json_file=os.path.join(root,"voc0712_train_completely_annotation200.json"),
                        image_root=os.path.join(root,"JPEGImages"))
data_dicts=DatasetCatalog.get("voc0712_200")
sampler = SequentialSampler(data_dicts)
for i,index in enumerate(sampler):
    dict = data_dicts[index]
    img = read_image(dict['file_name'],format='RGB')
    visualizer = Visualizer(img,metadata = {'thing_classes':["person", "bird", "cat", "cow", "dog", "horse", "sheep",
    "airplane", "bicycle", "boat", "bus", "car", "motorcycle", "train", "bottle", "chair","dining table", "potted plant", "couch", "tv"]})

    image_visual = visualizer.draw_dataset_dict(dict)
    # cv2.imshow('sample',image_visual.get_image())
    image_visual.save(os.path.join('./visualize/voc_completely_annotation200',dict['image_id']+'.png'))

