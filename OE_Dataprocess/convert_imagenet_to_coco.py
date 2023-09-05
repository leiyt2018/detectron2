# from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.data.catalog import DatasetCatalog
import sys
# sys.path.append(r'/home/leiyvtian/RAL/VOS_OE')
# from detection.core.datasets.pascal_voc_oe import register_pascal_voc_oe
from modeling.pascal_voc_oe import register_pascal_voc_oe

dir_name='E:/Datasets/ObjectDetection/ImageNet1k_Val_OE'
dataset_name = 'ImageNet1k_Val_OE_v3'
split = 'train_oe_v3'
year = '2012'

register_pascal_voc_oe(dataset_name,dir_name,split,year) ## dir_name:数据集根目录，split：划分方式，val.txt, train.txt, trainval.txt
data_dicts = DatasetCatalog.get(dataset_name) ## 结果获得一个数据字典，即datset

from detectron2.data.datasets.coco import convert_to_coco_json
convert_to_coco_json(dataset_name,'E:/Datasets/ObjectDetection/ImageNet1k_Val_OE/ImageNet1k_Val_OE_v3.json')