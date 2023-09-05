from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.catalog import DatasetCatalog
from torch.utils.data.sampler import SequentialSampler
import os
from detectron2.data.detection_utils import read_image, convert_image_to_rgb
from detectron2.utils.visualizer import Visualizer
import cv2

root = "/home/leiyvtian/datasets/coco"
# register_pascal_voc("voc_2007_trainval",os.path.join(root,"VOC2007"),"trainval","2007")
register_coco_instances("mixed_ood",metadata=dict(),
                        json_file=os.path.join(root,"annotations","instances_val2017_mixed_OOD.json"),
                        image_root=os.path.join(root,"val2017"))
data_dicts=DatasetCatalog.get("mixed_ood")
sampler = SequentialSampler(data_dicts)
for i,index in enumerate(sampler):
    dict = data_dicts[index]
    img = read_image(dict['file_name'],format='RGB')
    visualizer = Visualizer(img,metadata = {'thing_classes':["unkonwn"]})

    image_visual = visualizer.draw_dataset_dict(dict)
    # cv2.imshow('sample',image_visual.get_image())
    image_visual.save(os.path.join('./visualize/mixed_ood',str(dict['image_id'])+'.png'))

