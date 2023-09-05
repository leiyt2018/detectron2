import os
from collections import ChainMap

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Construct VOC metadata
VOC_THING_CLASSES = ['person',
                     'bird',
                     'cat',
                     'cow',
                     'dog',
                     'horse',
                     'sheep',
                     'airplane',
                     'bicycle',
                     'boat',
                     'bus',
                     'car',
                     'motorcycle',
                     'train',
                     'bottle',
                     'chair',
                     'dining table',
                     'potted plant',
                     'couch',
                     'tv',
                     ]
# Construct COCO metadata
COCO_THING_CLASSES = MetadataCatalog.get('coco_2017_train').thing_classes
COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID = MetadataCatalog.get(
    'coco_2017_train').thing_dataset_id_to_contiguous_id

VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(20)]))

def setup_all_datasets(dataset_dir):
    """
    Registers all custom datasets

    Args:
        dataset_dir(str): path to dataset directory

    """
    setup_coco_mixed_dataset(os.path.join(dataset_dir,"coco"))
    setup_coco_extended_ood_dataset(os.path.join(dataset_dir,"coco"))
    setup_voc_dataset(os.path.join(dataset_dir,"VOCdatasets/VOC_0712_converted"))
    setup_voc_completely_annotation_dataset(os.path.join("VOCdatasets/VOC_0712_converted"))

def setup_coco_mixed_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'val2017')
    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', 'instances_val2017_mixed_ID.json')
        # dataset_dir, 'annotations', 'COCO_ID.json')

    register_coco_instances(
        "coco_mixed_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_mixed_val").thing_classes = VOC_THING_CLASSES
    MetadataCatalog.get(
        "coco_mixed_val").thing_dataset_id_to_contiguous_id = VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_coco_extended_ood_dataset(dataset_dir):
    test_image_dir = os.path.join(dataset_dir, 'val2017')

    test_json_annotations = os.path.join(
        dataset_dir, 'annotations', "instances_val2017_coco_ood.json")#'instances_val2017_extended_ood.json')

    register_coco_instances(
        "coco_extended_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    # import ipdb; ipdb.set_trace()
    metadata_COCO_THING_CLASSES = COCO_THING_CLASSES
    metadata_COCO_THING_CLASSES = ["unknow object"]
    
    # metadata_COCO_THING_CLASSES.append("unknow object")
    # MetadataCatalog.get(
    #     "coco_extended_ood_val").thing_classes = metadata_COCO_THING_CLASSES
    # metadata_COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID = metadata.COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID
    # metadata_COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID[91] = 80    
    # MetadataCatalog.get(
    #     "coco_extended_ood_val").thing_dataset_id_to_contiguous_id = metadata_COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID

    MetadataCatalog.get(
        "coco_extended_ood_val").thing_classes = metadata_COCO_THING_CLASSES
    MetadataCatalog.get(
        "coco_extended_ood_val").thing_dataset_id_to_contiguous_id = {81: 0}

def setup_voc_dataset(dataset_dir):
    # train_image_dir = os.path.join(dataset_dir, 'JPEGImages')
    # # else:
    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')

    # train_json_annotations = os.path.join(
    #     dataset_dir, 'voc0712_train_all.json')
    test_json_annotations = os.path.join(
        dataset_dir, 'val_coco_format.json')

    # register_coco_instances(
    #     "voc_custom_train",
    #     {},
    #     train_json_annotations,
    #     train_image_dir)
    # MetadataCatalog.get(
    #     "voc_custom_train").thing_classes = metadata.VOC_THING_CLASSES
    # MetadataCatalog.get(
    #     "voc_custom_train").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID

    register_coco_instances(
        "voc_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_custom_val").thing_classes = VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_val").thing_dataset_id_to_contiguous_id = VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID

def setup_voc_completely_annotation_dataset(dataset_dir):
    # train_image_dir = os.path.join(dataset_dir, 'JPEGImages')
    # train_json_annotations = os.path.join(
    #     dataset_dir, 'voc0712_train_all.json')
    # register_coco_instances(
    #     "voc_custom_train",
    #     {},
    #     train_json_annotations,
    #     train_image_dir)
    # MetadataCatalog.get(
    #     "voc_custom_train").thing_classes = metadata.VOC_THING_CLASSES
    # MetadataCatalog.get(
    #     "voc_custom_train").thing_dataset_id_to_contiguous_id = metadata.VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID

    test_image_dir = os.path.join(dataset_dir, 'JPEGImages')
    test_json_annotations = os.path.join(
        dataset_dir, 'voc0712_train_completely_annotation200.json')
    register_coco_instances(
        "voc_completely_annotation_pretest",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_completely_annotation_pretest").thing_classes = VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_completely_annotation_pretest").thing_dataset_id_to_contiguous_id = VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID

