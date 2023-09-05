import os
import random
import json
import shutil
import json.decoder
def adjust_txt_format():
    file_path = "C:/Users/Administrator/Desktop/benchmark/Main"
    tgt_path ="E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/Main"
    data = []
    processed_data = []
    for path in os.listdir(file_path):
        with open(os.path.join(file_path,path),'r') as f:
            data = f.readlines()
            processed_data = [item.split('.')[0].strip('\n') for item in data]
        f.close()
        with open(os.path.join(tgt_path, path),'w') as f:
            for item in processed_data:
                f.write(item)
                f.write('\n')
        f.close()

def stat_pic_num(file_path):
    # file_path = "E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/Main"
    num = 0
    cat2num_dict = {}
    for path in os.listdir(file_path):
        with open(os.path.join(file_path, path), 'r') as f:
            data = f.readlines()
            processed_data = [item.strip('\n') for item in data]
            cat2num_dict[path.split('.')[0]] = len(processed_data)
            num += len(processed_data)
        f.close()
    print(num)
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

def generate_cls_balanced_benchmark():

    random.seed = 0
    file_path = "E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/Raw_data"
    tgt_path ="E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/BalancedBenchmark/Main"
    for path in os.listdir(file_path):
        with open(os.path.join(file_path,path),'r') as f:
            data = f.readlines()
        f.close()
        random_sampled_data = random.sample(data, min(random.randint(50, 60), len(data)))
        with open(os.path.join(tgt_path, path),'w') as f:
            for item in random_sampled_data:
                f.write(item)
        f.close()

def generate_cls_balanced_validation_set():
    random.seed = 0
    raw_data_path = "E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/Raw_data"
    test_path ="E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/BalancedBenchmark/ImageSets/Main_test"
    val_path = "E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/BalancedBenchmark/ImageSets/Main_val"
    val_path = "E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/BalancedBenchmark/ImageSets/Main_val_v2"
    for path in os.listdir(raw_data_path):
        with open(os.path.join(test_path,path),'r') as f:
            test_data = f.readlines()
            test_data = [item.strip('\n') for item in test_data]
        f.close()
        with open(os.path.join(raw_data_path,path) ,'r') as f:
            raw_data = f.readlines()
            raw_data = [item.strip('\n') for item in raw_data]
        f.close()
        with open(os.path.join(val_path, path),'w') as f:
            raw_val_data = [item for item in raw_data if item not in test_data]
            # val_data = random.sample(raw_val_data, min(random.randint(10, 20),len(raw_val_data)))
            if len(raw_val_data) >=15:
                val_data = random.sample(raw_val_data, min(random.randint(20, 25), len(raw_val_data)))
            else :val_data = []
            for item in val_data:
                f.write(item)
                f.write('\n')
        f.close()


def merge_dataset(file_path = "E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/BalancedBenchmark/Main",
                  tgt_path ="E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/BalancedBenchmark/merged_data.txt"):
    # file_path = "E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/BalancedBenchmark/Main"
    # tgt_path ="E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/BalancedBenchmark/merged_data.txt"
    processed_data = []
    # before_deduplicated_num = 0
    for path in os.listdir(file_path):
        with open(os.path.join(file_path, path), 'r') as f:
            data = f.readlines()
        f.close()
        processed_data.extend([item.strip('\n') for item in data])
        for item in data:
            if item.strip('\n') =='':
                print(path)
    deduplicated_data = list(set(processed_data))
    with open(tgt_path,'w') as f:
        for item in deduplicated_data:
            f.write(item)
            f.write('\n')
    f.close()
    print("Datasize before duplicated: {}".format(len(processed_data)))
    print("Datasize after duplicated: {}".format(len(deduplicated_data)))

def stat_openimages_ood():
    file_path = "E:/Datasets/ObjectDetection/OpenImages/OpenImages/ood_classes_rm_overlap/COCO-Format/val_coco_format.json"
    tgt_path = "E:/Datasets/ObjectDetection/OpenImages/OpenImages/ood_classes_rm_overlap/COCO-Format/openimages_vos_image_ids.txt"
    with open(file_path,'r') as f:
        json_data = json.load(f)
    f.close()
    image_ids = []
    for item in json_data['images']:
        image_ids.append(item['id'])
    with open(tgt_path,'w') as f:
        for item in image_ids:
            f.write(item)
            f.write('\n')
    f.close()

def move_oe():
    # with open("E:/Datasets/AAAI2023/OE_Analysis/oe_v3.txt", "w") as f:
    #     dicts = f.readlines()
    src_path = "D:\Project\RAL2023\Data\Subset_Subset_ImageNet_seed0"
    with open("E:/Datasets/AAAI2023/OE_Analysis/removed_ids_v3.txt", "r") as f:
        ids = f.readlines()
        ids = [item.strip('\n') for item in ids]
    f.close()
    for i in range(1,1001):
        if str(i) not in ids:
            shutil.copytree(os.path.join(src_path,str(i).rjust(4,'0')), os.path.join("E:/Datasets/AAAI2023/OE_Analysis/oe", str(i).rjust(4, '0')))

def collect_oe():
    for path in os.listdir(os.path.join("E:/Datasets/AAAI2023/OE_Analysis/oe")):
        for dir in os.listdir(os.path.join("E:/Datasets/AAAI2023/OE_Analysis/oe",path)):
            shutil.copy(os.path.join("E:/Datasets/AAAI2023/OE_Analysis/oe", path, dir),os.path.join("E:/Datasets/AAAI2023/OE_Analysis/oe_data"))


# generate_cls_balanced_benchmark()
stat_pic_num("E:\Datasets\AAAI2023\OpenImagesCOCO_Ours_Trainingset_Version\BalancedBenchmark\ImageSets\Main_val")
# merge_dataset()
# stat_openimages_ood()
# move_oe()
# move_oe()
# collect_oe()
# generate_cls_balanced_validation_set()
# merge_dataset("E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/BalancedBenchmark/ImageSets/Main_val_v2",
#             "E:/Datasets/AAAI2023/OpenImagesCOCO_Ours_Trainingset_Version/BalancedBenchmark/ImageSets/Layout/val_v2.txt")
# stat_pic_num("E:\Datasets\AAAI2023\OpenImagesCOCO_Ours_Trainingset_Version\BalancedBenchmark\ImageSets\Main_val_v2")