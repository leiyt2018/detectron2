import json

with open('/home/leiyvtian/datasets/coco/annotations/instances_val2017_ood_rm_overlap.json','r') as f:
    file = json.load(f)
f.close()

with open('/home/leiyvtian/RAL/detectron2/OE_Dataprocess/remained_no_nontest_ood.txt','r') as f:
    _remained_id = f.readlines()
    remained_id = []
    for item in _remained_id:
        remained_id.append(item.strip('\n'))
f.close()

file['annotations'] = []
images = []
for image in file['images']:
    if str(image['id']) in remained_id:
        images.append(image)
file['images'] = images

with open("./instances_val2017_no_nococoood.json", 'w') as fp:
    json.dump(file, fp)
fp.close()

with open('./instances_val2017_no_nococoood.json','r') as f:
    file = json.load(f)
f.close()

