import os.path as osp
import json
import argparse
import random

parser = argparse.ArgumentParser(description='data dir path and discount rate')
parser.add_argument('--data_dir', type=str, 
                    default="./refer/data/reverie/", help='data dir path')
parser.add_argument('--disc_rate', type=int, 
                    default=2, help='discount rate')
parser.add_argument('--no_shuffle', action='store_false', help='shuffle or not')
parser.add_argument('--reverie', action='store_true', help='dataset is reverie or not')
args = parser.parse_args()
random.seed(42)

DATA_DIR = args.data_dir
if args.is_reverie: 
    json_path = osp.join(DATA_DIR, "reverie_ant.json")
else:
    json_path = osp.join(DATA_DIR, "instances.json")
save_path = osp.join(DATA_DIR, "disc_inst.json")
id_path = osp.join(DATA_DIR, "ids_list.json")

load_json = open(json_path)
save_json = open(save_path, "w")
id_json = open(id_path, "w")

if args.is_reverie: 
    dic = json.load(load_json)
    dic_train = [data for data in dic["images"] if data["split"] == "train"] 
    dic_other = [data for data in dic["images"] if not data["split"] == "train"] 
    len_img = len(dic_train)
    dis_len = len_img // args.disc_rate
    random.shuffle(dic_train)
    dic["images"] = dic_train[:dis_len] + dic_other
else:
    dic = json.load(load_json)
    len_img = len(dic["images"])
    dis_len = len_img // args.disc_rate
    random.shuffle(dic["images"])
    dic["images"] = dic["images"][:dis_len]

print(f"discount rate:{args.disc_rate}")
print(f"before #samples:{len_img}")
print(f"after  #samples:{dis_len}")

id_list = dict()
id_list["img"] = list()
id_list["ant"] = list()
id_list["ctg"] = list()
for ids in dic["images"]:
    id_list["img"].append(ids["id"])

dic_tmp = dic
for i, ants in reversed(list(enumerate(dic_tmp["annotations"]))): 
    if ants["image_id"] not in id_list["img"]:
        del dic["annotations"][i]
    else:
        id_list["ant"].append(ants["id"])
        id_list["ctg"].append(ants["category_id"])

json.dump(dic, save_json)
json.dump(id_list, id_json)
    
