
"""
this module arranges the data in format showed in
https://www.v7labs.com/blog/coco-dataset-guide
RLE: https://www.kaggle.com/code/leahscherschel/run-length-encoding
"""
import json
import os.path as path
import pandas as pd
from pycocotools.coco import COCO

import numpy as np

RAW_DATA_ROOT = "C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\raw"
COCO_DATA_ROOT = "C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\coco_data"


# Create dictionary from polygon.json
def read_annotations(filename):
    with open(RAW_DATA_ROOT + "\\" + filename, 'r') as file:
        annotation_dict = {}
        line = file.readline()
        while line:
            # do the work
            annotation = json.loads(line)
            annotation_dict[annotation['id']] = annotation['annotations']
            line = file.readline()
        return annotation_dict

def get_cat_id(cat_id):
    match cat_id:
        case "blood_vessel":
            return 0
        case "glomerulus":
            return 1
        case "unsure":
            return 2
        case _:
            raise ValueError("no option matching for category id")

def get_bbox(annotation):
    coords = annotation["coordinates"]
    coords_arr = np.asarray(coords, dtype=np.float32)
    # coordinates array has only one element in it
    x_min, y_min = coords_arr[0].min(0)
    x_max, y_max = coords_arr[0].max(0)
    wd, ht = float(x_max - x_min), float(y_max - y_min)
    bbox = [float(x_min), float(y_min), wd, ht]
    return bbox

def prepare_coco_json(filename, df):
    coco_json = {
        "info": {},
        "categories": [
            {
                "id": 0,
                "name": "blood_vessel"
            },
            {
                "id": 1,
                "name": "glomerulus"
            },
            {
                "id": 2,
                "name": "unsure"
            }
        ]
    }
    all_annotation_dict = read_annotations(filename)
    df_ids = df['id'].values.tolist()
    annotation_dict = dict(filter(lambda x: x[0] in df_ids, all_annotation_dict.items()))
    images = []
    coco_annotations = []
    annotation_id = 1
    image_seq_id = 1
    for image_id, annotations in annotation_dict.items():
        images.append({
            "id": image_seq_id,
            "width": 512,
            "height": 512,
            "file_name": f"{image_id}.tif"
        })
        annotations_arr = np.asarray(annotations)
        for annotation in annotations_arr:
            bbox = get_bbox(annotation)
            coco_annotations.append({
                "id" : annotation_id,
                "image_id": image_seq_id,
                "category_id": get_cat_id(annotation["type"]),
                "segmentation" : np.asarray(annotation["coordinates"]).reshape(1,-1).tolist(),
                "is_crowd": 0,
                "bbox": bbox,
                "area": float(bbox[2]* bbox[3])
            })
            annotation_id+=1
        image_seq_id+=1
    coco_json["images"] = images
    coco_json["annotations"] = coco_annotations
    return coco_json



# read tile metadata
df = pd.read_csv('C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\raw\\tile_meta.csv')
ds3 = df[(df["dataset"] == 3)]
# tiles from Dataset 1 have annotations that have been expert reviewed. So, it is used for training.
wsi_1_ds_1 = df[(df["source_wsi"] == 1) & (df["dataset"] == 1)]
wsi_2_ds_1 = df[(df["source_wsi"] == 2) & (df["dataset"] == 1)]

# concat data from dataset 1
coco_data = pd.concat([
    wsi_1_ds_1, wsi_2_ds_1
], axis=0)


train1 = wsi_1_ds_1[wsi_1_ds_1['i'] <= wsi_1_ds_1['i'].quantile(0.8)]
val1 = wsi_1_ds_1[wsi_1_ds_1['i'] > wsi_1_ds_1['i'].quantile(0.8)]

train2 = wsi_2_ds_1[wsi_2_ds_1['i'] <= wsi_2_ds_1['i'].quantile(0.8)]
val2 = wsi_2_ds_1[wsi_2_ds_1['i'] > wsi_2_ds_1['i'].quantile(0.8)]

train = pd.concat([train1, train2], axis = 0)
val = pd.concat([val1, val2], axis = 0)

# prepare and save coco dataset - train + test
coco_data_json = prepare_coco_json("polygons.jsonl", coco_data)

with open(COCO_DATA_ROOT + "\\" + "coco_data.json", "w") as f:
    json.dump(coco_data_json, f, indent=4)

# prepare and save coco dataset - train
coco_train_data_json = prepare_coco_json("polygons.jsonl", train)

with open(COCO_DATA_ROOT + "\\" + "coco_train_data.json", "w") as f:
    json.dump(coco_train_data_json, f, indent=4)

# prepare and save coco dataset - test
coco_test_data_json = prepare_coco_json("polygons.jsonl", val)

with open(COCO_DATA_ROOT + "\\" + "coco_test_data.json", "w") as f:
    json.dump(coco_test_data_json, f, indent=4)

# load the COCO dataset
coco = COCO(COCO_DATA_ROOT + "\\" + "coco_data.json")
categories = coco.loadCats(coco.getCatIds())
print("Categories:", [cat['name'] for cat in categories])

# validate if the COCO dataset is valid by reading the json
image_ids = coco.getImgIds()
for img_id in image_ids:
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    print(f"Image ID: {img_id}, Annotations: {len(annotations)}")



