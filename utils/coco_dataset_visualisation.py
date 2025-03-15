import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Polygon
from pycocotools import mask
from pycocotools.coco import COCO
import numpy as np

RAW_DATA_ROOT = "C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\raw"
COCO_DATA_ROOT = "C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\coco_data"
coco = COCO(COCO_DATA_ROOT + "\\" + "coco_train_data.json")
image_ids = coco.getImgIds()
for idx in range(0,5): #range(0,len(image_ids)):
    # load image metadata
    image = coco.loadImgs(image_ids[idx])[0]
    img = cv2.imread( f"{RAW_DATA_ROOT + "\\train\\" + image['file_name']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get annotations for the image
    ann_ids = coco.getAnnIds(imgIds=image['id'])
    annotations = coco.loadAnns(ann_ids)

    # plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    ax = plt.gca()

    # overlay annotations
    for ann in annotations:
        # Draw bounding box
        x, y, w, h = ann['bbox']
        category = ann['category_id']
        if(category == 0):
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        elif(category == 1):
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        else:
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)

        # draw segmentation
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):  # Polygon
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) // 2, 2))
                    polygon = Polygon(poly, linewidth=2, edgecolor='b', facecolor='none')
                    ax.add_patch(polygon)
            else:  # RLE
                binary_mask = mask.decode(ann['segmentation'])
                plt.imshow(binary_mask, alpha=0.5)

    plt.axis('off')
    plt.show()

