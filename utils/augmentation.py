from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from pycocotools import mask as maskUtils


RAW_DATA_ROOT = "C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\raw"
COCO_DATA_ROOT = "C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\coco_data"
AUGMENTED_DATA_ROOT = "C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\augmented_data"

coco = COCO(COCO_DATA_ROOT + "\\" + "coco_data.json")
image_ids = coco.getImgIds()

print("Total Images:", len(image_ids))

IMAGE_SIZE = 512  # Resize images


def load_image(image_id):
    """loads an image from COCO dataset and resize it"""
    coco_path = COCO_DATA_ROOT+"\\data"
    image_info = coco.loadImgs(image_id)[0]

    image_path = os.path.join(coco_path, image_info["file_name"])
    print(image_path)
    # loads the image in color mode (ignoring transparency).
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to 256 for experimentation

    return image


def create_binary_mask(image_id, height=512, width=512, selected_classes=[0, 1, 2]):
    """
    Create a binary mask for a given image ID and selected classes.
    """
    mask = np.zeros((height, width, len(selected_classes)), dtype=np.uint8)  # Multi-channel binary mask

    # Get all annotations for this image
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        category_id = ann["category_id"]
        if category_id in selected_classes:  # Only process selected classes

            class_index = selected_classes.index(category_id)  # Get channel index
            segmentation = ann["segmentation"]

            # Convert segmentation to mask
            rle = maskUtils.frPyObjects(segmentation, height, width)
            binary_mask = maskUtils.decode(rle).sum(axis=2).astype(np.uint8)

            # Assign to the correct channel
            mask[:, :, class_index] = np.maximum(mask[:, :, class_index], binary_mask)

    return mask  # Shape: (height, width, 3)


def preprocess_mask(binary_mask, target_size=(512, 512)):
    """
    Resize binary masks to a fixed target size.
    """
    resized_mask = np.zeros((*target_size, binary_mask.shape[-1]), dtype=np.uint8)

    for i in range(binary_mask.shape[-1]):  # Resize each class separately
        resized_mask[:, :, i] = cv2.resize(binary_mask[:, :, i], target_size, interpolation=cv2.INTER_NEAREST)

    return resized_mask


def convert_to_one_hot(binary_mask):
    """
    Convert a binary mask (H, W, C) into a one-hot encoded mask (H, W, num_classes+1).
    Background is treated as class 0.
    """
    num_classes = binary_mask.shape[-1] + 1  # Include background
    one_hot_mask = np.zeros((*binary_mask.shape[:2], num_classes), dtype=np.uint8)

    for i in range(binary_mask.shape[-1]):  # Assign each class a different label
        one_hot_mask[:, :, i + 1] = binary_mask[:, :, i]  # Shift by 1 to reserve background (0)
        one_hot_mask[:, :, 0] += binary_mask[:, :, i]
    one_hot_mask[:, :, 0] = 1 - one_hot_mask[:, :, 0]
    return one_hot_mask


def load_mask(image_id, height=512, width=512, selected_classes=[0, 1, 2]):
    # Create binary mask
    binary_mask = create_binary_mask(image_id, height, width, selected_classes)
    binary_mask_resized = preprocess_mask(binary_mask, (height, width))
    one_hot_mask = convert_to_one_hot(binary_mask_resized)
    return one_hot_mask


image_ids = coco.getImgIds()

# load sample image and mask
sample_id = image_ids[421]  # 248 has all 3 classes

sample_image = load_image(sample_id)

sample_mask = load_mask(sample_id)

# display Image & Mask
# plt.figure(figsize=(10, 5))
#
# plt.imshow(sample_image)
# plt.axis("off")
# plt.title("Image")
# plt.show()
#
# plt.imshow(sample_mask[:, :, 0], cmap="gray")
# plt.axis("off")
# plt.title("Category 0: Segmentation Mask")
# plt.show()
#
# plt.imshow(sample_mask[:, :, 1], cmap="gray")
# plt.axis("off")
# plt.title("Category 1: Segmentation Mask")
# plt.show()
#
# plt.imshow(sample_mask[:, :, 2], cmap="gray")
# plt.axis("off")
# plt.title("Category 2: Segmentation Mask")
# plt.show()

# plt.imshow(sample_mask[:, :, 3], cmap="gray")
# plt.axis("off")
# plt.title("Category 3: Segmentation Mask")
#
# plt.show()

X_train = np.array([load_image(img_id) for img_id in image_ids[:420]])  # Load first 420 images
Y_train = np.array([load_mask(img_id) for img_id in image_ids[:420]])
print("Image Data Shape:", X_train.shape)
print("Mask Data Shape:", Y_train.shape)


import numpy as np

#compute class freq and weights
def compute_class_weights(masks):
    pixels_per_class = masks.sum(axis=(0,1,2))
    total_pixels = pixels_per_class.sum()
    class_weights = total_pixels / (len(pixels_per_class) * pixels_per_class)
    return class_weights, pixels_per_class

class_weights, class_counts = compute_class_weights(Y_train)

# identify rare classes based on threshold
def get_rare_classes(class_counts, percentile=25):
    print(class_counts)
    threshold = np.percentile(class_counts, percentile)
    rare_classes = np.where(class_counts <= threshold)[0]
    return rare_classes

rare_classes = get_rare_classes(class_counts)
print(f"rare_classes: {rare_classes}")



#Filter the dataset to find which images contain rare classes:
def contains_rare_class(mask, rare_classes):
    mask_classes_present = np.unique(np.argmax(mask, axis=-1))
    return any(rc in mask_classes_present for rc in rare_classes)
images = []
masks= []
for img, mask in zip(X_train, Y_train):
    if contains_rare_class(mask, rare_classes):
        for _ in range(2):  # duplicate rare class samples
            images.append(img)
            masks.append(mask)
X_train = np.concatenate((X_train,np.array(images)), axis=0)
Y_train = np.concatenate((Y_train, np.array(masks)), axis=0)

rare_class_indices = [i for i, mask in enumerate(Y_train) if contains_rare_class(mask, rare_classes)]

#print(f"rare_class_indices: {rare_class_indices}")

import albumentations as A

# Aggressive augmentation for rare classes
aggressive_aug = A.Compose([
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.8),
    A.RandomRotate90(p=0.8),
    #A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.7),
    A.GridDistortion(p=0.7),
    A.RandomBrightnessContrast(p=0.5),
], additional_targets={'mask': 'mask'})

# Moderate augmentation for common classes
moderate_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], additional_targets={'mask': 'mask'})

def one_hot_to_int(mask):
    return np.argmax(mask, axis=-1).astype(np.uint8)

def int_to_one_hot(mask_int, num_classes):
    return np.eye(num_classes)[mask_int]

def conditional_augment(image, mask_one_hot, rare_classes):
    mask_int = one_hot_to_int(mask_one_hot)
    num_classes = mask_one_hot.shape[-1]

    # Decide augmentation pipeline
    if contains_rare_class(mask_one_hot, rare_classes):
        augmented = aggressive_aug(image=image, mask=mask_int)
    else:
        augmented = moderate_aug(image=image, mask=mask_int)

    aug_image = augmented['image']
    aug_mask_int = augmented['mask']
    aug_mask_one_hot = int_to_one_hot(aug_mask_int, num_classes)

    return aug_image, aug_mask_one_hot

augmented_images = []
augmented_masks = []

for img, mask in zip(X_train, Y_train):
    aug_img, aug_mask = conditional_augment(img, mask, rare_classes)
    augmented_images.append(aug_img)
    augmented_masks.append(aug_mask)

X_aug = np.array(augmented_images)
Y_aug = np.array(augmented_masks)

print(f"{X_aug.shape}{Y_aug.shape}")


X_train1 = np.concatenate((X_train,X_aug), axis=0)
Y_train1 = np.concatenate((Y_train, Y_aug), axis=0)


X_train = X_train1
Y_train = Y_train1

import os
import numpy as np
import cv2
import json
from pycocotools import mask as mask_utils
from skimage import measure
from tqdm import tqdm

def convert_numpy_to_coco(images, masks, output_dir, class_names, prefix="image"):
    os.makedirs(output_dir, exist_ok=True)

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for idx, name in enumerate(class_names):
        coco_output["categories"].append({
            "id": idx + 1,
            "name": name,
            "supercategory": "object"
        })

    ann_id = 1
    for i in tqdm(range(images.shape[0]), desc="Converting to COCO"):
        image = images[i]
        mask = masks[i]
        height, width = image.shape[:2]

        # Save image to disk (optional)
        file_name = f"{prefix}_{i:04d}.png"
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Add image metadata
        coco_output["images"].append({
            "id": i + 1,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # Process each class
        for class_idx in range(mask.shape[-1]):
            binary_mask = mask[:, :, class_idx].astype(np.uint8)
            if binary_mask.sum() == 0:
                continue  # Skip empty masks

            contours = measure.find_contours(binary_mask, 0.5)
            segmentation = []

            for contour in contours:
                contour = np.flip(contour, axis=1)
                if len(contour) >= 6:  # Must be at least 3 points (6 coords)
                    segmentation.append(contour.ravel().tolist())

            if not segmentation:
                continue

            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            area = float(mask_utils.area(rle))
            bbox = mask_utils.toBbox(rle).tolist()

            annotation = {
                "id": ann_id,
                "image_id": i + 1,
                "category_id": class_idx + 1,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }

            coco_output["annotations"].append(annotation)
            ann_id += 1

    # Save final COCO JSON
    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"COCO dataset saved to {output_dir}\\annotations.json")

convert_numpy_to_coco(
    images=X_train,               # shape: (N, H, W, 3)
    masks=Y_train,         # shape: (N, H, W, num_classes)
    output_dir=AUGMENTED_DATA_ROOT + "\\data",
    class_names=["background", "blood_vessel", "vein", "other"]
)


# os.makedirs(AUGMENTED_DATA_ROOT + "\\data", exist_ok=True)
#
# for i in range(X_train.shape[0]):
#     img = X_train[i]
#     img_path = os.path.join(AUGMENTED_DATA_ROOT + "\\data", f"image_{i:03d}.tif")
#     cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

