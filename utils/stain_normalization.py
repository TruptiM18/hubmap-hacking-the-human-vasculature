import cv2
import pandas as pd
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import  numpy as np

RAW_DATA_ROOT = "C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\raw"
COCO_DATA_ROOT = "C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\coco_data"

class ReinhardColorNormalizer:
    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target_image):

        #compute the mean and standard deviation of the target image in Lab color space.

        target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB)
        self.target_means, self.target_stds = self._compute_mean_std(target_lab)

    def transform(self, source_image):
        # apply Reinhard normalization to the source image based on the target statistics.

        if self.target_means is None or self.target_stds is None:
            raise ValueError("Target statistics not computed. Call fit(target_image) first.")

        source_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2LAB)
        source_means, source_stds = self._compute_mean_std(source_lab)

        # normalize each channel using the target's mean and standard deviation
        normalized_lab = source_lab.astype(np.float32)
        for i in range(3):  # L, a, b channels
            normalized_lab[:, :, i] = ((normalized_lab[:, :, i] - source_means[i]) / source_stds[i]) * \
                                      self.target_stds[i] + self.target_means[i]

        # clip values to valid Lab color space range
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)

        # convert back to RGB
        normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
        return normalized_rgb

    def _compute_mean_std(self, image_lab):

        #compute mean and standard deviation for each channel in Lab color space.

        means = np.mean(image_lab, axis=(0, 1))
        stds = np.std(image_lab, axis=(0, 1))
        return means, stds

def visualize_source_target_img(image_id, source, target, normalized_image):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(source)
    plt.title(f"Source {image_id}")

    plt.subplot(1, 3, 2)
    plt.imshow(target)
    plt.title("Target Image")

    plt.subplot(1, 3, 3)
    plt.imshow(normalized_image)
    plt.title("Reinhard Normalized Image")
    plt.show()

coco = COCO(COCO_DATA_ROOT + "\\" + "coco_test_data.json")
image_ids = coco.getImgIds()
for idx in range(0,len(image_ids)): #range(0,len(image_ids)):
    image = coco.loadImgs(image_ids[idx])[0]
    source = cv2.imread(f"{RAW_DATA_ROOT + "\\train\\" + image['file_name']}")
    target = cv2.imread(f"{RAW_DATA_ROOT + "\\train\\" + "b78c3f072465.tif"}")

    # convert BGR to RGB (OpenCV loads images in BGR format)
    target_image = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    source_image = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    # apply Reinhard Stain Normalization
    normalizer = ReinhardColorNormalizer()
    normalizer.fit(target_image)
    normalized_image = normalizer.transform(source_image)
    # visualize_source_target_img(image['file_name'],source_image,target_image,normalized_image)

    # save and display result
    cv2.imwrite(COCO_DATA_ROOT+ "\\test\\" + f"{image['file_name'] + ".tif"}", cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))


