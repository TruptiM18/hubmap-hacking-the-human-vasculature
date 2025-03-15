import cv2
import numpy as np
import matplotlib.pyplot as plt
import histomicstk.preprocessing.color_deconvolution as htk
import pandas as pd

RAW_DATA_ROOT = "C:\\Users\\rushik\\Desktop\\AIForHealthcare\\hubmap_data\\raw"

def load_image(image_path):
    # load and convert an image to RGB format
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def plot_histogram(image):
    # plot RGB color histogram
    colors = ('r', 'g', 'b')
    plt.figure(figsize=(8, 4))
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.title("RGB Histogram")
    plt.show()


def compute_lab_stats(image):
    # compute mean and standard deviation in Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_mean, a_mean, b_mean = np.mean(lab_image, axis=(0, 1))
    l_std, a_std, b_std = np.std(lab_image, axis=(0, 1))
    return (l_mean, a_mean, b_mean), (l_std, a_std, b_std)


def stain_deconvolution(image):
    # perform color deconvolution for hematoxylin and eosin separation
    stain_matrix = np.array([[0.65, 0.70, 0.29],  # Hematoxylin
                             [0.07, 0.99, 0.11],  # Eosin
                             [0.27, 0.57, 0.78]])  # Background
    stain_deconv = htk.color_deconvolution(image, stain_matrix)
    return stain_deconv.Stains[:, :, 0], stain_deconv.Stains[:, :, 1]


def compute_cv(stain_channel):
    # calculate Coefficient of Variation (CV) for staining intensity
    mean_intensity = np.mean(stain_channel)
    std_intensity = np.std(stain_channel)
    cv = (std_intensity / mean_intensity) * 100
    return cv


def visualize_stains(image_id, image, hematoxylin, eosin):
    # visualize hematoxylin and eosin stain components
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Image {image_id}")

    plt.subplot(1, 3, 2)
    plt.imshow(eosin, cmap="gray")
    plt.title("Eosin Channel")

    plt.subplot(1, 3, 3)
    plt.imshow(hematoxylin, cmap="gray")
    plt.title("Hematoxylin Channel")

    plt.show()

# all images
# df = pd.read_csv(RAW_DATA_ROOT + '\\tile_meta.csv')
# print(df.head(5))
# image_ids = df['id'].values.tolist()

# images with h_cv > 2 and e_cv < 40
# image_ids = ["0006ff2aa7cd",
# "0f5b52a768e2",
# "24bc5d5889b4",
# "5ab182e75656",
# "7e695638b845",
# "a63fa35aa326",
# "b137de0d8f5c",
# "b78c3f072465",
# "c956a02b5d22",
# "d71ee2e0b7b5",
# "dd83f2d82305",
# "dfe0f68f1f43",
# "e3bd701c768d",
# ]

# images with h_cv < 15 and e_cv < 15
#image_ids = ['22429e2b9c48','46d9ee4f8a44','4aa434181421','55cab2c9c13a','596a08cbd66b','c1da0ec94d0e','d6fd6d5bf8ee','eff3a8b0bf65']

# selected target image
image_ids = ['b78c3f072465']
for image_id in image_ids:
# --- Main Execution ---
    image_path = RAW_DATA_ROOT + "\\train\\" + f"{image_id}.tif"
    image = load_image(image_path)


    # print(f"{image_id}")
    # Step 1: Plot RGB Histogram
    plot_histogram(image)

    # Step 2: Compute Lab Color Statistics
    mean_lab, std_lab = compute_lab_stats(image)
    print(f"Lab Means: {mean_lab}")
    print(f"Lab Std Dev: {std_lab}")

    # Step 3: Perform Stain Deconvolution
    hematoxylin_channel, eosin_channel = stain_deconvolution(image)

    # Step 4: Compute Staining Coefficient of Variation (CV)
    h_cv = compute_cv(hematoxylin_channel)
    e_cv = compute_cv(eosin_channel)
    print(image_id + f" {h_cv} {e_cv}")
    print(f"Coefficient of Variation (Hematoxylin): {h_cv:.2f}%")
    print(f"Coefficient of Variation (Eosin): {e_cv:.2f}%")

    # Step 5: Visualize Separated Stains
    visualize_stains(image_id, image,hematoxylin_channel, eosin_channel)

    # Step 6: Check Staining Consistency
    if h_cv < 35 and e_cv < 35:
        print(image_id + f" {h_cv} {e_cv}"  " ✅ Staining is uniform.")
    else:
        print("⚠ Staining is inconsistent. Consider normalization.")
