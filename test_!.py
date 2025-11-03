import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.disturbedhelper import otsu_algo


def region_growing(image, seeds, threshold=25):
    """
    Perform region growing segmentation.

    Parameters:
        image (numpy.ndarray): Grayscale image
        seeds (list of tuples): List of seed points (x, y)
        threshold (int): Intensity difference threshold

    Returns:
        segmented (numpy.ndarray): Binary segmented image
    """
    height, width = image.shape
    segmented = np.zeros((height, width), np.uint8)
    visited = np.zeros((height, width), np.bool_)

    for seed in seeds:
        stack = [seed]
        while stack:
            x, y = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            segmented[y, x] = 255

            # Check 8-neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                        if abs(int(image[ny, nx]) - int(image[y, x])) <= threshold:
                            stack.append((nx, ny))
    return segmented


# Load the image
path = r"C:\Users\artdude\Pictures\Screenshots\mask_close.png"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Define seed points [(x1, y1), (x2, y2), ...]
seeds = [(50, 50)]  # Example seed

# Apply region growing
segmented_image,_ = otsu_algo(image)

# Show results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Region Growing Segmentation')
plt.imshow(segmented_image, cmap='gray')

plt.show()
