import numpy as np
import pydicom
from cv2 import imread
from matplotlib import pyplot as plt

from src.Disturbed.GrowingRegion import growingRegion
from utils.disturbedhelper import otsu_algo

#otsu algorithm test
img_path = r"Z:\Users\Artin\coiled\raw_file\ANY_103_1"
ds = pydicom.dcmread(img_path)
arr = ds.pixel_array.astype(np.float32)

# Select one slice if 3D
i = arr[9, :, :] if arr.ndim == 3 else arr

# Apply Otsu thresholding
seg, threshold = otsu_algo(i)

# Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(i, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(seg, cmap='gray')
plt.title(f'Otsu Segmentation (Threshold={threshold})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.hist(i.ravel(), bins=256, color='gray')
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2)
plt.title('Histogram with Otsu Threshold')

plt.tight_layout()
plt.show()


#growingregion test
img_path = r"Z:\Users\Artin\coiled\raw_file\ANY_103_1"
ds = pydicom.dcmread(img_path)
arr = ds.pixel_array.astype(np.float32)

# Rescale if necessary
if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
    arr = arr * ds.RescaleSlope + ds.RescaleIntercept

# Pick a single frame if 3D
i = arr[9, :, :] if arr.ndim == 3 else arr

# Initialize and grow regions
a = growingRegion(i, 2)
seg = a.growing(threshold=None, show_debug=True)  # threshold=None uses automatic ~10% of intensity range

# Display original vs segmented
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(i, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Segmented")
plt.imshow(seg, cmap='nipy_spectral')
plt.axis('off')
plt.show()


# crop test
# --- File paths ---
img_path = r"Z:\Users\Artin\coiled\raw_file\ANY_106_1"
mask_path = r"Z:\Users\Artin\coiled\aneurysms\ANY_106_1.tif"

# --- Load DICOM image ---
ds = pydicom.dcmread(img_path)
arr = ds.pixel_array.astype(np.float32)  # shape (time, height, width)
#crop test
# --- Load mask (TIF image) ---
mask = imread(mask_path).astype(bool)  # shape (height, width)

# --- Check shapes ---
if arr.ndim != 3:
    raise ValueError(f"Expected 3D DICOM image (time, height, width), got {arr.shape}")
if mask.shape != arr.shape[1:]:
    raise ValueError(f"Mask shape {mask.shape} must match image spatial dims {arr.shape[1:]}")

# --- Apply mask (set masked region to white across all frames) ---
arr_masked = arr.copy()
arr_masked[:, mask] = 255.0

# --- Clip and cast for display ---
arr_masked = np.clip(arr_masked, 0, 255).astype(np.uint8)

# --- Visualize each frame ---
num_frames = arr_masked.shape[0]
for i in range(num_frames):
    plt.imshow(arr_masked[i], cmap='gray')
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.show()
