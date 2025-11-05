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


#test get_API

import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.APIMaps.GetAPI import getAPI


# --- Paths ---
img_path = r"Z:\Users\Artin\coiled\raw_file\ANY_103_0"
mask_path = r"Z:\Users\Artin\coiled\aneurysms\ANY_103_0.tif"
inlet_path = r"Z:\Users\Artin\coiled\inlets\ANY_103_0_inl.tif"

# --- Load DICOM ---
ds = pydicom.dcmread(img_path)
arr = ds.pixel_array.astype(np.float32)

# Make sure arr is 3D: (frames, height, width)
if arr.ndim == 2:
    arr = arr[None, :, :]  # add a frames dimension

# --- Load mask and inlet ---
mask_img = Image.open(mask_path).convert("L")  # grayscale
mask = np.array(mask_img) > 0  # binary mask

inlet_img = Image.open(inlet_path).convert("L")
inlet = np.array(inlet_img) > 0  # boolean inlet

# --- Threshold fraction ---
frac = 0.1

# --- Run getAPI ---
api_obj = getAPI(dsa=arr, mask=mask, inlet=inlet, frac=frac)

# --- Access results ---
print("Mean parameters:", api_obj.results_mean)
print("Std parameters:", api_obj.results_std)
print("Number of qualifying pixels:", api_obj.qualifying_pixels)
print("Qualifying indices:", api_obj.qualifying_indices)

# --- Plot average TDC ---
plt.plot(api_obj._x_inter, api_obj.tdc_average)
plt.xlabel("Time (s)")
plt.ylabel("Average Concentration")
plt.title("Average TDC")
plt.show()

import numpy as np

# Example data: rows = time steps, columns = connected components
# Each value is the contrast at that time for that component
contrast_data = np.array([
    [0.1, 0.5, 0.8],
    [0.2, 0.5, 0.7],
    [0.3, 0.6, 0.9],
    [0.1, 0.7, 0.85]
])

# Step 1: Discretize contrast into bins (0-1 range, 3 bins: low, medium, high)
bins = [0.0, 0.33, 0.66, 1.0]  # 3 bins
discrete_data = np.digitize(contrast_data, bins) - 1  # subtract 1 to make 0-indexed

# Step 2: Compute transition matrix for each component
num_states = len(bins) - 1
num_components = contrast_data.shape[1]

transition_matrices = []

for comp in range(num_components):
    # Initialize count matrix
    counts = np.zeros((num_states, num_states))

    # Count transitions
    for t in range(discrete_data.shape[0] - 1):
        current_state = discrete_data[t, comp]
        next_state = discrete_data[t + 1, comp]
        counts[current_state, next_state] += 1

    # Normalize to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

    transition_matrices.append(probs)

# Print transition matrices
for i, matrix in enumerate(transition_matrices):
    print(f"Transition matrix for component {i}:")
    print(matrix)
    print()

