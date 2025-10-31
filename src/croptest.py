
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread  # or use from PIL import Image if you prefer

# --- File paths ---
img_path = r"Z:\Users\Artin\coiled\raw_file\ANY_106_1"
mask_path = r"Z:\Users\Artin\coiled\aneurysms\ANY_106_1.tif"

# --- Load DICOM image ---
ds = pydicom.dcmread(img_path)
arr = ds.pixel_array.astype(np.float32)  # shape (time, height, width)

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
