import numpy as np
import pydicom
import tifffile
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion

from src.Disturbed.VesselSegment import VesselSegment
from utils.disturbedhelper import global_thresholding

dcm = pydicom.dcmread(r"Z:\Users\Artin\coiled\raw_file\ANY_103_1")
# Convert pixel data to NumPy array
dsa = dcm.pixel_array  # shape: (frames, height, width) or (height, width) if single frame

if dsa.ndim == 2:
    dsa = np.expand_dims(dsa, axis=0)

mask = tifffile.imread(r"Z:\Users\Artin\coiled\aneurysms\ANY_103_1.tif").astype(bool)

# ---- Initialize VesselSegment ----
frame_num= 9 # choose frame to process
    # padding around mask for focused region



vessel = VesselSegment(dsa, mask, frame_num)

# ---- Get focused mask and valid coordinates ----
f_mask = vessel.f_mask
valid_y, valid_x = vessel.grow_coordinates

# ---- Create boolean map of valid coordinates ----
valid_map = np.zeros_like(mask, dtype=bool)
# valid_map[valid_y, valid_x] = True

# ---- Visualization ----
plt.figure(figsize=(8, 8))
plt.imshow(dsa[frame_num], cmap='gray')
plt.imshow(f_mask, cmap='Blues', alpha=0.2, label='Focused Mask')
plt.scatter(valid_x, valid_y, s=10, c='red', label='Valid Coordinates')
plt.title("Focused Mask and Valid Coordinates on DICOM Frame")
plt.axis('off')
plt.show()
# ---- Visualization of segmentation ----
plt.figure(figsize=(15, 5))

# Original zoomed region (before thresholding)
plt.subplot(1, 3, 1)
plt.imshow(dsa[frame_num], cmap='gray')
plt.imshow(f_mask, cmap='Blues', alpha=0.2)
plt.title("Zoomed DSA Region (Before Threshold)")
plt.axis('off')

# Segmented vessel mask
plt.subplot(1, 3, 2)
plt.imshow(dsa[frame_num], cmap='gray')
plt.imshow(vessel.segmented, cmap='Reds', alpha=0.5)
plt.title("Segmented Vessel (After Threshold)")
plt.axis('off')

# Valid coordinates over segmentation
plt.subplot(1, 3, 3)
plt.imshow(dsa[frame_num], cmap='gray')
plt.imshow(vessel.segmented, cmap='Reds', alpha=0.5)
plt.scatter(valid_x, valid_y, s=10, c='yellow', label='Valid Coordinates')
plt.title("Valid Coordinates Overlay")
plt.axis('off')
plt.legend()

plt.tight_layout()
plt.show()

# ---- Extract padded region coordinates ----
y_coord, x_coord = vessel.coords
y_min, y_max = y_coord.min(), y_coord.max()
x_min, x_max = x_coord.min(), x_coord.max()

# Apply padding
padding = 50
y_min_pad = max(y_min - padding, 0)
y_max_pad = min(y_max + padding, vessel.dsa.shape[1] - 1)
x_min_pad = max(x_min - padding, 0)
x_max_pad = min(x_max + padding, vessel.dsa.shape[2] - 1)

# Crop DSA, f_mask, and segmentation to padded region
dsa_crop = vessel.dsa[frame_num, y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
f_mask_crop = vessel.f_mask[y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
seg_crop = vessel.segmented[y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]

# ---- Visualization ----
plt.figure(figsize=(12, 5))

# Padded zoom region (DSA + f_mask overlay)
plt.subplot(1, 2, 1)
plt.imshow(dsa_crop, cmap='gray')
plt.imshow(f_mask_crop, cmap='Blues', alpha=0.3)
plt.title("Zoomed Padded Region (f_mask)")
plt.axis('off')

# Segmented region in zoomed patch
plt.subplot(1, 2, 2)
plt.imshow(dsa_crop, cmap='gray')
plt.imshow(seg_crop, cmap='Reds', alpha=0.5)
plt.title("Segmented Vessel (Zoomed Region)")
plt.axis('off')

plt.tight_layout()
plt.show()

# ---- Find missing coordinates ----
y_missing, x_missing = vessel.find_missing_coords(frame_num, difference= 90)

# ---- Crop to padded region if you want zoomed view ----
y_coord, x_coord = vessel.coords
y_min, y_max = y_coord.min(), y_coord.max()
x_min, x_max = x_coord.min(), x_coord.max()
padding = 40
y_min_pad = max(y_min - padding, 0)
y_max_pad = min(y_max + padding, vessel.dsa.shape[1] - 1)
x_min_pad = max(x_min - padding, 0)
x_max_pad = min(x_max + padding, vessel.dsa.shape[2] - 1)

dsa_crop = vessel.dsa[frame_num, y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]

# Adjust missing coordinates relative to crop
y_missing_crop = y_missing - y_min_pad
x_missing_crop = x_missing - x_min_pad

# ---- Visualization ----
plt.figure(figsize=(6,6))
plt.imshow(dsa_crop, cmap='gray')
plt.scatter(x_missing_crop, y_missing_crop, s=10, c='red', label='Missing Mask Pixels')
plt.title("Missing Mask Pixels in Zoomed Padded Region")
plt.axis('off')
plt.legend()
plt.show()

# ---- Find missing coordinates ----
y_missing, x_missing = vessel.find_missing_coords(frame_num, difference=20)

# ---- Compute padded region (again, consistent with earlier) ----
y_coord, x_coord = vessel.coords
y_min, y_max = y_coord.min(), y_coord.max()
x_min, x_max = x_coord.min(), x_coord.max()
padding = 10
y_min_pad = max(y_min - padding, 0)
y_max_pad = min(y_max + padding, vessel.dsa.shape[1] - 1)
x_min_pad = max(x_min - padding, 0)
x_max_pad = min(x_max + padding, vessel.dsa.shape[2] - 1)

# ---- Crop DSA frame ----
dsa_crop = vessel.dsa[frame_num, y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]

# ---- Adjust coordinates relative to cropped region ----
y_valid_crop = valid_y - y_min_pad
x_valid_crop = valid_x - x_min_pad
y_missing_crop = y_missing - y_min_pad
x_missing_crop = x_missing - x_min_pad

# ---- Visualization: valid + missing ----
plt.figure(figsize=(7, 7))
plt.imshow(dsa_crop, cmap='gray')
plt.scatter(x_valid_crop, y_valid_crop, s=10, c='yellow', label='Valid Coordinates', alpha=0.7)
plt.scatter(x_missing_crop, y_missing_crop, s=10, c='red', label='Missing Mask Pixels', alpha=0.7)
plt.title("Valid and Missing Coordinates (Zoomed Padded Region)")
plt.axis('off')
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()


