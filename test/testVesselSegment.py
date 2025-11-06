import numpy as np
import pydicom
import tifffile
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion

from src.Disturbed.VesselSegment import VesselSegment
from utils.disturbedhelper import global_thresholding

# ---- VesselSegment class here ----
# Make sure your fixed VesselSegment class is in the same script or imported

# ---- Load DICOM ----
dcm = pydicom.dcmread(r"Z:\Users\Artin\coiled\raw_file\ANY_340_1")
# Convert pixel data to NumPy array
dsa = dcm.pixel_array  # shape: (frames, height, width) or (height, width) if single frame

# If single frame, expand dims to make consistent
if dsa.ndim == 2:
    dsa = np.expand_dims(dsa, axis=0)

# ---- Load mask TIFF ----
mask = tifffile.imread(r"Z:\Users\Artin\coiled\aneurysms\ANY_340_1.tif").astype(bool)

# ---- Initialize VesselSegment ----
frame_num = 7   # choose frame to process
padding = 30     # padding around mask for focused region

import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion

# import matplotlib.pyplot as plt
# from skimage.morphology import binary_erosion
#
#
# def visualize_boundary_overlay(vessel):
#     # Region inside the zoom area
#     region = vessel.segmented & vessel.f_mask
#
#     # Eroded region
#     eroded_region = binary_erosion(region)
#
#     # Boundary of region
#     region_boundary = region & (~eroded_region)
#
#     # Original mask boundary
#     mask_eroded = binary_erosion(vessel.mask)
#     mask_boundary = vessel.mask & (~mask_eroded)
#
#     # Overlay: start with region
#     plt.figure(figsize=(6, 6))
#     plt.imshow(region, cmap='gray', alpha=0.8)
#
#     # Region boundary in red
#     y_r, x_r = np.where(region_boundary)
#     plt.scatter(x_r, y_r, color='red', s=1, label='Region Boundary')
#
#     # Mask boundary in blue
#     y_m, x_m = np.where(mask_boundary)
#     plt.scatter(x_m, y_m, color='blue', s=1, label='Mask Boundary')
#
#     plt.title("Mask Boundary projected on Region")
#     plt.axis('off')
#     plt.legend(loc='upper right')
#     plt.show()
#
#
# vessel = VesselSegment(dsa, mask, frame_num)
# visualize_boundary_overlay(vessel)

vessel = VesselSegment(dsa, mask, frame_num, padding)

# ---- Get focused mask and valid coordinates ----
f_mask = vessel.f_mask
valid_y, valid_x = vessel.grow_coordinates

# ---- Create boolean map of valid coordinates ----
valid_map = np.zeros_like(mask, dtype=bool)
valid_map[valid_y, valid_x] = True

# ---- Visualization ----
plt.figure(figsize=(8, 8))
plt.imshow(dsa[frame_num], cmap='gray')
plt.imshow(f_mask, cmap='Blues', alpha=0.2, label='Focused Mask')
plt.scatter(valid_x, valid_y, s=10, c='red', label='Valid Coordinates')
plt.title("Focused Mask and Valid Coordinates on DICOM Frame")
plt.axis('off')
plt.show()

#
#
# import numpy as np
# import pydicom
# import tifffile
# import matplotlib.pyplot as plt
# from skimage.morphology import binary_erosion
# from utils.disturbedhelper import global_thresholding
#
# # ---- VesselSegment class here ----
# # Make sure your fixed VesselSegment class is in the same script or imported
#
# # ---- Load DICOM ----
# dcm = pydicom.dcmread(r"Z:\Users\Artin\coiled\raw_file\ANY_103_1")
# # Convert pixel data to NumPy array
# dsa = dcm.pixel_array  # shape: (frames, height, width) or (height, width) if single frame
#
# # If single frame, expand dims to make consistent
# if dsa.ndim == 2:
#     dsa = np.expand_dims(dsa, axis=0)
#
# # ---- Load mask TIFF ----
# mask = tifffile.imread(r"Z:\Users\Artin\coiled\aneurysms\ANY_103_1.tif").astype(bool)
#
# # ---- Initialize VesselSegment ----
# frame_num = 9   # choose frame to process
# padding = 30     # padding around mask for focused region
# vessel = VesselSegment(dsa, mask, frame_num, padding)
#
# # ---- Get focused mask and valid coordinates ----
#
# import matplotlib.pyplot as plt
# from skimage.morphology import binary_erosion
#
# # ---- Masks ----
# seg_mask = vessel.segmented        # segmented mask
# f_mask = vessel.f_mask             # focused mask with padding
#
# # Erode the focused mask to see the boundary
# eroded_mask = binary_erosion(f_mask)
# boundary_mask = f_mask & (~eroded_mask)
#
# # ---- Plot ----
# plt.figure(figsize=(8, 8))
# plt.imshow(dsa[frame_num], cmap='gray')                # original DSA frame
#
# # Overlay masks
# plt.imshow(seg_mask, cmap='Reds', alpha=0.3, label='Segmented Mask')
# plt.imshow(f_mask, cmap='Blues', alpha=0.2, label='Focused Mask')
# plt.imshow(boundary_mask, cmap='Greens', alpha=0.5, label='Boundary after Erosion')
#
# plt.title("Segmented Mask + Focused Mask + Boundary")
# plt.axis('off')
# plt.show()
#
# import matplotlib.pyplot as plt
# from skimage.morphology import binary_erosion
# from utils.disturbedhelper import global_thresholding
#
# frame = dsa[frame_num, :, :]  # DSA frame
#
# # --- Step 1: Original DSA frame ---
# plt.figure(figsize=(16, 4))
# plt.subplot(1, 4, 1)
# plt.imshow(frame, cmap='gray')
# plt.title("Original DSA Frame")
# plt.axis('off')
#
# # --- Step 2: After Global Thresholding ---
# _, seg = global_thresholding(frame)
# plt.subplot(1, 4, 2)
# plt.imshow(frame, cmap='gray')
# plt.imshow(seg, cmap='Reds', alpha=0.3)
# plt.title("After Thresholding")
# plt.axis('off')
#
# # --- Step 3: After Removing Aneurysm Mask ---
# seg_masked = seg.copy().astype(bool)
# seg_masked[vessel.mask] = 0
# plt.subplot(1, 4, 3)
# plt.imshow(frame, cmap='gray')
# plt.imshow(seg_masked, cmap='Reds', alpha=0.3)
# plt.imshow(vessel.mask, cmap='Greens', alpha=0.3)
# plt.title("After Removing Mask Region")
# plt.axis('off')
#
# # --- Step 4: After Adding Padding (Focused Mask) ---
# plt.subplot(1, 4, 4)
# plt.imshow(frame, cmap='gray')
# plt.imshow(seg_masked, cmap='Reds', alpha=0.3)
# plt.imshow(vessel.f_mask, cmap='Blues', alpha=0.2)
# plt.title("After Adding Padded Region")
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()
#