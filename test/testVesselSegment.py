import numpy as np
import pydicom
import tifffile
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion

from src.Disturbed.VesselSegment import VesselSegment
from utils.disturbedhelper import global_thresholding
from utils.helperfunction import cubicinter

dcm = pydicom.dcmread(r"Z:\Users\Artin\coiled\raw_file\ANY_134_1")
# Convert pixel data to NumPy array
dsa = dcm.pixel_array  # shape: (frames, height, width) or (height, width) if single frame

if dsa.ndim == 2:
    dsa = np.expand_dims(dsa, axis=0)

mask = tifffile.imread(r"Z:\Users\Artin\coiled\aneurysms\ANY_134_1.tif").astype(bool)


frame_num= 14

vessel = VesselSegment(dsa, mask, frame_num)
clean_dsa , x_noisy, y_noisy, y_clean, x_clean = vessel.detect_noisy_pixels(cubicinter=None,  deriv_percentile=90, std_percentile=90, smoothing=True)

f_mask = vessel.f_mask
valid_y, valid_x = vessel.grow_coordinates
# #########################################################################################################################################################
# ---- Create boolean map of valid coordinates ----
valid_map = np.zeros_like(mask, dtype=bool)
# clean_dsa, noisy_dsa, y_clean, x_clean, y_noisy, x_noisy
frame_to_show = 14  # choose the frame to visualize

# Original DSA frame
dsa_frame = dsa[frame_to_show]

plt.figure(figsize=(8, 8))
plt.imshow(dsa_frame, cmap='gray')

# Overlay clean pixels in blue
plt.scatter(x_clean, y_clean, color='blue', s=5, label='Clean Pixels')

# Optionally overlay noisy pixels in red for comparison
plt.scatter(x_noisy, y_noisy, color='red', s=5, label='Noisy Pixels', alpha=0.5)

plt.title(f"Frame {frame_to_show}: Clean vs Noisy Pixels")
plt.axis('off')
plt.legend()
plt.show()

# valid_map[valid_y, valid_x] = True


#######################################################################################################################################################
# Valid coordinates over segmentation
# ########################################################################################################################################################
# # # ---- Extract padded region coordinates ----
# # y_coord, x_coord = vessel.coords
# # y_min, y_max = y_coord.min(), y_coord.max()
# # x_min, x_max = x_coord.min(), x_coord.max()
# #
# # # Apply padding
# # padding = 50
# # y_min_pad = max(y_min - padding, 0)
# # y_max_pad = min(y_max + padding, vessel.dsa.shape[1] - 1)
# # x_min_pad = max(x_min - padding, 0)
# # x_max_pad = min(x_max + padding, vessel.dsa.shape[2] - 1)
# #
# # # Crop DSA, f_mask, and segmentation to padded region
# # dsa_crop = vessel.dsa[frame_num, y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
# # f_mask_crop = vessel.f_mask[y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
# # seg_crop = vessel.segmented[y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
# #
# # # ---- Visualization ----
# # plt.figure(figsize=(12, 5))
# #
# # # Padded zoom region (DSA + f_mask overlay)
# # plt.subplot(1, 2, 1)
# # plt.imshow(dsa_crop, cmap='gray')
# # plt.imshow(f_mask_crop, cmap='Blues', alpha=0.3)
# # plt.title("Zoomed Padded Region (f_mask)")
# # plt.axis('off')
# #
# # # Segmented region in zoomed patch
# # plt.subplot(1, 2, 2)
# # plt.imshow(dsa_crop, cmap='gray')
# # plt.imshow(seg_crop, cmap='Reds', alpha=0.5)
# # plt.title("Segmented Vessel (Zoomed Region)")
# # plt.axis('off')
# #
# # plt.tight_layout()
# # plt.show()
# #
# # # ---- Find missing coordinates ----
# y_missing, x_missing = vessel.find_missing_coords(frame_num, difference= 90)
#
# # ---- Crop to padded region if you want zoomed view ----
# y_coord, x_coord = vessel.coords
# y_min, y_max = y_coord.min(), y_coord.max()
# x_min, x_max = x_coord.min(), x_coord.max()
# padding = 40
# y_min_pad = max(y_min - padding, 0)
# y_max_pad = min(y_max + padding, vessel.dsa.shape[1] - 1)
# x_min_pad = max(x_min - padding, 0)
# x_max_pad = min(x_max + padding, vessel.dsa.shape[2] - 1)
#
# dsa_crop = vessel.dsa[frame_num, y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
#
# # Adjust missing coordinates relative to crop
# y_missing_crop = y_missing - y_min_pad
# x_missing_crop = x_missing - x_min_pad
#
# # ---- Visualization ----
# plt.figure(figsize=(6,6))
# plt.imshow(dsa_crop, cmap='gray')
# plt.scatter(x_missing_crop, y_missing_crop, s=10, c='red', label='Missing Mask Pixels')
# plt.title("Missing Mask Pixels in Zoomed Padded Region")
# plt.axis('off')
# plt.legend()
# plt.show()

# ---- Find missing coordinates ----
y_missing, x_missing = vessel.find_missing_coords(frame_num, difference=30)

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

# from skimage.morphology import medial_axis
# import matplotlib.pyplot as plt
# import numpy as np
#
# # ---- 1. Compute full centerline from segmented region ----
# centerline, distance = medial_axis(vessel.segmented, return_distance=True)
#
# # ---- 2. Get coordinates of centerline ----
# cy, cx = np.where(centerline)
#
# # ---- 3. Convert missing coordinates into a boolean map ----
# missing_map = np.zeros_like(centerline, dtype=bool)
# missing_map[y_missing, x_missing] = True
#
# # ---- 4. Restrict centerline to missing-mask areas ----
# centerline_in_missing = centerline & missing_map
#
# # ---- 5. Crop everything to padded region ----
# centerline_crop = centerline_in_missing[y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
# cy_crop, cx_crop = np.where(centerline_crop)
#
# # ---- 6. Adjust other coordinates relative to crop ----
# y_valid_crop = valid_y - y_min_pad
# x_valid_crop = valid_x - x_min_pad
# y_missing_crop = y_missing - y_min_pad
# x_missing_crop = x_missing - x_min_pad
#
# # ---- 7. Visualization ----
# plt.figure(figsize=(7, 7))
# plt.imshow(dsa_crop, cmap='gray')
#
# # Overlays
# plt.scatter(x_valid_crop, y_valid_crop, s=10, c='yellow', label='Valid Coordinates', alpha=0.7)
# plt.scatter(x_missing_crop, y_missing_crop, s=10, c='red', label='Missing Mask Pixels', alpha=0.7)
# plt.scatter(cx_crop, cy_crop, s=8, c='cyan', label='Centerline (inside missing)', alpha=0.9)
#
# plt.title("Centerline on Missing Coordinates (Zoomed Region)")
# plt.axis('off')
# plt.legend(loc='lower right', frameon=True)
# plt.tight_layout()
# plt.show()


#######################################################################################################################################################
# import numpy as np
# import pandas as pd
# import pydicom
# import tifffile
# import matplotlib.pyplot as plt
# from skimage.morphology import medial_axis
# from pathlib import Path
#
# from src.Disturbed.VesselSegment import VesselSegment
#
# # === PATH CONFIG ===
# raw_path = Path(r"Z:\Users\Artin\coiled\raw_file")
# mask_path = Path(r"Z:\Users\Artin\coiled\aneurysms")
#
# csv_path = Path(__file__).resolve().parent.parent / "Data" / "Inlet_Treat_Parameter_wphframe.csv"
# df = pd.read_csv(csv_path)
# df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
#
# # === PARAMETERS ===
# # frame_num = 8
# padding = 10
# difference = 30
#
# # === CASE LOOP ===
# for i in range(120, 137):
#     case_name = f"ANY_{i}_1"
#     case = f"ANY_{i}"
#     print(f"\n--- Processing {case_name} ---")
#
#     dcm_file = raw_path / case_name
#     mask_file = mask_path / f"{case_name}.tif"
#
#     if not dcm_file.exists() or not mask_file.exists():
#         print(f"⚠️ Missing files for {case_name}, skipping.")
#         continue
#
#     # ---- Load data ----
#     dcm = pydicom.dcmread(dcm_file)
#     dsa = dcm.pixel_array
#     if dsa.ndim == 2:
#         dsa = np.expand_dims(dsa, axis=0)
#     mask = tifffile.imread(mask_file).astype(bool)
#
#
#     row = df.loc[df["ID"] == case, "PH_Frame"]
#
#     if not row.empty and not np.isnan(row.values[0]):
#         ph_frame = int(row.values[0])
#     else:
#         print(f"No PH_Frame found for case {case}")
#         continue
#
#     if ph_frame >= dsa.shape[0]:
#         continue
#     elif ph_frame / dsa.shape[0] > 0.5:
#         continue
#
#     # ---- Initialize VesselSegment ----
#     vessel = VesselSegment(dsa, mask, ph_frame)
#
#     # ---- Get valid and missing coordinates ----
#     valid_y, valid_x = vessel.grow_coordinates
#     y_missing, x_missing = vessel.find_missing_coords(ph_frame, difference=difference)
#
#     # ---- Compute full centerline ----
#     centerline, _ = medial_axis(vessel.segmented, return_distance=True)
#
#     # ---- Create missing mask and restrict centerline ----
#     missing_map = np.zeros_like(centerline, dtype=bool)
#     missing_map[y_missing, x_missing] = True
#     centerline_in_missing = centerline & missing_map
#
#     # ---- Crop region ----
#     y_coord, x_coord = vessel.coords
#     y_min, y_max = y_coord.min(), y_coord.max()
#     x_min, x_max = x_coord.min(), x_coord.max()
#
#     y_min_pad = max(y_min - padding, 0)
#     y_max_pad = min(y_max + padding, vessel.dsa.shape[1] - 1)
#     x_min_pad = max(x_min - padding, 0)
#     x_max_pad = min(x_max + padding, vessel.dsa.shape[2] - 1)
#
#     dsa_crop = vessel.dsa[ph_frame, y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
#     centerline_crop = centerline_in_missing[y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
#     cy_crop, cx_crop = np.where(centerline_crop)
#
#     # ---- Adjust coordinates relative to crop ----
#     y_valid_crop = valid_y - y_min_pad
#     x_valid_crop = valid_x - x_min_pad
#     y_missing_crop = y_missing - y_min_pad
#     x_missing_crop = x_missing - x_min_pad
#
#     # ---- Visualization ----
#     plt.figure(figsize=(7, 7))
#     plt.imshow(dsa_crop, cmap='gray')
#     plt.scatter(x_valid_crop, y_valid_crop, s=10, c='yellow', label='Valid Coordinates', alpha=0.7)
#     plt.scatter(x_missing_crop, y_missing_crop, s=10, c='red', label='Missing Mask Pixels', alpha=0.7)
#     plt.scatter(cx_crop, cy_crop, s=8, c='cyan', label='Centerline (inside missing)', alpha=0.9)
#     plt.title(f"{case_name}: Centerline on Missing Coordinates")
#     plt.axis('off')
#     plt.legend(loc='lower right', frameon=True)
#     plt.tight_layout()
#
#     # plt.show()
#     # ---- Save figure ----
#     save_dir = Path(r"Z:\Users\Artin\coiled\output_centerline")
#     save_dir.mkdir(exist_ok=True)
#     plt.savefig(save_dir / f"{case_name}_centerline_missing.png", dpi=200)
#     plt.close()
#
#     print(f"✅ Saved {case_name}_inside_intensity.png")
#
# print("\n🎯 All cases processed successfully!")
