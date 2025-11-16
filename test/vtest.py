import numpy as np
import matplotlib.pyplot as plt
import cv2
import pydicom
import tifffile
from src.Disturbed.VesselSegment import VesselSegment

# --- Load DICOM ---
dcm = pydicom.dcmread(r"Z:\Users\Artin\coiled\raw_file\ANY_103_1")
dsa = dcm.pixel_array

if dsa.ndim == 2:
    dsa = np.expand_dims(dsa, axis=0)

# --- Load mask ---
mask = tifffile.imread(r"Z:\Users\Artin\coiled\aneurysms\ANY_103_1.tif").astype(bool)

# --- Select frame ---
frame_num = 9
vessel = VesselSegment(dsa, mask, frame_num)
dsa_frame = dsa[frame_num, :, :]

# Optional: set background to max+1
dsa_frame[~mask] = dsa_frame.max() + 1

# --- Compute centroids per threshold ---
pixel_centroids = vessel.find_true_diff(dsa_frame)  # list of (threshold, [centroids])

# --- Bounding box of the mask to zoom ---
ys, xs = np.where(mask)
min_y, max_y = ys.min(), ys.max()
min_x, max_x = xs.min(), xs.max()
pad = 5
min_y = max(min_y - pad, 0)
min_x = max(min_x - pad, 0)
max_y = min(max_y + pad, dsa_frame.shape[0] - 1)
max_x = min(max_x + pad, dsa_frame.shape[1] - 1)

# --- Show centroids for each threshold separately ---
for threshold, centroids in pixel_centroids:
    # Convert frame to color for plotting
    dsa_color = cv2.cvtColor(dsa_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Draw centroids for this threshold
    for cx, cy in centroids:
        if min_x <= cx <= max_x and min_y <= cy <= max_y:  # only draw inside zoomed region
            cv2.circle(dsa_color, (int(cx), int(cy)), radius=2, color=(0, 0, 255), thickness=-1)

    # Crop to mask region
    dsa_crop = dsa_color[min_y:max_y + 1, min_x:max_x + 1]

    # Show
    plt.figure(figsize=(6, 6))
    plt.imshow(dsa_crop)
    plt.title(f"Centroids for Threshold {threshold}")
    plt.axis('off')
    plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import pydicom
# import tifffile
# from src.Disturbed.VesselSegment import VesselSegment
# from utils.disturbedhelper import otsu_algo
#
# # --- Load DICOM ---
# dcm = pydicom.dcmread(r"Z:\Users\Artin\coiled\raw_file\ANY_103_1")
# dsa = dcm.pixel_array
#
# if dsa.ndim == 2:
#     dsa = np.expand_dims(dsa, axis=0)
#
# # --- Load mask ---
# mask = tifffile.imread(r"Z:\Users\Artin\coiled\aneurysms\ANY_103_1.tif").astype(bool)
#
# # --- Select frame ---
# frame_num = 9
# vessel = VesselSegment(dsa, mask, frame_num)
# dsa_frame = dsa[frame_num, :, :]
# dsa_frame[~mask] = dsa_frame.max() + 1
#
# import matplotlib.pyplot as plt
#
# # --- Run Otsu ---
# import cv2
# frame_uint8 = cv2.normalize(dsa_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# seg, thresh, max_score, score = otsu_algo(frame_uint8)
# dsa_frame = frame_uint8 <= 100


# --- Visualize the segmentation ---
# plt.figure(figsize=(6,6))
# plt.imshow(dsa_frame, cmap='gray')
# plt.title(f"Otsu Segmentation (Threshold = {thresh})")
# plt.axis('off')
# plt.show()

