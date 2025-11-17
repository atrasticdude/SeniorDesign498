import numpy as np
import pandas as pd
import pydicom
import tifffile
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
from pathlib import Path

from src.Disturbed.VesselSegment import VesselSegment

# === PATH CONFIG ===
raw_path = Path(r"Z:\Users\Artin\coiled\raw_file")
mask_path = Path(r"Z:\Users\Artin\coiled\aneurysms")

csv_path = Path(__file__).resolve().parent.parent / "Data" / "Inlet_Treat_Parameter_wphframe.csv"
df = pd.read_csv(csv_path)
df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

# === PARAMETERS ===
# frame_num = 8
padding = 10
difference = 30

# === CASE LOOP ===
for i in range(300,400):
    case_name = f"ANY_{i}_View2_1"
    case = f"ANY_{i}_View2"
    print(f"\n--- Processing {case_name} ---")

    dcm_file = raw_path / case_name
    mask_file = mask_path / f"{case_name}.tif"

    if not dcm_file.exists() or not mask_file.exists():
        print(f"⚠️ Missing files for {case_name}, skipping.")
        continue

    # ---- Load data ----
    dcm = pydicom.dcmread(dcm_file)
    dsa = dcm.pixel_array
    if dsa.ndim == 2:
        dsa = np.expand_dims(dsa, axis=0)
    mask = tifffile.imread(mask_file).astype(bool)


    row = df.loc[df["ID"] == case, "PH_Frame"]

    if not row.empty and not np.isnan(row.values[0]):
        ph_frame = int(row.values[0])
    else:
        print(f"No PH_Frame found for case {case}")
        continue

    if ph_frame >= dsa.shape[0]:
        continue
    elif ph_frame / dsa.shape[0] > 0.5:
        continue

    # ---- Initialize VesselSegment ----
    vessel = VesselSegment(dsa, mask, ph_frame)

    # ---- Get valid and missing coordinates ----
    valid_y, valid_x = vessel.grow_coordinates
    y_missing, x_missing = vessel.find_missing_coords(ph_frame, difference=difference)

    # ---- Compute full centerline ----
    centerline, _ = medial_axis(vessel.segmented, return_distance=True)

    # ---- Create missing mask and restrict centerline ----
    missing_map = np.zeros_like(centerline, dtype=bool)
    missing_map[y_missing, x_missing] = True
    centerline_in_missing = centerline & missing_map

    # ---- Crop region ----
    y_coord, x_coord = vessel.coords
    y_min, y_max = y_coord.min(), y_coord.max()
    x_min, x_max = x_coord.min(), x_coord.max()

    y_min_pad = max(y_min - padding, 0)
    y_max_pad = min(y_max + padding, vessel.dsa.shape[1] - 1)
    x_min_pad = max(x_min - padding, 0)
    x_max_pad = min(x_max + padding, vessel.dsa.shape[2] - 1)

    dsa_crop = vessel.dsa[ph_frame, y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
    centerline_crop = centerline_in_missing[y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
    cy_crop, cx_crop = np.where(centerline_crop)

    # ---- Adjust coordinates relative to crop ----
    y_valid_crop = valid_y - y_min_pad
    x_valid_crop = valid_x - x_min_pad
    y_missing_crop = y_missing - y_min_pad
    x_missing_crop = x_missing - x_min_pad

    # ---- Visualization ----
    # plt.figure(figsize=(7, 7))
    # plt.imshow(dsa_crop, cmap='gray')
    # plt.scatter(x_valid_crop, y_valid_crop, s=10, c='yellow', alpha=0.7)
    # plt.scatter(x_missing_crop, y_missing_crop, s=10, c='red', alpha=0.7)
    # # plt.scatter(cx_crop, cy_crop, s=8, c='cyan', alpha=0.9)
    # # plt.title(f"{case_name}: Centerline on Missing Coordinates")
    # plt.axis('off')
    # # plt.legend(loc='lower right', frameon=True)
    # # plt.tight_layout()
    # plt.figure(figsize=(6, 6), dpi=200)  # <--- FIXED output size
    # # plt.show()
    pfig = plt.figure(figsize=(2, 2), dpi=64)  # 128x128 px
    ax = plt.axes([0, 0, 1, 1])               # fill entire canvas

    ax.imshow(dsa_crop, cmap='gray')
    ax.scatter(x_valid_crop, y_valid_crop, s=10, c='yellow', alpha=0.7)
    ax.scatter(x_missing_crop, y_missing_crop, s=10, c='red', alpha=0.7)

    ax.set_axis_off()

    save_dir = Path(r"Z:\Users\Artin\coiled\cnn_pics")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / f"{case_name}.png", dpi=200)
    plt.close()
    print(f"✅ Saved {case_name}_inside_intensity.png")

print("\n🎯 All cases processed successfully!")