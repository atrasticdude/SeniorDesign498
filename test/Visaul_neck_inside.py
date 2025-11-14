import numpy as np
import matplotlib.pyplot as plt
import pydicom
from PIL import Image
from pathlib import Path
import pandas as pd
from src.Disturbed.VesselSegment import VesselSegment
from utils.helperfunction import sort_files_numerically
from src.Separation import Separate


def GetMasks(path):
    _, treat, _ = Separate.SepMasks(path)
    data = {}
    for key, file_list in treat.items():
        if len(file_list) > 1:
            file_list = sort_files_numerically(file_list)
        for idx, file_path in enumerate(file_list):
            try:
                with Image.open(file_path).convert("L") as img:
                    arr = np.array(img, dtype=np.float32) > 0
                    name = f"{key}"
                    if len(file_list) > 1:
                        name += f"_View{idx + 1}"
                    data[name] = arr
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
    return data


def GetImages(path):
    _, treat, _ = Separate.SepImages(path)
    data = {}
    path_dict = {}
    for key, file_list in treat.items():
        if len(file_list) > 1:
            file_list = sort_files_numerically(file_list)
        for idx, file_path in enumerate(file_list):
            try:
                ds = pydicom.dcmread(file_path)
                img_array = ds.pixel_array.astype(np.float32)
                name = f"{key}"
                if len(file_list) > 1:
                    name += f"_View{idx + 1}"
                data[name] = img_array
                path_dict[name] = ds
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
    return data, path_dict


if __name__ == "__main__":
    img_path = r"Z:\Users\Artin\coiled\raw_file"
    mask_path = r"Z:\Users\Artin\coiled\aneurysms"

    img_data, path_dict = GetImages(img_path)
    mask_data = GetMasks(mask_path)

    csv_path = Path(__file__).resolve().parent.parent / "Data" / "Inlet_Treat_Parameter_wphframe.csv"
    df = pd.read_csv(csv_path)
    df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

    cases = list(mask_data.keys())
    cases_per_figure = 9
    rows, cols = 3, 3

    for i in range(0, len(cases), cases_per_figure):
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()

        for j in range(cases_per_figure):
            if i + j >= len(cases):
                axes[j].axis('off')
                continue

            case = cases[i + j]
            mask = mask_data[case]

            if case not in img_data:
                axes[j].axis('off')
                continue

            dsa = img_data[case]
            dcm = path_dict[case]

            row = df.loc[df["ID"] == case, "PH_Frame"]

            if not row.empty and not np.isnan(row.values[0]):
                ph_frame = int(row.values[0])
            else:
                print(f"No PH_Frame found for case {case}")
                axes[j].axis('off')
                continue

            if ph_frame >= dsa.shape[0]:
                axes[j].axis('off')
                continue
            elif ph_frame / dsa.shape[0] > 0.5:
                continue
            vessel = VesselSegment(dsa, mask, frame_num=ph_frame, padding=20)
            valid_y, valid_x = vessel.grow_coordinates
            y_missing, x_missing = vessel.find_missing_coords(ph_frame, difference=90)

            y_coord, x_coord = vessel.coords
            y_min, y_max = y_coord.min(), y_coord.max()
            x_min, x_max = x_coord.min(), x_coord.max()
            pad = 40
            y_min_pad = max(y_min - pad, 0)
            y_max_pad = min(y_max + pad, vessel.dsa.shape[1] - 1)
            x_min_pad = max(x_min - pad, 0)
            x_max_pad = min(x_max + pad, vessel.dsa.shape[2] - 1)

            dsa_crop = vessel.dsa[ph_frame, y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
            y_valid_crop = valid_y - y_min_pad
            x_valid_crop = valid_x - x_min_pad
            y_missing_crop = y_missing - y_min_pad
            x_missing_crop = x_missing - x_min_pad

            axes[j].imshow(dsa_crop, cmap='gray')
            axes[j].scatter(x_valid_crop, y_valid_crop, s=8, c='yellow', label='Valid', alpha=0.7)
            axes[j].scatter(x_missing_crop, y_missing_crop, s=8, c='red', label='Missing', alpha=0.7)
            axes[j].set_title(f"{case}\nPH Frame: {ph_frame}")
            axes[j].axis('off')

        for ax in axes:
            ax.legend(loc='upper right')

        plt.suptitle("Valid (yellow) and Missing (red) Coordinates per Case", fontsize=16)
        plt.tight_layout()
        plt.show()


# if __name__ == "__main__":
#     img_path = r"Z:\Users\Artin\coiled\raw_file"
#     mask_path = r"Z:\Users\Artin\coiled\aneurysms"
#
#     img_data, path_dict = GetImages(img_path)
#     mask_data = GetMasks(mask_path)
#
#     csv_path = Path(__file__).resolve().parent.parent / "Data" / "Inlet_Treat_Parameter_wphframe.csv"
#     df = pd.read_csv(csv_path)
#     df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
#
#     num_cases = len(mask_data)
#     cols = 3
#     rows = int(np.ceil(num_cases / cols))
#     fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
#     axes = axes.flatten()
#
#     for idx, (case, mask) in enumerate(mask_data.items()):
#         if case not in img_data:
#             continue
#
#         dsa = img_data[case]
#         dcm = path_dict[case]
#
#         row = df.loc[df["ID"] == case, "PH_Frame"]
#
#         if not row.empty and not np.isnan(row.values[0]):
#             ph_frame = int(row.values[0])
#         else:
#             print(f"No PH_Frame found for case {case}")
#             continue
#         if ph_frame > dsa.shape[0]:
#             continue
#
#         vessel = VesselSegment(dsa, mask, frame_num=ph_frame, padding=10)
#         valid_y, valid_x = vessel.grow_coordinates
#         y_missing, x_missing = vessel.find_missing_coords(ph_frame, difference=90)
#
#         y_coord, x_coord = vessel.coords
#         y_min, y_max = y_coord.min(), y_coord.max()
#         x_min, x_max = x_coord.min(), x_coord.max()
#         pad = 40
#         y_min_pad = max(y_min - pad, 0)
#         y_max_pad = min(y_max + pad, vessel.dsa.shape[1] - 1)
#         x_min_pad = max(x_min - pad, 0)
#         x_max_pad = min(x_max + pad, vessel.dsa.shape[2] - 1)
#
#         dsa_crop = vessel.dsa[ph_frame, y_min_pad:y_max_pad+1, x_min_pad:x_max_pad+1]
#         y_valid_crop = valid_y - y_min_pad
#         x_valid_crop = valid_x - x_min_pad
#         y_missing_crop = y_missing - y_min_pad
#         x_missing_crop = x_missing - x_min_pad
#
#         ax = axes[idx]
#         ax.imshow(dsa_crop, cmap='gray')
#         ax.scatter(x_valid_crop, y_valid_crop, s=8, c='yellow', label='Valid', alpha=0.7)
#         ax.scatter(x_missing_crop, y_missing_crop, s=8, c='red', label='Missing', alpha=0.7)
#         ax.set_title(f"{case}\nPH Frame: {ph_frame}")
#         ax.axis('off')
#
#     for ax in axes[num_cases:]:
#         ax.axis('off')
#
#     plt.suptitle("Valid (yellow) and Missing (red) Coordinates per Case", fontsize=16)
#     plt.tight_layout()
#     plt.show()
