from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from src.APIMaps.GetAPI import getAPI
from src.Separation import Separate
from utils.helperfunction import sort_files_numerically
import pandas as pd


def GetInlets(path):
    pre, _, _ = Separate.SepInlet(path)
    data = {}

    for key, file_list in pre.items():
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
    pre, _, _ = Separate.SepImages(path)
    data = {}
    path_dict = {}
    for key, file_list in pre.items():
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

def GetMasks(path):
    pre, _, _ = Separate.SepMasks(path)
    data = {}

    for key, file_list in pre.items():
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






if __name__ == "__main__":

    # --- Paths ---
    img_path = r"Z:\Users\Artin\coiled\raw_file\ANY_391_0"
    mask_path = r"Z:\Users\Artin\coiled\aneurysms\ANY_391_0.tif"
    inlet_path = r"Z:\Users\Artin\coiled\inlets\ANY_391_0_inl.tif"

    # Extract case number from filename
    case_number = "ANY_391"

    # --- Load DICOM ---
    ds = pydicom.dcmread(img_path)
    arr = ds.pixel_array.astype(np.float32)

    if arr.ndim == 2:
        arr = arr[None, :, :]  # make 3D if needed

    # --- Load mask and inlet ---
    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img) > 0

    inlet_img = Image.open(inlet_path).convert("L")
    inlet = np.array(inlet_img) > 0

    # --- Initialize getAPI ---
    api_obj = getAPI(dsa=arr, inlet=inlet, dsa_temp=ds, mask = mask, frac=0.1, show_mask_stats=False,show_okelly_scale=True)

    scale = api_obj.get_okm_scale()

    csv_path = Path(__file__).resolve().parent.parent / "Data" / "OKM_Scale_Pre.csv"
    df = pd.read_csv(csv_path)
    df.loc[case_number,"Scale"] = scale
    df.to_csv(csv_path, index = False)







