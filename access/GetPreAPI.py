from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from GetAPI import getAPI
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


if __name__ == "__main__":
    img_path = r"Z:\Users\Artin\coiled\raw_file"
    inlet_path = r"Z:\Users\Artin\coiled\inlets"

    img_data, path_dict = GetImages(img_path)
    inlet_data = GetInlets(inlet_path)

    param_keys = ["PH", "TTP", "AUC", "MTT", "Max_Df", "BAT"]
    auc_intervals = ["AUC0.5", "AUC1", "AUC1.5", "AUC2.0"]
    header = param_keys + auc_intervals

    df = pd.DataFrame(columns=header, index=list(img_data.keys()))

    for case, data in img_data.items():
        if case in inlet_data:
            api_obj = getAPI(
                dsa=data,
                inlet=inlet_data[case],
                dsa_temp=path_dict[case],
                frac=0.1,
                show_mask_stats=False
            )

            inlet_params = api_obj.get_inlet_API()

            for key in param_keys:
                df.loc[case, key] = inlet_params.get(key, np.nan)

            auc_values = inlet_params.get("AUC_interval", [np.nan] * 4)
            if auc_values is not None:
                for i, auc_val in enumerate(auc_values):
                    df.loc[case, auc_intervals[i]] = auc_val

    output_csv = "inlet_results_pre.csv"
    df.to_csv(output_csv, index=True)
    print(f"CSV file '{output_csv}' created successfully!")

















