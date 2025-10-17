
from Separation import Separate
import numpy as np
import pydicom
from collections import defaultdict
from PIL import Image


@staticmethod
def GetInlets_eff(path):
    pre, treat, post = Separate.SepInlet(path)
    data = defaultdict(lambda: defaultdict(list))

    for key, value in pre.items():
        for item in value:
            try:
                img = Image.open(item)
                data["PreTreatment"][key].append(np.array(img, dtype=np.float32))
            except Exception as e:
                print(f"Failed to load {item}: {e}")

    for key, value in treat.items():
        for item in value:
            try:
                img = Image.open(item)
                data["Treatment"][key].append(np.array(img, dtype=np.float32))
            except Exception as e:
                print(f"Failed to load {item}: {e}")

    for key, value in post.items():
        for item in value:
            try:
                img = Image.open(item)
                data["PostTreatment"][key].append(np.array(img, dtype=np.float32))
            except Exception as e:
                print(f"Failed to load {item}: {e}")

    return data

@staticmethod
def GetInlets(path):
    pre, treat, post = Separate.SepInlet(path)
    data = defaultdict(lambda: defaultdict(list))
    stages = {"PreTreatment": pre, "Treatment": treat, "PostTreatment": post}

    for stage_name, stage_dict in stages.items():
        for key, file_list in stage_dict.items():
            for file_path in file_list:
                try:
                    with Image.open(file_path) as img:
                        data[stage_name][key].append(np.array(img, dtype=np.float32))
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

    return data

@staticmethod
def GetImages_eff(path):
    pre, treat, post = Separate.SepImages(path)
    data = defaultdict(lambda: defaultdict(list))

    for key, value in pre.items():
        for item in value:
            try:
                ds = pydicom.dcmread(item)
                data["PreTreatment"][key].append(ds.pixel_array.astype(np.float32))
            except Exception as e:
                print(f"Failed to read {item}: {e}")

    for key, value in treat.items():
        for item in value:
            try:
                ds = pydicom.dcmread(item)
                data["Treatment"][key].append(ds.pixel_array.astype(np.float32))
            except Exception as e:
                print(f"Failed to read {item}: {e}")

    for key, value in post.items():
        for item in value:
            try:
                ds = pydicom.dcmread(item)
                data["PostTreatment"][key].append(ds.pixel_array.astype(np.float32))
            except Exception as e:
                print(f"Failed to read {item}: {e}")

    return data



@staticmethod
def GetImages(path):
    pre, treat, post = Separate.SepImages(path)
    data = defaultdict(lambda: defaultdict(list))
    stages = {"PreTreatment": pre, "Treatment": treat, "PostTreatment": post}

    for stage_name, stage_dict in stages.items():
        for key, file_list in stage_dict.items():
            for file_path in file_list:
                try:
                    ds = pydicom.dcmread(file_path)
                    data[stage_name][key].append(ds.pixel_array.astype(np.float32))
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")

    return data


















