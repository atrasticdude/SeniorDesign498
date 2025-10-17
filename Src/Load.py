
from Separation import Separate
import numpy as np
import pydicom
from collections import defaultdict
from PIL import Image

class Load(object):
    def __init__(self,image_path,inlet_path):
        if image_path:
            self.img_data = self.GetImages_eff(image_path)
        else:
            self.img_data = None

        if inlet_path:
            self.inlet_data = self.GetInlets_eff(inlet_path)
        else:
            self.inlet_data = None

        if image_path and inlet_path:
            self.imginl_data = self.GetImgInlet(image_path=image_path, inlet_path=inlet_path)
        else:
            self.imginl_data = None

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

    # @staticmethod
    # def GetImgInlet(image_path = None , inlet_path = None):
    #
    #     if image_path is None or inlet_path is None:
    #         raise ValueError("Both image_path and inlet_path must be provided")
    #
    #     pre, treat, post = Separate.ImageInlet(image_path, inlet_path)
    #     data = defaultdict(lambda: defaultdict(list))
    #     stages = {"PreTreatment": pre, "Treatment": treat, "PostTreatment": post}
    #     for stage_name, stage_dict in stages.items():
    #         for key, file_list in stage_dict.items():
    #             for file_path in file_list:
    #                 ds = pydicom.
    #                 img = Image.open(file_path[1])























