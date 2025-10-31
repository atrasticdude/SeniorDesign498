from pydicom.datadict import masks

from Separation import Separate
import numpy as np
import pydicom
from collections import defaultdict
from PIL import Image
import re
from utils.helperfunction import getview,sort_files_numerically



class Load(object):
    def __init__(self,image_path,inlet_path,mask_path):
        self.path = defaultdict(lambda:defaultdict(list))
        if image_path:
            self.img_data = self.GetImages(image_path)
        else:
            self.img_data = None

        if inlet_path:
            self.inlet_data = self.GetInlets(inlet_path)
        else:
            self.inlet_data = None
        if mask_path:
            self.mask_data = self.GetMasks(mask_path)
        else:
            self.mask_data = None
        if self.img_data and self.inlet_data:
            self.imginl_data = self.GetImgInlet()
        else:
            self.imginl_data = None
        if self.img_data and self.mask_data:
            self.imgmas_data = self.GetImgMask()
        else:
            self.imgmas_data = None



    @staticmethod
    def GetInlets(path):
        pre, treat, post = Separate.SepInlet(path)
        data = defaultdict(lambda: defaultdict(list))
        stages = {"PreTreatment": pre, "Treatment": treat, "PostTreatment": post}
        for stage_name, stage_dict in stages.items():
            for key, file_list in stage_dict.items():
                if len(file_list) > 1:
                    file_list = sort_files_numerically(file_list)
                for file_path in file_list:
                    try:
                        with Image.open(file_path) as img:
                            data[stage_name][key].append(np.array(img, dtype=np.float32))
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")
        return data


    @staticmethod
    def GetImages(path):
        pre, treat, post = Separate.SepImages(path)
        data = defaultdict(lambda: defaultdict(list))
        stages = {"PreTreatment": pre, "Treatment": treat, "PostTreatment": post}

        for stage_name, stage_dict in stages.items():
            for key, file_list in stage_dict.items():
                if len(file_list) > 1:
                    file_list = sort_files_numerically(file_list)
                for file_path in file_list:
                    try:
                        ds = pydicom.dcmread(file_path)
                        data[stage_name][key].append(ds.pixel_array.astype(np.float32))
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
        return data

    @staticmethod
    def GetMasks(path):
        pre, treat, post = Separate.SepMasks(path)
        data = defaultdict(lambda: defaultdict(list))
        stages = {"PreTreatment": pre, "Treatment": treat, "PostTreatment": post}
        for stage_name, stage_dict in stages.items():
            for key, file_list in stage_dict.items():
                if len(file_list) > 1:
                    file_list = sort_files_numerically(file_list)
                for file_path in file_list:
                    try:
                        with Image.open(file_path) as img:
                            data[stage_name][key].append(np.array(img, dtype=np.float32))
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")
        return data

    def GetImgInlet(self):
        data = defaultdict(lambda: defaultdict(list))
        stages = ["PreTreatment", "Treatment", "PostTreatment"]

        for stage in stages:
            images_dict = self.img_data.get(stage, {})
            inlets_dict = self.inlet_data.get(stage, {})

            for key, inlet_list in inlets_dict.items():
                if key in images_dict:
                    for img_array, inlet_array in zip(images_dict[key], inlet_list):
                        data[stage][key].append((img_array, inlet_array))
        return data

    def GetImgMask(self):
        data = defaultdict(lambda: defaultdict(list))
        stages = ["PreTreatment", "Treatment", "PostTreatment"]

        for stage in stages:
            images_dict = self.img_data.get(stage, {})
            masks_dict = self.mask_data.get(stage, {})
            for key, mask_list in masks_dict.items():
                if key in images_dict:
                    for img_array, mask_array in zip(images_dict[key], mask_list):
                        data[stage][key].append((img_array, mask_array))
        return data
    def crop(self):
    
        stage_num = {"PreTreatment": 0, "Treatment": 1, "PostTreatment": 2}
        cropped = {}

        for stage, IDs in self.imgmas_data.items():
            num = stage_num.get(stage, None)
            if num is None:
                continue

            for ID, value in IDs.items():
                for img_array, mask in value:
                    mask_bool = mask.astype(bool)

                    if len(img_array) == 1:
                        k = f"{ID}_{num}"
                        img_cropped = img_array[0].copy()
                        img_cropped[mask_bool] = 255
                        cropped[k] = img_cropped

                    elif len(img_array) == 2:
                        k1 = f"{ID}_{num}_View1"
                        k2 = f"{ID}_{num}_View2"
                        img_cropped1 = img_array[0].copy()
                        img_cropped2 = img_array[1].copy()
                        img_cropped1[mask_bool] = 255
                        img_cropped2[mask_bool] = 255
                        cropped[k1] = img_cropped1
                        cropped[k2] = img_cropped2

                    else:
                        raise ValueError(f"Unexpected number of views ({len(img_array)}) for {ID} at stage {stage}")

        return cropped

    def get_images(self):
        return self.img_data
    def get_inlets(self):
        return self.inlet_data
    def get_masks(self):
        return self.mask_data
    def get_inlet_image(self):
        return self.imginl_data
    def get_mask_image(self):
        return self.imgmas_data























