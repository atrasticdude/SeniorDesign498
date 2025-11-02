from pydicom.datadict import masks

from Separation import Separate
import numpy as np
import pydicom
from collections import defaultdict
from PIL import Image
import re
from utils.helperfunction import getview,sort_files_numerically



class Load(object):
    def __init__(self,image_path,inlet_path, mask_path):
        self.img_path = Separate.SepImages(image_path)
        self.mask_path = Separate.SepImages(mask_path)
        self.inlet_path = Separate.SepImages(inlet_path)
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




    @staticmethod
    def GetInlets(path):
        pre, treat, post = Separate.SepInlet(path)
        data = {}
        stages = {"0": pre, "1": treat, "2": post}
        
        for stage_name, stage_dict in stages.items():
            for key, file_list in stage_dict.items():
                if len(file_list) > 1:
                    file_list = sort_files_numerically(file_list)
        
                for idx, file_path in enumerate(file_list):
                    try:
                        with Image.open(file_path) as img:
                            arr = np.array(img, dtype=np.float32)
                            name = f"{key}_{stage_name}"
                            if len(file_list) > 1:
                                name += f"_View{idx+1}"
                            data[name] = arr
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")
        return data


    @staticmethod
    def GetImages(path):
        pre, treat, post = Separate.SepImages(path)
        data = {}
        stages = {"0": pre, "1": treat, "2": post}
    
        for stage_num, stage_dict in stages.items():
            for key, file_list in stage_dict.items():
                if len(file_list) > 1:
                    file_list = sort_files_numerically(file_list)
    
                for idx, file_path in enumerate(file_list):
                    try:
                        ds = pydicom.dcmread(file_path)
                        img_array = ds.pixel_array.astype(np.float32)
                        name = f"{key}_{stage_num}"
                        if len(file_list) > 1:
                            name += f"_View{idx+1}"
                        data[name] = img_array
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
        return data


    @staticmethod
    def GetMasks(path):
        pre, treat, post = Separate.SepMasks(path)
        data = {}
        stages = {"0": pre, "1": treat, "2": post}
        
        for stage_name, stage_dict in stages.items():
            for key, file_list in stage_dict.items():
                if len(file_list) > 1:
                    file_list = sort_files_numerically(file_list)
                for idx, file_path in enumerate(file_list):
                    try:
                        with Image.open(file_path) as img:
                            arr = np.array(img, dtype=np.float32)
                            name = f"{key}_{stage_name}"
                            if len(file_list) > 1:
                                name += f"_View{idx+1}"
                            data[name] = arr
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")
        return data


    def crop(self):
    
            cropped = {}
        
            for key, img_array in self.img_data.items():
                if key not in self.mask_data:
                    print(f" No mask found for {key}, skipping...")
                    continue
                mask = self.mask_data[key]
                mask_bool = mask.astype(bool)
                if img_array.ndim == 3:
                    img_cropped = img_array.copy()
                    img_cropped[:, mask_bool] = 255
                else:
                    raise ValueError(f"Unexpected image shape {img_array.shape} for key {key}")
        
                cropped[key] = img_cropped
            return cropped

    def img_sizes(self):
        imgs = {}
        stages = {"0": self.img_data[0], "1": self.img_data[1], "2": self.img_data[2]}

        for stage_num, stage_dict in stages.items():
            for key, file_list in stage_dict.items():
                if len(file_list) > 1:
                    file_list = sort_files_numerically(file_list)

                for idx, file_path in enumerate(file_list):
                    try:
                        ds = pydicom.dcmread(file_path)
                        h = getattr(ds, "Rows", None)
                        w = getattr(ds, "Columns", None)
                        frames = int(getattr(ds, "NumberOfFrames", 1))
                        if h is None or w is None:
                            print(f"Missing dimension info in {file_path}")
                            continue
                        name = f"{key}_{stage_num}"
                        if len(file_list) > 1:
                            name += f"_View{idx + 1}"
                        imgs[name] = (frames, h, w)
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
        return imgs
    def mask_sizes(self):
        sizes = {}
        stages = {"0": self.mask_data[0], "1": self.mask_data[1], "2": self.mask_data[2]}

        for stage_num, stage_dict in stages.items():
            for key, file_list in stage_dict.items():
                if len(file_list) > 1:
                    file_list = sort_files_numerically(file_list)
                for idx, file_path in enumerate(file_list):
                    try:
                        ds = pydicom.dcmread(file_path)
                        h = getattr(ds, "Rows", None)
                        w = getattr(ds, "Columns", None)
                        frames = int(getattr(ds, "NumberOfFrames", 1))
                        if h is None or w is None:
                            print(f"Missing dimension info in {file_path}")
                            continue
                        name = f"{key}_{stage_num}"
                        if len(file_list) > 1:
                            name += f"_View{idx + 1}"
                        sizes[name] = (frames, h, w)
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
        return sizes


    def mask_sizes(self):
        sizes = {}
        stages = {"0": self.mask_data[0], "1": self.mask_data[1], "2": self.mask_data[2]}

        for stage_num, stage_dict in stages.items():
            for key, file_list in stage_dict.items():
                if len(file_list) > 1:
                    file_list = sort_files_numerically(file_list)
                for idx, file_path in enumerate(file_list):
                    try:
                        with Image.open(file_path) as img:
                            w, h = img.size
                            name = f"{key}_{stage_num}"
                            if len(file_list) > 1:
                                name += f"_View{idx + 1}"
                            sizes[name] = (h, w)
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
        return sizes

    # def okm_grade(self):



    def get_images(self):
        return self.img_data
    def get_inlets(self):
        return self.inlet_data
    def get_masks(self):
        return self.mask_data
    def get_mask_sizes(self):
        return self.mask_sizes
    def get_img_sizes(self):
        return self.img_sizes
























