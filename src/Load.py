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
        # else:
        #     self.mask_data = None
        # if self.img_data and self.inlet_data:
        #     self.imginl_data = self.GetImgInlet()
        # else:
        #     self.imginl_data = None
        # if self.img_data and self.mask_data:
        #     self.imgmas_data = self.GetImgMask()
        # else:
        #     self.imgmas_data = None



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
























