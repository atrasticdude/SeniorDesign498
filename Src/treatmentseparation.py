# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 13:13:56 2025

@author: artas
"""



import os
from collections import defaultdict
import numpy as np

def get_bases_id(name):
    return "_".join(name.split("_")[:2])

# def get_treatment_steps(name):
#     return "_".join(name.split("_")[:3])

# def get_masks_base_id(name):
#     return "_".join(name.split("_")[1])

def get_masks_steps(name):
    return "_".join(name.split("_")[:-1])
#
# def get_masks(name):
#     return "_".join(name.split("_")[-1])
#
# def get_views(name):
#     return "_".join(name.split("_")[:4])

def get_length(name):
    return len(name.split("_"))

def crop(name,index):
    return "_".join(name.split("_")[:index])

def sepdot(name):
    return name.split(".")[0]




class get_files(object):
    @staticmethod
    def sep_treatment_steps(path = r"Z:\Users\Artin\coiled\raw_file"):
        files = os.listdir(path)
        IDs = defaultdict(list)
        for filename in files:
            baseID = get_bases_id(filename)
            IDs[baseID].append(filename)

        pretreatment = defaultdict(list)
        treatment = defaultdict(list)
        posttreatment = defaultdict(list)

        for baseID, file_list in IDs.items():
            for f in file_list:
                full_path = os.path.join(path, f)
                if full_path.endswith("_0"):
                    pretreatment[baseID].append(full_path)
                elif full_path.endswith("_1"):
                    treatment[baseID].append(full_path)
                elif full_path.endswith("_2"):
                    posttreatment[baseID].append(full_path)
        return pretreatment, treatment, posttreatment


    @staticmethod
    def sep_mask_steps(path = r"Z:\Users\Artin\coiled\inlets"):
        files = os.listdir(path)
        IDs = defaultdict(list)
        for filename in files:
            baseID = get_bases_id(filename)
            IDs[baseID].append(sepdot(filename))

        pretreatment = defaultdict(list)
        treatment = defaultdict(list)
        posttreatment = defaultdict(list)

        for baseID, file_list in IDs.items():
            for f in file_list:
                step = get_masks_steps(f)
                full_path = os.path.join(path, f)
                if step.endswith("_0"):
                    pretreatment[baseID].append(full_path + ".tif")
                elif step.endswith("_1"):
                    treatment[baseID].append(full_path + ".tif")
                elif step.endswith("_2"):
                    posttreatment[baseID].append(full_path + ".tif")
        return pretreatment, treatment, posttreatment


    @staticmethod
    def image_mask_tuple(mask_path, image_path):
         pre_masks, treat_masks, post_masks = get_files.sep_mask_steps(mask_path)
         pre_images, treat_images, post_images = get_files.sep_treatment_steps(image_path)

        image_masks_tup_pre = []
        image_masks_tup_treat = []
        image_masks_tup_post = []

        for key, image_files in pre_images.item():
            if key in pre_masks:
                image_files = image_files.sort(key = len)

        for key, image_files in images_dic.items():
            if key in masks_dic:
                for item in masks_dic[key]:
                    index = np.max()
                    if img_file in pre_images.get(key, []) and img_file in pre_masks.get(key, []):
                        image_masks_tup_pre.append((img_file, img_file))
                    elif img_file in treat_images.get(key, []) and img_file in treat_masks.get(key, []):
                        image_masks_tup_treat.append((img_file, img_file))
                    elif img_file in post_images.get(key, []) and img_file in post_masks.get(key, []):
                        image_masks_tup_post.append((img_file, img_file))

        return image_masks_tup_pre, image_masks_tup_treat, image_masks_tup_post


a = get_files()
b = a.sep_mask_steps()[0]
print(b["ANY_331"])

