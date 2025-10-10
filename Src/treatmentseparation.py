# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 13:13:56 2025

@author: artas
"""

import os
from collections import defaultdict

def get_bases_id(name):
    return "_".join(name.split("_")[:2])

def get_treatment_steps(name):
    return "_".join(name.split("_")[:3])

def get_masks_base_id(name):
    return "_".join(name.split("_")[1])

def get_masks_steps(name):
    return "_".join(name.split("_")[:2])

def get_masks(name):
    return "_".join(name.split("_")[:3])

def get_views(name):
    return "_".join(name.split("_")[:4])



class get_files(object):
    @staticmethod
    def sep_treatment_steps(path = r"Z:\Projects\Aneurysm\Raw Imaging Data\Coil"):
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
                step = get_treatment_steps(f)
                full_path = os.path.join(path, f)
                if step.endswith("_0"):
                    pretreatment[baseID].append(full_path)
                elif step.endswith("_1"):
                    treatment[baseID].append(full_path)
                elif step.endswith("_2"):
                    posttreatment[baseID].append(full_path)
        return IDs,pretreatment, treatment, posttreatment
    
    
    @staticmethod         
    def sep_mask_steps(path = ""):
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
                step = get_treatment_steps(f)
                full_path = os.path.join(path, f)
                if step.endswith("_0"):
                    pretreatment[baseID].append(full_path)
                elif step.endswith("_1"):
                    treatment[baseID].append(full_path)
                elif step.endswith("_2"):
                    posttreatment[baseID].append(full_path)
        return IDs,pretreatment, treatment, posttreatment


    @staticmethod
    def image_mask_tuple(mask_path, image_path):
        masks_dic, pre_masks, treat_masks, post_masks = get_files.sep_mask_steps(mask_path)
        images_dic, pre_images, treat_images, post_images = get_files.sep_treatment_steps(image_path)
        
        image_masks_tup_pre = []
        image_masks_tup_treat = []
        image_masks_tup_post = []

        for key, image_files in images_dic.items():
            if key in masks_dic:
                for img_file in image_files:
                    if img_file in pre_images.get(key, []) and img_file in pre_masks.get(key, []):
                        image_masks_tup_pre.append((img_file, img_file))  
                    elif img_file in treat_images.get(key, []) and img_file in treat_masks.get(key, []):
                        image_masks_tup_treat.append((img_file, img_file))
                    elif img_file in post_images.get(key, []) and img_file in post_masks.get(key, []):
                        image_masks_tup_post.append((img_file, img_file))

        return image_masks_tup_pre, image_masks_tup_treat, image_masks_tup_post

        
