
import os
from collections import defaultdict
import numpy as np
from utils.helperfunction import get_bases_id, sepdot, crop


class Separate(object):
    @staticmethod
    def SepImages(path =r"Z:\Users\Artin\coiled\raw_file"):
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
                full_path = os.path.join(path, f)
                if full_path.endswith("_0"):
                    pretreatment[baseID].append(full_path)
                elif full_path.endswith("_1"):
                    treatment[baseID].append(full_path)
                elif full_path.endswith("_2"):
                    posttreatment[baseID].append(full_path)
        return pretreatment, treatment, posttreatment


    @staticmethod
    def SepInlet(path = r"Z:\Users\Artin\coiled\inlets"):
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
                step = crop(f)
                full_path = os.path.join(path, f)
                if step.endswith("_0"):
                    pretreatment[baseID].append(full_path + ".tif")
                elif step.endswith("_1"):
                    treatment[baseID].append(full_path + ".tif")
                elif step.endswith("_2"):
                    posttreatment[baseID].append(full_path + ".tif")
        return pretreatment, treatment, posttreatment

    @staticmethod
    def SepMasks(path=r"Z:\Users\Artin\coiled\aneurysms"):
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
                full_path = os.path.join(path, f)
                if full_path.endswith("_0"):
                    pretreatment[baseID].append(full_path + ".tif")
                elif full_path.endswith("_1"):
                    treatment[baseID].append(full_path + ".tif")
                elif full_path.endswith("_2"):
                    posttreatment[baseID].append(full_path + ".tif")
        return pretreatment, treatment, posttreatment



    @staticmethod
    def ImageInlet(image_path = r"Z:\Users\Artin\coiled\raw_file" , mask_path = r"Z:\Users\Artin\coiled\inlets"):
        pre_inlet, treat_inlet, post_inlet = Separate.SepInlet(mask_path)
        pre_images, treat_images, post_images = Separate.SepImages(image_path)


        image_inlet_tup_pre = defaultdict(list)
        image_inlet_tup_treat = defaultdict(list)
        image_inlet_tup_post = defaultdict(list)

        for key, file in pre_inlet.items():
            if key in pre_images:
                for mask_item in file:
                    image_item = os.path.join(image_path, crop(sepdot(os.path.basename(mask_item))))
                    if image_item in pre_images.get(key,[]):
                        image_inlet_tup_pre[key].append((image_item, mask_item))

        for key, file in treat_inlet.items():
            if key in treat_images:
                for mask_item in file:
                    image_item = os.path.join(image_path, crop(sepdot(os.path.basename(mask_item))))
                    if image_item in treat_images.get(key,[]):
                        image_inlet_tup_treat[key].append((image_item, mask_item))

        for key, file in post_inlet.items():
            if key in post_images:
                for mask_item in file:
                    image_item = os.path.join(image_path, crop(sepdot(os.path.basename(mask_item))))
                    if image_item in post_images.get(key,[]):
                        image_inlet_tup_post[key].append((image_item, mask_item))

        return image_inlet_tup_pre, image_inlet_tup_treat, image_inlet_tup_post






