import os

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
        IDs = {}
        for filename in files:
            baseID = get_bases_id(filename)
            IDs.setdefault(baseID, []).append(filename)

        pretreatment = []
        treatment = []
        posttreatment = []

        for baseID, file_list in IDs.items():
            for f in file_list:
                step = get_treatment_steps(f)
                full_path = os.path.join(path, f)
                if step.endswith("_0"):
                    pretreatment.append(full_path)
                elif step.endswith("_1"):
                    treatment.append(full_path)
                elif step.endswith("_2"):
                    posttreatment.append(full_path)
        return pretreatment, treatment, posttreatment
        
