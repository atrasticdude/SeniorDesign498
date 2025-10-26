import os
import re
import numpy as np


def get_bases_id(name):
    return "_".join(name.split("_")[:2])

def crop(name):
    return "_".join(name.split("_")[:-1])

def sepdot(name):
    return name.split(".")[0]

def getview(name):
    return sepdot(os.path.basename(name)).split("_")[2]


def sort_files_numerically(file_list):
    return sorted(file_list, key=lambda x: int(re.search(r'\d+', getview(x)).group()))

def BolusArrivalTime1D(y):
    thereshold = np.max(y) * 0.1
    bat_index = next((j for j, value in enumerate(y) if value > thereshold), -1)
    if bat_index > 0:
        bat_index -= 1
    return bat_index

