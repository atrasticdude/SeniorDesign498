import os
import re
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
