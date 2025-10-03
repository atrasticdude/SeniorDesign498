from skimage.transform import resize
import pydicom
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as pl


path = r"Z:\Projects\Aneurysm\Raw Imaging Data\Coil"

files = os.listdir(path)

def get_bases_id(name):
    return "_".join(name.split("_")[:2])
def get_treatment_steps(name):
    return "_".join(name.split("_")[:3])
IDs = {}

                    
for _,iD in enumerate(files):
    baseID = get_bases_id(iD)
    if baseID in IDs:
        IDs[baseID].append(get_treatment_steps(iD))
    else:
        IDs[baseID] = [get_treatment_steps(iD)]
pretreatment = []
treatment = []
posttreatment = []
                    
                        
for key,value in IDs.items():
    if (key + "_0") in value:
        pretreatment.append(key + "_0")
    if (key + "_1") in value:
        treatment.append(key + "_1")
    if (key + "_2") in value:
        posttreatment.append(key + "_2")
        
