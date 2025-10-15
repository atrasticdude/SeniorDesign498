
from treatmentseparation import get_files
import numpy as np
import pydicom



def pixel_data(links = get_files.sep_treatment_steps()):
    pre,treat,post = links
    data = {}

    for i in pre:
        ds = pydicom.dcmread(i)
        data[i] = ds.pixel_array.astype(np.float32)

    for i in treat:
        ds = pydicom.dcmread(i)
        data[i] = ds.pixel_array.astype(np.float32)

    for i in post:
        ds = pydicom.dcmread(i)
        data[i] = ds.pixel_array.astype(np.float32)

    return data
