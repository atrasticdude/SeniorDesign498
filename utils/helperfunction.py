import os
import re
from collections import deque
import numpy as np
import cv2
from tensorboard.compat.tensorflow_stub.dtypes import float32
from scipy.interpolate import interp1d

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

def getindices(x):
    return np.transpose(np.nonzero(x))

def okm_grade_from_fill(fill_percent):
    if np.isnan(fill_percent):
        return "N/A"
    elif fill_percent > 95:
        return "A"
    elif fill_percent > 50:
        return "B"
    elif fill_percent > 0:
        return "C"
    else:
        return "D"

def covariance(x,y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.mean((x- mean_x)**2)
    var_y = np.mean((y-mean_y)**2)
    cov_xy = np.mean((x- mean_x)*(y - mean_y)**2)
    cov_matrix = np.array([[var_x, cov_xy],
                           [cov_xy, var_y]])
    print(cov_matrix)

def cubicinter(x,y):
    f_cubic = interp1d(x,y, kind='cubic')
    return f_cubic





# def connected_sets(mat, class_label=1, conn_type="4point"):
#     mat = np.array(mat)
#     x, y = mat.shape
#     visited = np.zeros_like(mat, dtype=bool)
#     connected_components = []
#     if conn_type == "4point":
#         neighbors = [(-1,0), (1,0), (0,-1), (0,1)]
#     elif conn_type == "8point":
#         neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
#     else:
#         raise ValueError("conn_type must be '4point' or '8point'")
#     for i in range(x):
#         for j in range(y):
#             if mat[i, j] == class_label and not visited[i, j]:
#                 q = deque([(i, j)])
#                 visited[i, j] = True
#                 component = [(i, j)]
#                 while q:
#                     ci, cj = q.popleft()
#                     for di, dj in neighbors:
#                         ni, nj = ci + di, cj + dj
#                         if 0 <= ni < x and 0 <= nj < y:
#                             if mat[ni, nj] == class_label and not visited[ni, nj]:
#                                 visited[ni, nj] = True
#                                 q.append((ni, nj))
#                                 component.append((ni, nj))
#
#                 connected_components.append(component)
#
#     return connected_components










