from collections import defaultdict

from scipy.ndimage import variance

from utils.helperfunction import connected_component
import numpy as np


class connectedflow(object):
    def __init__(self):
        self.out = defaultdict(list)


    def connected(self,mat,thereshold,api):
        connected_com, num_class = connected_component(mat,8)
        pics = []

        val, counts = np.unique(connected_com, return_counts=True)
        sort_index = np.argsort(counts)[::-1]
        sort_index = sort_index[:thereshold]
        top_val = val[sort_index]

        for v in top_val:
            th_mat = np.zeros_like(mat, dtype=np.uint8)
            th_mat[connected_com == v] = 255
            pics.append(th_mat)

        self.out[api] = pics

































