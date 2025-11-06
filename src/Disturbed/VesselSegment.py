import math
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_erosion

from utils.disturbedhelper import global_thresholding


class VesselSegment(object):
    def __init__(self,dsa,mask,frame_num, padding = 20):
        self.dsa = dsa
        self.mask = mask.astype(bool)
        self.coords = self.find_mask_boundaries()
        self.segmented = self.add_segment_mask(frame_num)
        self.f_mask = self.zoom_mask(self.segmented, self.coords,padding)
        self.grow_coordinates = self.find_valid_coords_distance(self.coords, self.f_mask)



    def find_mask_boundaries(self):
       eroded = binary_erosion(self.mask)
       boundaries = self.mask & (~eroded)
       coord = np.where(boundaries)
       return coord

    def zoom_mask(self, frame, coords, padding = 20):
        y_coord, x_coord = coords
        y_min, y_max = y_coord.min(), y_coord.max()
        x_min, x_max = x_coord.min(), x_coord.max()
        y_min_pad = max(y_min - padding, 0)
        y_max_pad = min(y_max + padding, frame.shape[0] - 1)
        x_min_pad = max(x_min - padding, 0)
        x_max_pad = min(x_max + padding, frame.shape[1] - 1)

        f_mask = np.zeros_like(frame, dtype = bool)
        f_mask[y_min_pad:y_max_pad + 1, x_min_pad:x_max_pad + 1] = True
        return f_mask


    def find_valid_coords_distance(self, coords, f_mask, margin=2):
        region = self.segmented & f_mask
        eroded_region = binary_erosion(region)
        boundary_mask = region & (~eroded_region)
        dist = distance_transform_edt(~boundary_mask)

        y_coords, x_coords = coords
        valid = [(y, x) for y, x in zip(y_coords, x_coords) if dist[y, x] > margin]

        if valid:
            valid_y, valid_x = zip(*valid)
            return np.array(valid_y), np.array(valid_x)
        else:
            return np.array([]), np.array([])

    def add_segment_mask(self, frame_num):
        frame = self.dsa[frame_num, :, :]
        _, seg = global_thresholding(frame)
        seg = seg.copy()
        seg = seg.astype(bool)
        seg[self.mask] = 0
        return seg



