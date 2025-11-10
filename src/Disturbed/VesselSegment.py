import math
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_erosion

from utils.disturbedhelper import global_thresholding, otsu_algo


class VesselSegment(object):
    def __init__(self,dsa,mask,frame_num, padding = None):
        self.dsa = dsa
        self.mask = mask.astype(bool)
        self.coords = self.find_mask_boundaries()
        if padding is None:
            self.segmented, self.f_mask = self.add_sgement_mask_bestpadding(frame_num,self.coords)
        else:
            self.f_mask = self.zoom_mask(self.dsa[frame_num,:,:], self.coords,padding)
            self.segmented = self.add_segment_mask(frame_num)
        self.grow_coordinates = self.find_valid_coords_distance(self.coords)



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


    def find_valid_coords_distance(self, coords, margin=3):
        region = self.segmented
        eroded_region = binary_erosion(region)
        boundary_mask = region & (~eroded_region)
        dist = distance_transform_edt(~boundary_mask)

        y_coords, x_coords = coords
        valid = [(y, x) for y, x in zip(y_coords, x_coords) if dist[y, x] > margin]

        if valid:
            valid_y, valid_x = zip(*valid)
            valid_y = np.array(valid_y, dtype=int)
            valid_x = np.array(valid_x, dtype=int)
            return valid_y, valid_x
        else:
            return np.array([], dtype=int), np.array([], dtype=int)

    def add_segment_mask(self, frame_num):
        frame = self.dsa[frame_num, :, :].copy()
        frame[self.mask] = 0
        frame[~self.f_mask] = frame.max() + 1
        T, seg = global_thresholding(frame)
        return seg.astype(bool)
    # def add_segment_mask(self, frame_num):
    #     frame = self.dsa[frame_num, :, :]
    #     _, seg = global_thresholding(frame)
    #     seg = seg.copy()
    #     seg = seg.astype(bool)
    #     seg[self.mask] = 0
    #     return seg

    def find_missing_coords(self,frame_num, difference):
        frame = self.dsa[frame_num, :, :].copy()
        frame[~self.f_mask] = frame.max() + 1
        hist, bins = np.histogram(frame, bins=256, range=(0, 256))
        sorted_hist_indices = np.argsort(hist)[::-1]
        sorted_counts = hist[sorted_hist_indices]
        sorted_bins = bins[sorted_hist_indices]
        threshold = min(sorted_bins[0],sorted_bins[1])
        for k in range(3,len(sorted_bins)-1):
            if np.abs(sorted_bins[k+1] - sorted_bins[k]) > difference:
                threshold = min(sorted_bins[k+1],sorted_bins[k])
                break
        _, seg = global_thresholding(frame,threshold=threshold)
        seg = seg.astype(bool)
        mask_coords = set(zip(*np.where(self.mask)))
        seg_coords = set(zip(*np.where(seg)))
        missing_coords = mask_coords - seg_coords
        if missing_coords:
            y_missing, x_missing = zip(*missing_coords)
            return np.array(y_missing, dtype=int), np.array(x_missing, dtype=int)
        else:
            return np.array([], dtype=int), np.array([], dtype=int)

    def add_sgement_mask_bestpadding(self,frame_num,coords):
        padding = [10,20,30,40,50]
        frame = self.dsa[frame_num,:,:]
        best_mask = None
        best_score = -np.inf
        best_seg = None
        for index,p in enumerate(padding):
            f_mask = self.zoom_mask(frame, coords,p)
            frame_mask = frame.copy()
            # frame_mask[self.mask] = 0
            frame_mask[~f_mask] = frame.max() + 1
            seg,thresh,max_score = otsu_algo(frame_mask)
            if max_score > best_score:
                best_score = max_score
                best_seg = seg
                best_mask = f_mask
        return best_seg.astype(bool),best_mask


















