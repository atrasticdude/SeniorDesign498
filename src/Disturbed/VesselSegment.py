import math

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, median_filter
from skimage.morphology import binary_erosion

from utils.disturbedhelper import global_thresholding, otsu_algo, mask_stats, skeleton_midpoint
from utils.helperfunction import cubicinter


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
        _, seg = global_thresholding(frame,threshold)
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
            seg,thresh,max_score,score = otsu_algo(frame_mask)
            if max_score > best_score:
                best_score = max_score
                best_seg = seg
                best_mask = f_mask
        return best_seg.astype(bool),best_mask

    # def adjust_frame(self):
    #    dsa_copy = self.dsa.copy()
    #    num_frames = dsa_copy.shape[0]
    #    y, x = np.where(self.mask)
    #    pixel_values = dsa_copy[:, y, x]
    #    if cubicinter is not None:
    #        x_axis = np.arange(num_frames)
    #        y_interp = np.apply_along_axis(lambda y: cubicinter(x_axis, y), 0, pixel_values)
    #    else:
    #        y_interp = pixel_values
    #    dy_dt = np.diff(y_interp, axis=0)
    #    threshold = np.percentile(np.abs(dy_dt), 80)
    #    max_deriv = np.max(np.abs(dy_dt), axis=0)
    #    pixel_spike_mask = max_deriv >= threshold
    #    y_spike = y[pixel_spike_mask]
    #    x_spike = x[pixel_spike_mask]
    #    spike_dsa = dsa_copy[:, y_spike, x_spike]
    #    return spike_dsa, y_spike, x_spike
    import numpy as np
    from scipy.ndimage import median_filter

    def detect_noisy_pixels(self, cubicinter=None, deriv_percentile=80, std_percentile=80, smoothing=True):
        dsa_copy = self.dsa.copy()
        num_frames = dsa_copy.shape[0]

        if smoothing:
            dsa_copy = np.array([median_filter(frame, size=3) for frame in dsa_copy])

        y, x = np.where(self.mask)
        pixel_values = dsa_copy[:, y, x]

        if cubicinter is not None:
            x_axis = np.arange(num_frames)
            pixel_values = np.apply_along_axis(lambda yv: cubicinter(x_axis, yv), 0, pixel_values)

        dy_dt = np.diff(pixel_values, axis=0)
        max_deriv = np.max(np.abs(dy_dt), axis=0)
        temporal_std = np.std(pixel_values, axis=0)

        deriv_threshold = np.percentile(max_deriv, deriv_percentile)
        std_threshold = np.percentile(temporal_std, std_percentile)
        noise_mask = (max_deriv >= deriv_threshold) | (temporal_std >= std_threshold)

        y_noisy = y[noise_mask]
        x_noisy = x[noise_mask]
        y_clean = y[~noise_mask]
        x_clean = x[~noise_mask]

        noisy_dsa = dsa_copy[:, y_noisy, x_noisy]
        clean_dsa = dsa_copy[:, y_clean, x_clean]

        return clean_dsa , x_noisy, y_noisy, y_clean, x_clean

    # def find_true_diff(self,mask_image):
    #     mask = mask_image.copy()
    #     mask_area, mask_w, mask_h = mask_stats(mask)
    #     area_thres = mask_area / 5
    #     width_thres = mask_w / 5
    #     height_thres = mask_h / 5
    #
    #     seg, thresh, max_score, score = otsu_algo(mask_image)
    #     pixels = np.arange(len(score))
    #
    #     zipped = list(zip(score, pixels))
    #     sorted_list = sorted(zipped, key=lambda x: x[0], reverse=True)
    #     top_ten = sorted_list[:10]
    #     pixel_indices = [item[1] for item in top_ten]
    #
    #     pixel_centroids = []
    #     for p in pixel_indices:
    #         thres_mask = mask <= p
    #         thres_mask_uint8 = thres_mask.astype(np.uint8) * 255
    #         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thres_mask_uint8)
    #         # if num_labels > 8 or num_labels < 2:
    #         #     continue
    #
    #         valid_centroids = []
    #         for i in range(1, num_labels):
    #             x, y, w, h, area = stats[i]
    #             # if area > area_thres and w > width_thres and h > height_thres:
    #             #     continue
    #             component = labels == i
    #             if w < 4:
    #                 sy, sx = skeleton_midpoint(component)
    #                 valid_centroids.append((sx, sy))
    #             else:
    #                 valid_centroids.append(tuple(centroids[i]))
    #         pixel_centroids.append((p, valid_centroids))
    #     return pixel_centroids
    def find_true_diff(self, mask_image):
        mask = mask_image.copy()
        mask_area, mask_w, mask_h = mask_stats(mask)
        area_thres = mask_area / 5
        width_thres = mask_w / 5
        height_thres = mask_h / 5

        seg, thresh, max_score, score = otsu_algo(mask_image)
        pixels = np.arange(len(score))
        zipped = list(zip(score, pixels))
        top_ten = sorted(zipped, key=lambda x: x[0], reverse=True)[:10]
        pixel_indices = [item[1] for item in top_ten]

        pixel_centroids = []

        for p in pixel_indices:
            thres_mask = (mask_image <= p).astype(np.uint8) * 255

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thres_mask)

            valid_centroids = []

            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                # if area > area_thres or w > width_thres or h > height_thres:
                #     continue
                component = labels == i
                if w < 4 or h < 4:
                    midpoint = skeleton_midpoint(component)
                    if midpoint is not None:
                        sy, sx = midpoint
                        valid_centroids.append((sx, sy))
                else:
                    valid_centroids.append(tuple(centroids[i]))

            pixel_centroids.append((p, valid_centroids))

        return pixel_centroids






































