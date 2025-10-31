# from matplotlib import pyplot as plt
# from scipy.ndimage import binary_dilation
# from skimage.measure import label
# from utils.disturbedhelper import connected_component_binary, otsu_algo
# import numpy as np
# import cv2
# import pydicom
#
# class growingRegion(object):
#     def __init__(self, mat, connect):
#         self.mat = mat
#         self.connect = connect
#
#     def growing(self, threshold=None):
#         num_labels, labels, stats,  centroids = connected_component_binary(self.mat, self.connect)
#         seeds = {l: centroids[l] for l in range(1, num_labels)}
#
#         grown = np.zeros_like(self.mat, dtype=bool)
#
#         structure = np.ones((3, 3), dtype=bool)
#
#         for l, (cy, cx) in seeds.items():
#             y = int(round(cy))
#             x = int(round(cx))
#
#             seed_val = self.mat[y, x]
#
#             component_mask = (labels == l)
#
#             if threshold is None:
#                 comp_values = self.mat[component_mask]
#                 _, k = otsu_algo(comp_values)
#             else:
#                 k = threshold
#
#             region = np.zeros_like(self.mat, dtype=bool)
#             region[y, x] = True
#             prev_region = np.zeros_like(region)
#             while not np.array_equal(region, prev_region):
#                 prev_region = region.copy()
#                 region = binary_dilation(region, structure=structure) & (np.abs(self.mat - seed_val) < k)
#
#             grown |= region
#         labeled_image = label(grown, connectivity=2)
#
#         return labeled_image
#
#     # def growing(self):
#     #     num_labels, labels,stats, stats, centroids  = connected_component_binary(self.mat, self.connect)
#     #     connected = {}
#     #     for l in range(1,num_labels):
#     #         connected[l] = centroids[l]
#     #
#     #
#     #     pixel_class = {}
#     #     seed_img = np.zeros_like(self.mat)
#     #     for key, value in connected.items():
#     #         y = int(round(value[1]))
#     #         x = int(round(value[0]))
#     #         seed_img[x, y] = 1
#     #
#     #
#     #     _, k = otsu_algo(seed_img)
#     #     f_predicate = np.zeros_like(self.mat)
#     #
#     #     for l in range(1, num_labels):
#     #         seed_x = int(round(connected[l][0]))
#     #         seed_y = int(round(connected[l][1]))
#     #         seed_val = self.mat[seed_x, seed_y]
#     #         mask = (labels == l)
#     #         thresh_mask = np.abs(self.mat - seed_val) < k
#     #         final_mask = mask & thresh_mask
#     #         f_predicate[final_mask] = 1
#     #         coords = list(zip(*np.where(final_mask)))
#     #         pixel_class[l] = coords
#     #
#     #     g = np.zeros_like(self.mat)
#     #
#     #
#     #     for k, v in pixel_class.items():
#     #         seed_point = connected[k]
#     #         g[seed_point[0], seed_point[1]] = 1
#     #         y,x = zip(*v)
#     #         g[y,x] = 1
#     #
#     #     labels = connected_component_binary(g, self.connect)[1]
#     #     return labels
#
#
#
#
# img_path = r"Z:\Users\Artin\coiled\raw_file\ANY_103_1"
# ds = pydicom.dcmread(img_path)
# arr = ds.pixel_array.astype(np.float32)
#
# # Pick one frame
# i = arr[9, :, :] if arr.ndim == 3 else arr
#
# # Initialize region-growing object
# a = growingRegion(i, 8)
#
# # Run region growing (optional threshold=None for automatic Otsu)
# seg = a.growing(threshold=None)
#
# # Show results
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.title("Original")
# plt.imshow(i, cmap='gray')
# plt.axis('off')
#
# plt.subplot(1,2,2)
# plt.title("Segmented")
# plt.imshow(seg, cmap='nipy_spectral')
# plt.axis('off')
#
# plt.show()
#
#


import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


class growingRegion:
    def __init__(self, mat, connect=2):
        self.mat = mat.astype(np.float32)
        self.connect = connect

    def growing(self, threshold=None, show_debug=False):

        seed_mask = self.mat > np.mean(self.mat)
        num_labels, labels = label(seed_mask, connectivity=self.connect), label(seed_mask)[0]
        props = regionprops(num_labels, intensity_image=self.mat)

        seeds = []
        for p in props:
            y, x = p.centroid
            seeds.append((int(round(y)), int(round(x))))

        if show_debug:
            seed_img = np.zeros_like(self.mat)
            for y, x in seeds:
                seed_img[y, x] = 1
            plt.imshow(seed_img, cmap='gray')
            plt.title("Seed Points")
            plt.axis('off')
            plt.show()

        grown = np.zeros_like(self.mat, dtype=bool)
        structure = np.ones((3, 3), dtype=bool)

        for y, x in seeds:
            seed_val = self.mat[y, x]

            k = threshold
            if k is None:
                k = 0.1 * (self.mat.max() - self.mat.min())

            region = np.zeros_like(self.mat, dtype=bool)
            region[y, x] = True

            prev_region = np.zeros_like(region)
            while not np.array_equal(region, prev_region):
                prev_region = region.copy()

                region = binary_dilation(region, structure=structure) & (np.abs(self.mat - seed_val) < k)

            grown |= region

        labeled_image = label(grown, connectivity=self.connect)

        if show_debug:
            plt.imshow(labeled_image, cmap='nipy_spectral')
            plt.title("Grown Regions")
            plt.axis('off')
            plt.show()

        return labeled_image


















