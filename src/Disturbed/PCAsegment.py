import cv2
import pydicom
import tifffile
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from qiskit_ibm_runtime.visualization.utils import pie_slice

from utils.disturbedhelper import global_thresholding


class PCAsegment(object):
    def __init__(self, dsa, mask):
        self.dsa = dsa
        self.mask = mask.astype(bool)

    def pca_matrix(self, K=5):
        mask = self.mask
        dsa = self.dsa
        T = dsa.shape[0]
        N = mask.sum().item()
        X = dsa[:, mask]
        mean_pixel = X.mean(dim=0, keepdim=True)
        Xc = X - mean_pixel
        U, S, V = torch.pca_lowrank(Xc, q=K)
        scores = U * S
        components = V.T

        pixel_contribution = torch.sqrt((components ** 2).sum(dim=0))
        pixel_contribution_map = torch.zeros(mask.shape)
        pixel_contribution_map[mask] = pixel_contribution

        pixel_contribution_matrix = scores @ components

        pixel_contribution_matrix += mean_pixel

        return pixel_contribution_matrix, pixel_contribution_map

    def min_pixs(self, pc_matrix):
        dsa = self.dsa
        mask = torch.tensor(self.mask, dtype=torch.bool)
        y, x = torch.where(mask)
        time = pc_matrix.shape[0]
        min_value_pixel = torch.argmin(torch.tensor(pc_matrix), dim=1)

        frame_stack = []
        thresholds = []

        for i in range(1,time):
            crop_out = dsa[i-1][y, x]
            threshold = crop_out[min_value_pixel[i]]
            segmented = (crop_out >= threshold).to(torch.uint8)
            stack = torch.zeros_like(mask, dtype=torch.uint8)
            stack[y, x] = segmented
            frame_stack.append(stack)
            thresholds.append(threshold)
        return frame_stack, thresholds

    def connected_components(self, K = 2, threshold = 0.85):
        mask = torch.tensor(self.mask, dtype=torch.bool)
        k_arr = torch.arange(1, K+1)
        map_arr = []

        for k in k_arr:
            _,pc_map = self.pca_matrix(K = k)
            map_arr.append(pc_map)
        concat = torch.zeros_like(mask, dtype=torch.uint8)

        for pc_map in map_arr:
            thresh = torch.quantile(pc_map[mask], threshold)
            pc_bin = (pc_map >= thresh).to(torch.uint8)
            pc_np = pc_bin.cpu().numpy()
            pc_uint8 = (pc_np * 255).astype('uint8')
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pc_uint8)
            if num_labels <= 1:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = areas.argmax() + 1
            largest_comp = (labels == largest_idx).astype(np.uint8)
            concat = concat | torch.from_numpy(largest_comp)

        return concat







if __name__ == '__main__':
    dcm = pydicom.dcmread(r"Z:\Users\Artin\coiled\raw_file\ANY_375_1")
    dsa = dcm.pixel_array
    # dsa = dsa[4:11,:,:]

    if dsa.ndim == 2:
        dsa = np.expand_dims(dsa, axis=0)

    mask = tifffile.imread(r"Z:\Users\Artin\coiled\aneurysms\ANY_375_1.tif").astype(bool)

    dsa_tensor = torch.tensor(dsa, dtype=torch.float32)

    import torch
    import matplotlib.pyplot as plt

    pca_seg = PCAsegment(dsa_tensor, mask)
    concat = pca_seg.connected_components(K=2)

    plt.figure(figsize=(6, 6))
    plt.imshow(concat.numpy(), cmap='hot')
    plt.axis('off')
    plt.show()

    import numpy as np
    import matplotlib.pyplot as plt

    dsa_frame = dsa[6]
    dsa_norm = (dsa_frame - dsa_frame.min()) / (dsa_frame.max() - dsa_frame.min())
    dsa_rgb = np.stack([dsa_norm] * 3, axis=-1)

    concat_mask = concat.numpy().astype(bool)
    dsa_rgb[concat_mask, 0] = 1.0
    dsa_rgb[concat_mask, 1] = 0.0
    dsa_rgb[concat_mask, 2] = 0.0

    plt.figure(figsize=(6, 6))
    plt.imshow(dsa_rgb)
    plt.axis('off')
    plt.show()

    # pca_seg = PCAsegment(dsa_tensor, mask)
    # pixel_matrix, pixel_map = pca_seg.pca_matrix(K = 3)
    # T = pixel_matrix.shape[0]
    # mask = pca_seg.mask
    #
    # for t in range(T):
    #     frame_pixel_map = np.zeros(mask.shape)
    #     frame_pixel_map[mask] = pixel_matrix[t]
    #
    #     plt.imshow(frame_pixel_map, cmap='hot')
    #     plt.title(f"Pixel Contribution - Frame {t}")
    #     plt.colorbar()
    #     plt.axis('off')
    #     plt.show()
    #     time.sleep(0.1)
    #
    #
    #
    # dcm = pydicom.dcmread(r"C:\Users\artdude\Documents\lookslike_493\ANY_109_1")
    # dsa = dcm.pixel_array
    # if dsa.ndim == 2:
    #     dsa = np.expand_dims(dsa, axis=0)
    # # dsa = dsa[4:11, :, :]
    #
    # mask = tifffile.imread(r"C:\Users\artdude\Documents\lookslike_493\ANY_109_1.tif").astype(bool)
    #
    # dsa_tensor = torch.tensor(dsa, dtype=torch.float32)
    #
    # pca_seg = PCAsegment(dsa_tensor, mask)
    # pixel_mat,pixel_map = pca_seg.pca_matrix(K = 3)
    # pixel_map_np = pixel_map.numpy()
    #
    #
    # plt.figure(figsize=(6, 6))
    # # pm = pixel_map_np
    # # pm = pm - pm.min()
    # # pm = pm / pm.max()
    # # plt.imshow(pm, cmap='hot')
    # plt.imshow(pixel_map_np, cmap='hot')
    # plt.colorbar(label="Contribution")
    # plt.title("Pixel Contribution Map")
    # plt.axis('off')
    # plt.show()
    # #
    # frame_stack,thresh = pca_seg.min_pixs(pixel_mat.numpy())
    # thresholds = [t.item() for t in thresh]
    #
    # plt.figure(figsize=(10, 4))
    # plt.plot(thresholds, marker='o', linewidth=2)
    # plt.title("Threshold Values Across Frames")
    # plt.xlabel("Frame Index")
    # plt.ylabel("Threshold Intensity")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # print(thresh)
    #
    # for i in range(len(frame_stack)):
    #     original = dsa[i]
    #     segmented = frame_stack[i]
    #
    #     plt.figure(figsize=(12, 5))
    #
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(original, cmap='gray')
    #     plt.title(f"Original Frame {i}")
    #     plt.axis('off')
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(segmented, cmap='gray')
    #     plt.title(f"Segmented Frame {i}")
    #     plt.axis('off')
    #
    #     plt.show()