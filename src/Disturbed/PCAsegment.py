import pydicom
import tifffile
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from utils.disturbedhelper import global_thresholding


class PCAsegment(object):
    def __init__(self, dsa, mask):
        self.dsa = dsa
        self.mask = mask.astype(bool)

    def pca_matrix(self, var_threshold=0.01):
        mask = self.mask
        dsa = self.dsa
        T = dsa.shape[0]
        N = mask.sum().item()
        X = dsa[:, mask].float()
        mean_pixel = X.mean(dim=0, keepdim=True)
        Xc = X - mean_pixel
        U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
        scores = U @ torch.diag(S)
        signal_idx = (scores.var(dim=0) > var_threshold)
        signal_scores = scores[:, signal_idx]
        signal_components = Vt[signal_idx, :]
        pixel_contribution = torch.sqrt((signal_components ** 2).sum(dim=0))
        pixel_contribution_map = torch.zeros(mask.shape)
        pixel_contribution_map[mask] = pixel_contribution
        pixel_contribution_matrix = signal_scores @ signal_components
        pixel_contribution_matrix += mean_pixel
        return pixel_contribution_matrix, pixel_contribution_map

    def connected_components(self, pc_matrix):
        # dsa = self.dsa.clone().numpy()
        # y, x = np.where(self.mask)
        #
        # time = pc_matrix.shape[0]
        # min_value_pixel = np.argmin(pc_matrix, axis=1)
        #
        # frame_stack = []
        #
        # for i in range(time):
        #     crop_out = dsa[i][y, x]
        #     threshold = crop_out[min_value_pixel[i]]
        #     _, segmented = global_thresholding(crop_out, threshold)
        #     stack = np.zeros(self.mask.shape, dtype=np.uint8)
        #     stack[y, x] = segmented
        #     frame_stack.append(stack)
        #
        # return frame_stack
        dsa = self.dsa  #
        mask = torch.tensor(self.mask, dtype=torch.bool)
        y, x = torch.where(mask)
        time = pc_matrix.shape[0]
        min_value_pixel = torch.argmin(torch.tensor(pc_matrix), dim=1)

        frame_stack = []
        thresholds = []

        for i in range(time):
            crop_out = dsa[i][y, x]
            threshold = crop_out[min_value_pixel[i]]
            segmented = (crop_out <= threshold).to(torch.uint8)
            stack = torch.zeros_like(mask, dtype=torch.uint8)
            stack[y, x] = segmented
            frame_stack.append(stack)
            thresholds.append(threshold)
        return frame_stack, thresholds

if __name__ == '__main__':
    # dcm = pydicom.dcmread(r"C:\Users\artdude\Documents\lookslike_493\ANY_109_1")
    # dsa = dcm.pixel_array
    #
    # if dsa.ndim == 2:
    #     dsa = np.expand_dims(dsa, axis=0)
    #
    # mask = tifffile.imread(r"C:\Users\artdude\Documents\lookslike_493\ANY_109_1.tif").astype(bool)
    #
    # dsa_tensor = torch.tensor(dsa, dtype=torch.float32)
    #
    # pca_seg = PCAsegment(dsa_tensor, mask)
    # pixel_matrix, pixel_map = pca_seg.pca_matrix(var_threshold=0.01)
    #
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



    dcm = pydicom.dcmread(r"C:\Users\artdude\Documents\lookslike_493\ANY_109_1")
    dsa = dcm.pixel_array
    if dsa.ndim == 2:
        dsa = np.expand_dims(dsa, axis=0)


    mask = tifffile.imread(r"C:\Users\artdude\Documents\lookslike_493\ANY_109_1.tif").astype(bool)

    dsa_tensor = torch.tensor(dsa, dtype=torch.float32)

    pca_seg = PCAsegment(dsa_tensor, mask)
    pixel_mat,pixel_map = pca_seg.pca_matrix(var_threshold=0.01)
    pixel_map_np = pixel_map.numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(pixel_map_np, cmap='hot')
    plt.colorbar(label="Contribution")
    plt.title("Pixel Contribution Map")
    plt.axis('off')
    plt.show()

    frame_stack,thresh = pca_seg.connected_components(pixel_mat.numpy())
    thresholds = [t.item() for t in thresh]

    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, marker='o', linewidth=2)
    plt.title("Threshold Values Across Frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Threshold Intensity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(thresh)

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