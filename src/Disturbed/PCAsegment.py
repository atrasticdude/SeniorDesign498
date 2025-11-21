import pydicom
import tifffile
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

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

dcm = pydicom.dcmread(r"Z:\Users\Artin\coiled\raw_file\ANY_103_1")
dsa = dcm.pixel_array

if dsa.ndim == 2:
    dsa = np.expand_dims(dsa, axis=0)

mask = tifffile.imread(r"Z:\Users\Artin\coiled\aneurysms\ANY_103_1.tif").astype(bool)

dsa_tensor = torch.tensor(dsa, dtype=torch.float32)

pca_seg = PCAsegment(dsa_tensor, mask)
pixel_matrix, pixel_map = pca_seg.pca_matrix(var_threshold=0.01)

T = pixel_matrix.shape[0]
mask = pca_seg.mask

for t in range(T):
    frame_pixel_map = np.zeros(mask.shape)
    frame_pixel_map[mask] = pixel_matrix[t]

    plt.imshow(frame_pixel_map, cmap='hot')
    plt.title(f"Pixel Contribution - Frame {t}")
    plt.colorbar()
    plt.axis('off')
    plt.show()
    time.sleep(0.1)
