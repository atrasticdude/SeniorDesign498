import time

import cv2
import pydicom
import tifffile
import torch
import numpy as np
from mpmath import eps


class PCAsegment(object):
    def __init__(self, dsa, mask, num_compoents = 3, stand_frame = 3):
        self.dsa = dsa
        self.mask = mask.astype(bool)
        self.components = num_compoents
        self.time_frames = stand_frame
        self.bin = self.connected_components()
        self.coords = self.regions_coords(self.bin)
    #
    # def pca_matrix(self, K = 5):
    #     mask = self.mask
    #     dsa = self.dsa
    #     T = dsa.shape[0]
    #     N = mask.sum().item()
    #     X = dsa[:, mask]
    #     mean_pixel = X.mean(dim=0, keepdim=True)
    #     Xc = X - mean_pixel
    #     U, S, V = torch.pca_lowrank(Xc, q= K)
    #     scores = U * S
    #     components = V.T
    #
    #     pixel_contribution = torch.sqrt((components ** 2).sum(dim=0))
    #     pixel_contribution_map = torch.zeros(mask.shape)
    #     pixel_contribution_map[mask] = pixel_contribution
    #
    #     # pixel_contribution_matrix = scores @ components
    #
    #     # pixel_contribution_matrix += mean_pixel
    #     pca_only = scores @ components  # [T, N]
    #
    #     binary_contribution = (pca_only.abs() > eps).int()  # [T, N]
    #
    #     pixel_contribution_matrix = pca_only + mean_pixel
    #
    #     return pixel_contribution_matrix, pixel_contribution_map, binary_contribution
    def pca_matrix(self, K=5, eps=1e-6):
        mask = self.mask
        dsa = self.dsa
        T = dsa.shape[0]

        X = dsa[:, mask]

        mean_pixel = X.mean(dim=0, keepdim=True)
        Xc = X - mean_pixel

        var_mask = (Xc.var(dim=0) > eps)

        U, S, V = torch.pca_lowrank(Xc, q=K)
        scores = U * S
        components = V.T

        pca_only = scores @ components
        pca_only = pca_only * var_mask

        pixel_contribution_matrix = pca_only + mean_pixel * var_mask

        pixel_contribution = torch.sqrt((components ** 2).sum(dim=0))
        pixel_contribution_map = torch.zeros(mask.shape, device=X.device)
        pixel_contribution_map[mask] = pixel_contribution * var_mask

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

    def connected_components(self, threshold = 0.85):
        mask = torch.tensor(self.mask, dtype=torch.bool)
        k_arr = torch.arange(1, self.components+1)
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
    def regions_coords(self,bin):
        top = self.components
        bin = bin.to(torch.uint8)
        bin_np = bin.cpu().numpy()
        bin_unit8 = (bin_np *255).astype('uint8')
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_unit8)
        num_fg = num_labels - 1
        if num_fg == 0:
            return np.empty((0, 2))
        top = min(top, num_fg)
        areas = stats[1:, cv2.CC_STAT_AREA]
        centroids_fg = centroids[1:]
        idx = np.argsort(areas)[::-1][:top]
        coords = centroids_fg[idx]
        return coords

    def avg_pc_mask(self):
        T, H, W = self.dsa.shape
        pca_frames = []

        for k in range(1, self.components + 1):
            mat, _ = self.pca_matrix(K=k)  # [T, N]

            full = torch.zeros(
                (T, H, W),
                device=mat.device,
                dtype=mat.dtype
            )
            full[:, self.mask] = mat
            pca_frames.append(full)
        pca_stack = torch.stack(pca_frames, dim=0)  # [K, T, H, W]
        pca_stack = pca_stack.transpose(0, 1)#[T,K,H,W]
        # idx = (pca_stack != 0).float().argmax(dim=1)
        mask = pca_stack != 0
        has_nonzero = mask.any(dim=1)  # [T,H,W]
        idx = mask.float().argmin(dim=1)  # [T,H,W]
        T_idx = torch.arange(T, device=pca_stack.device)[:, None, None]  # [T,1,1]
        H_idx = torch.arange(H, device=pca_stack.device)[None, :, None]  # [1,H,1]
        W_idx = torch.arange(W, device=pca_stack.device)[None, None, :]  # [1,1,W]
        values = pca_stack[idx, T_idx, H_idx, W_idx]
        fused = torch.where(has_nonzero, values, torch.zeros_like(values))# [T,H,W]

        return fused


    # def avg_pc_mask(self):
    #     T, H, W = self.dsa.shape
    #     pca_frames = []
    #
    #     for k in range(1, self.components + 1):
    #         mat, _ = self.pca_matrix(K=k)  # [T, N]
    #
    #         full = torch.zeros(
    #             (T, H, W),
    #             device=mat.device,
    #             dtype=mat.dtype
    #         )
    #
    #         full[:, self.mask] = mat
    #         pca_frames.append(full)
    #
    #     pca_stack = torch.stack(pca_frames, dim=0)  # [K, T, H, W]
    #     dominant_idx = pca_stack.abs().argmin(dim=0)  # [T, H, W]
    #
    #     fused = torch.gather(
    #         pca_stack,
    #         dim=0,
    #         index=dominant_idx.unsqueeze(0)
    #     ).squeeze(0)  # [T, H, W]
    #
    #     return fused

    def contribution_frame(self):
        # self.dsa: [T, H, W]
        T, H, W = self.dsa.shape
        final_list = []

        for k in range(1, self.components + 1):
            mat, _ = self.pca_matrix(K=k)  # [T, N_mask_pixels]
            mat = torch.tensor(mat, dtype=torch.float32)

            # Create full-frame tensor
            full_frame = torch.zeros(T, H * W, dtype=torch.float32)
            mask_flat = self.mask.flatten()
            full_frame[:, mask_flat] = mat  # fill masked pixels
            full_frame = full_frame.unsqueeze(1)  # [T, 1, H*W]

            final_list.append(full_frame)

        final_mat = torch.cat(final_list, dim=1)  # [T, components, H*W]
        final_mat = final_mat.reshape(T, self.components, H, W)  # now safe

        # final_mat = final_mat.mean(dim=1)  # [T, H, W]
        final_mat = (final_mat > 0).any(dim=1).float()  # [T, H, W]

        if T >= self.time_frames:
            chunks = torch.chunk(final_mat, self.time_frames, dim=0)
            final_mat = torch.stack([c.mean(dim=0) for c in chunks], dim=0)  # [time_frames, H, W]
        else:
            pad = final_mat[-1:].repeat(self.time_frames - T, 1, 1)
            final_mat = torch.cat([final_mat, pad], dim=0)  # [time_frames, H, W]

        return final_mat

    def get_coords(self):
        if self.coords.size == 0:
            return np.empty((0, 2), dtype=int)
        return np.round(self.coords).astype(int)


if __name__ == '__main__':
    dcm = pydicom.dcmread(r"Z:\Users\Artin\coiled\raw_file\ANY_109_1")
    dsa = dcm.pixel_array
    # dsa = dsa[4:11,:,:]
    print(dsa.shape[2])

    if dsa.ndim == 2:
        dsa = np.expand_dims(dsa, axis=0)

    mask = tifffile.imread(r"Z:\Users\Artin\coiled\mask\ANY_109_1.tif").astype(bool)

    dsa_tensor = torch.tensor(dsa, dtype=torch.float32)

    # import torch
    import matplotlib.pyplot as plt
    pca_seg = PCAsegment(dsa_tensor, mask,3, stand_frame= 10)
    # final_mat = pca_seg.avg_pc_mask()  # [time_frames, H, W]
    #
    # # Visualize each averaged frame
    # for t in range(final_mat.shape[0]):
    #     frame_map = final_mat[t]
    #
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(frame_map.numpy(), cmap='hot')
    #     plt.title(f"Averaged Frame {t + 1}")
    #     plt.colorbar()
    #     plt.axis('off')
    #     plt.show()
    #     time.sleep(0.1)

    concat = pca_seg.connected_components()

    plt.figure(figsize=(6, 6))
    plt.imshow(concat.numpy(), cmap='hot')
    plt.axis('off')
    plt.show()
    coords = pca_seg.get_coords()
    print(coords)
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

    import numpy as np
    import matplotlib.pyplot as plt

    dsa_frame = dsa[6]
    dsa_norm = (dsa_frame - dsa_frame.min()) / (dsa_frame.max() - dsa_frame.min())
    dsa_rgb = np.stack([dsa_norm] * 3, axis=-1)

    concat_mask = concat.numpy().astype(bool)

    ys, xs = np.where(concat_mask)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    pad = 10
    ymin = max(ymin - pad, 0)
    ymax = min(ymax + pad, dsa_rgb.shape[0])
    xmin = max(xmin - pad, 0)
    xmax = min(xmax + pad, dsa_rgb.shape[1])

    dsa_crop = dsa_rgb[ymin:ymax, xmin:xmax].copy()
    mask_crop = concat_mask[ymin:ymax, xmin:xmax]

    dsa_crop[mask_crop, 0] = 1.0
    dsa_crop[mask_crop, 1] = 0.0
    dsa_crop[mask_crop, 2] = 0.0

    plt.figure(figsize=(6, 6))
    plt.imshow(dsa_crop)
    plt.axis('off')
    plt.show()

    pca_seg = PCAsegment(dsa_tensor, mask)
    pixel_matrix, pixel_map = pca_seg.pca_matrix(K = 3)
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



    # dcm = pydicom.dcmread(r"C:\Users\artdude\Documents\lookslike_493\ANY_125_1")
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
    # pixel_mat,pixel_map = pca_seg.pca_matrix(K = 1)
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