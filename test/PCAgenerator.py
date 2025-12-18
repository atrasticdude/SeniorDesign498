from pathlib import Path
import numpy as np
import pydicom
import tifffile
import torch
from src.Disturbed.PCAsegment import PCAsegment
from utils.helperfunction import get_bases_id

raw_path = Path(r"Z:\Users\Artin\coiled\raw_file")
mask_path = Path(r"Z:\Users\Artin\coiled\aneurysms")
save_path = Path(r"Z:\Users\Artin\coiled\PCA_masks")
save_path.mkdir(parents=True, exist_ok=True)


any_direct = {p.name: p for p in raw_path.iterdir() if p.is_file()}
mask_direct = {p.name: p for p in mask_path.iterdir() if p.is_file()}

base_visited = set()

for dcm_name, dcm_path in any_direct.items():
    base = get_bases_id(dcm_name)
    if base in base_visited:
        continue
    base_visited.add(base)

    possible_names = (
        f"{base}_1",
        f"{base}_View1_1",
        f"{base}_View2_1",
    )

    for name in possible_names:
        if name not in any_direct:
            continue
        mask_file_name = f"{name}.tif"
        if mask_file_name not in mask_direct:
            continue

        dcm = pydicom.dcmread(any_direct[name])
        dsa = dcm.pixel_array
        if dsa.ndim == 2:
            dsa = np.expand_dims(dsa, axis=0)

        mask = tifffile.imread(mask_direct[mask_file_name]).astype(bool)

        if mask.shape[-2:] != dsa.shape[-2:]:
            print(f"Shape mismatch: {name}, skipping")
            continue

        dsa_tensor = torch.tensor(dsa, dtype=torch.float32)
        if dsa_tensor.max() > 0:
            dsa_tensor /= dsa_tensor.max()

        pca_seg = PCAsegment(dsa_tensor, mask)
        concat = pca_seg.connected_components(K=3)

        concat_np = concat.cpu().numpy() if torch.is_tensor(concat) else concat
        concat_to_save = (concat_np > 0).astype(np.uint8) * 255

        save_file = save_path / f"{name}.tif"
        tifffile.imwrite(save_file, concat_to_save)
        print(f"Saved: {save_file}")



#####Crop out part

import os
import tifffile
import numpy as np



def find_largest(mask_dir, pca_dir):
    largest_area = 0
    max_pad = 0

    mask_files = os.listdir(mask_dir)
    pca_files = os.listdir(pca_dir)

    for f in pca_files:
        if f not in mask_files:
            continue

        mask_file = os.path.join(mask_dir, f)
        mask = tifffile.imread(mask_file).astype(bool)
        area = np.sum(mask)

        if area > largest_area:
            rows, cols = np.where(mask)
            if len(rows) == 0 or len(cols) == 0:
                continue
            h = rows.max() - rows.min() + 1
            w = cols.max() - cols.min() + 1
            max_pad = max(h, w)
            largest_area = area

    return max_pad

import os
import numpy as np
import tifffile

def crop_pad_center(img, pad, mask=None):
    """
    Crop to pad x pad, centered on mask if provided.
    Pads with zeros if image is smaller than pad.
    """
    h, w = img.shape

    if mask is not None and np.any(mask):
        rows, cols = np.where(mask)
        cy = (rows.min() + rows.max()) // 2
        cx = (cols.min() + cols.max()) // 2
    else:
        cy, cx = h // 2, w // 2

    y0 = max(0, cy - pad // 2)
    x0 = max(0, cx - pad // 2)
    y1 = min(h, y0 + pad)
    x1 = min(w, x0 + pad)

    cropped = img[y0:y1, x0:x1]

    # Pad if needed
    pad_h = pad - cropped.shape[0]
    pad_w = pad - cropped.shape[1]

    if pad_h > 0 or pad_w > 0:
        cropped = np.pad(
            cropped,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0
        )

    return cropped


def apply_padding_to_masks(mask_dir, pca_dir, save_dir, pad):
    os.makedirs(save_dir, exist_ok=True)

    for f in os.listdir(mask_dir):
        if not f.endswith(".tif"):
            continue

        mask_path = os.path.join(mask_dir, f)
        pca_path = os.path.join(pca_dir, f)

        if not os.path.exists(pca_path):
            continue

        mask = tifffile.imread(mask_path)
        pca = tifffile.imread(pca_path)

        mask_bool = mask.astype(bool)

        mask_out = crop_pad_center(mask, pad, mask_bool)
        pca_out = crop_pad_center(pca, pad, mask_bool)

        base = os.path.splitext(f)[0]
        tifffile.imwrite(os.path.join(save_dir, f"{base}.tif"), mask_out)
        tifffile.imwrite(os.path.join(save_dir, f"{base}_PCA.tif"), pca_out)




mask_dir = r"Z:\Users\Artin\coiled\mask"
pca_dir = r"Z:\Users\Artin\coiled\PCA_masks"
save_dir = r"Z:\Users\Artin\coiled\cropped_PCAandmasks"

pad = find_largest(mask_dir, pca_dir)
print("Using pad size:", pad)

apply_padding_to_masks(mask_dir, pca_dir, save_dir, pad)


