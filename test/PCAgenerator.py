from pathlib import Path
import numpy as np
import pydicom
import tifffile
import torch

from src.Disturbed.PCAsegment import PCAsegment
from utils.helperfunction import get_bases_id

raw_path = Path(r"Z:\Users\Artin\coiled\raw_file")
mask_path = Path(r"Z:\Users\Artin\coiled\aneurysms")

any_direct = {p.name: p for p in raw_path.iterdir() if p.is_file()}
mask_direct = {p.name for p in mask_path.iterdir() if p.is_file()}

base_visited = set()

pca_arr = []
max_mask_area = np.inf
max_area_file = ""

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
        if f"{name}.tif" not in mask_direct:
            continue

        dcm = pydicom.dcmread(any_direct[name])
        dsa = dcm.pixel_array
        if dsa.ndim == 2:
            dsa = np.expand_dims(dsa, axis=0)

        mask = tifffile.imread(mask_path / f"{name}.tif").astype(bool)
        dsa_tensor = torch.tensor(dsa, dtype=torch.float32)
        pca_seg = PCAsegment(dsa_tensor, mask)
        concat = pca_seg.connected_components(K=3)
        pca_arr.append(concat)
        area = mask.sum()
        if area > max_mask_area:
            max_mask_area = area
            mask_area_file = mask_path / f"{name}.tif"



