import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from access.GetAPI import getAPI

# --- Paths ---
img_path = r"Z:\Users\Artin\coiled\raw_file\ANY_136_1"
mask_path = r"Z:\Users\Artin\coiled\aneurysms\ANY_136_1.tif"
inlet_path = r"Z:\Users\Artin\coiled\inlets\ANY_136_1_inl.tif"

# Extract case number from filename
case_number = "ANY_116_0"

# --- Load DICOM ---
ds = pydicom.dcmread(img_path)
arr = ds.pixel_array.astype(np.float32)

if arr.ndim == 2:
    arr = arr[None, :, :]  # make 3D if needed

# --- Load mask and inlet ---
# mask_img = Image.open(mask_path).convert("L")
# mask = np.array(mask_img) > 0

inlet_img = Image.open(inlet_path).convert("L")
inlet = np.array(inlet_img) > 0

# --- Initialize getAPI ---
api_obj = getAPI(dsa=arr, inlet=inlet, dsa_temp=ds, frac=0.1, show_mask_stats=False)

# --- Get inlet TDC and time ---
tdc_inlet = api_obj.inlet_tdc_inlet
time_vector = api_obj._x_inter

# --- Get inlet API parameters ---
inlet_params = api_obj.get_inlet_API()
param_keys = ["PH", "TTP", "AUC", "MTT", "Max_Df", "BAT"]
param_values = [inlet_params.get(k, np.nan) for k in param_keys]
table_data = [[k, f"{v:.3f}" if isinstance(v, (float, int)) else str(v)] for k, v in zip(param_keys, param_values)]

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_vector, tdc_inlet, label="Inlet TDC", color="blue", linewidth=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Concentration")
ax.set_title(f"Inlet Time Density Curve (TDC) for {case_number}")
ax.grid(True)
ax.legend()

# --- Add API parameter table below the plot ---
table = plt.table(cellText=table_data,
                  colLabels=["Parameter", "Value"],
                  cellLoc='center',
                  colLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.45, 1.0, 0.3])

plt.subplots_adjust(left=0.1, bottom=0.3)
plt.show()
