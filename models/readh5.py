
import h5py
file_path = r"C:\Users\artdude\PycharmProjects\SeniorDesign498\models\best_weights_fold1.h5"
with h5py.File(file_path, "r") as f:
    dense_kernel = f["model_weights/dense/sequential/dense/kernel"][:]
    print(dense_kernel)

