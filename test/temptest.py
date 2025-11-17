from pathlib import Path
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt


def load_data(directory, file):
    csv_path = Path(__file__).resolve().parent.parent / directory / file
    df = pd.read_csv(csv_path)
    return df

inclusion_df= load_data("Data", "cnn_inclusion.csv")
label_df = load_data("Data", "Coil_study_occlusion_key.csv")

cases_inclusion = inclusion_df["Case Number"].values
cases_label = label_df["Case Number"].values

mask = inclusion_df["Outcome"] == 1
true_cases = cases_inclusion[mask]

label_map = label_df.set_index("Case Number")["Outcome"]
true_labels = label_map.loc[true_cases].values

label_counts = label_df["Outcome"].value_counts().sort_index()
print("Label DF Outcome Counts (0 or 1):")
print(label_counts)
print("\n")

inclusion_counts = inclusion_df["Outcome"].value_counts().sort_index()
print("Inclusion DF Outcome Counts (1 or 2):")
print(inclusion_counts)
print("\n")

true_case_ids = inclusion_df.loc[inclusion_df["Outcome"] == 1, "Case Number"]

true_labels = label_df[label_df["Case Number"].isin(true_case_ids)]["Outcome"]

true_label_counts = true_labels.value_counts().sort_index()

print("TRUE LABEL Counts (only for cases where inclusion_df Outcome = 1):")
print(true_label_counts)
