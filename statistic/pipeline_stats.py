import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Z:/Users/Artin/Data/01_SPSS.csv")


column_order = [
    "PH_NR_mean",
    "TTP_NR_mean",
    "AUC_NR_mean",
    "MTT_NR_mean",
    "max_Df_NR_mean",
    "AUC_05MTT_NR_mean",
    "AUC_1MTT_NR_mean",
    "AUC_15MTT_NR_mean",
    "AUC_2MTT_NR_mean"
]

titles = ["PH", "TTP", "AUC", "MTT", "Max_Df", "AUC0.5", "AUC1", "AUC1.5", "AUC2.0"]

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
axes = axes.flatten()

for ax, col, title in zip(axes, column_order, titles):
    ax.hist(df[col].dropna(), bins=30, edgecolor='black')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Frequency", fontsize=10)

plt.tight_layout(w_pad=3, h_pad=3)
plt.show()
