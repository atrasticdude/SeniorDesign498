import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(directory, file):
    csv_path = Path(__file__).resolve().parent.parent / directory / file
    df = pd.read_csv(csv_path)
    return df

# Load and clean data
data = load_data("Data", "normalized_file_Inlet_Treat_Parameter.csv")
# data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
data.drop("ID", axis=1, inplace=True)

# Display info
print(data.info())

# Plot histograms for all numeric columns
data.hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Histograms of All Numeric Columns", fontsize=16)
plt.tight_layout()
plt.show()