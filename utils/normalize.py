from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


csv_path = Path(__file__).resolve().parent.parent / "Data" / "Inlet_Treat_Parameter.csv"
df = pd.read_csv(csv_path)
df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
numeric_cols = df.select_dtypes(include=['number']).columns


scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df.to_csv('normalized_file_Inlet_Treat_Parameter.csv', index=False)
