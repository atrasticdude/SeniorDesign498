from pathlib import Path
import pandas as pd



def load_data(directory, file):
    csv_path = Path(__file__).resolve().parent.parent / directory / file
    df = pd.read_csv(csv_path)
    return df


inlet_df = load_data("Data", "Inlet_Treat_Parameter.csv")
label_df = load_data("Data", "Coil_study_occlusion_key.csv")


inlet_df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
inlet_df = inlet_df.merge(
    label_df[['Case Number', 'Outcome']],
    left_on='ID',
    right_on='Case Number',
    how='left'
)


inlet_df = inlet_df.dropna(subset=["Outcome"])
inlet_df.to_csv("inlet_df_clean.csv", index=False)





