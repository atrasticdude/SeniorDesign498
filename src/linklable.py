
from pathlib import Path
import pandas as pd
from utils.helperfunction import get_bases_id,sort_files_numerically
from Separation import Separate

def create_case_label_pairs(input_dict):
    if not isinstance(input_dict, dict):
        raise TypeError("Input must be a dictionary")

    data = []
    csv_path = Path(__file__).resolve().parent.parent / "Data" / "coil_study_occlusion_key.csv"
    labels = pd.read_csv(csv_path)

    case_numbers = labels[" Case Number"].values
    outcomes = labels["Outcome"].values
    use_second_inlet = False

    for i, case in enumerate(case_numbers[:-1]):
        if get_bases_id(case) in input_dict:
            inlet_list = input_dict[get_bases_id(case)]
            if len(inlet_list) > 1:
                inlet_list = sort_files_numerically(inlet_list)
                if use_second_inlet:
                    data.append((inlet_list[1], outcomes[i]))
                    use_second_inlet = False
                else:
                    data.append((inlet_list[0], outcomes[i]))
            elif len(inlet_list) > 0:
                data.append((inlet_list[0], outcomes[i]))
            if get_bases_id(case) == get_bases_id(case_numbers[i + 1]):
                use_second_inlet = True

    last_case = case_numbers[-1]
    if get_bases_id(last_case) in input_dict:
        inlet_list = input_dict[get_bases_id(last_case)]
        if use_second_inlet:
            inlet_list = sort_files_numerically(inlet_list)
            data.append((inlet_list[1], outcomes[-1]))
        else:
            data.append((inlet_list[0], outcomes[-1]))
    return data






