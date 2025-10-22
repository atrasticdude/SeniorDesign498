
from pathlib import Path
import pandas as pd
from utils.helperfunction import get_bases_id

csv_path = Path(__file__).resolve().parent.parent / "Data" / "coil_study_occlusion_key.csv"
labels = pd.read_csv(csv_path)
cases = labels.columns.tolist()


def create_case_label_pairs(input_dict):
    if not isinstance(input_dict, dict):
        raise TypeError("Input must be a dictionary")
    data = []
    csv_path = Path(__file__).resolve().parent.parent / "Data" / "coil_study_occlusion_key.csv"
    labels = pd.read_csv(csv_path)
    arr = labels["Case Number"].values
    boolean = False
    for index,col in enumerate(arr[:-1]):
        if col in input_dict:
            if boolean:
                data.append((input_dict[col][2], labels["Outcome"].values[index]))
                boolean = False
            else:
                data.append((input_dict[col][1], labels["Outcome"].values[index]))
                if(get_bases_id(arr[index]) == get_bases_id(arr[index + 1])):
                    boolean = True
    return data


# from pathlib import Path
# import pandas as pd
#
# def create_case_label_pairs(input_dict):
#     if not isinstance(input_dict, dict):
#         raise TypeError("Input must be a dictionary")
#
#     data = []
#     csv_path = Path(__file__).resolve().parent.parent / "Data" / "coil_study_occlusion_key.csv"
#     labels = pd.read_csv(csv_path)
#
#     case_numbers = labels["Case Number"].values
#     outcomes = labels["Outcome"].values
#     use_second_inlet = False
#
#     for i, case in enumerate(case_numbers[:-1]):  # iterate till second to last
#         if case in input_dict:
#             # Safety: make sure the list has enough elements
#             inlet_list = input_dict[case]
#             if use_second_inlet and len(inlet_list) > 2:
#                 data.append((inlet_list[2], outcomes[i]))
#                 use_second_inlet = False
#             elif len(inlet_list) > 1:
#                 data.append((inlet_list[1], outcomes[i]))
#                 # check if next case has the same base id
#                 if get_bases_id(case) == get_bases_id(case_numbers[i + 1]):
#                     use_second_inlet = True
#
#     return data







