import numpy as np
import pandas as pd
import json
from generate_raw import generate_raw
from process_data import process_data
from model_data import model_data
from burdened_class_logreg import run_regression
from plotting_funcs import plot_data

# Main LFP configure function
def config_dash(pt_name: str):
    # open trbd and ocd patient info jsons
    if pt_name[0] == 'T':
        with open('trbd_patient_info.json', 'r') as f:
            pt_info = json.load(f)
    elif pt_name[0] == 'A' or pt_name[0] == 'B':
        with open('ocd_patient_info.json', 'r') as f:
            pt_info = json.load(f)
    else:
        print('Patient not in the database!')
        return 

    # Analyze LFP data for the patient
    raw_df, pt_changes_df = generate_raw(pt_name, pt_info[pt_name])
    processed_data = process_data(pt_name, raw_df, pt_info[pt_name])
    df_w_preds = model_data(processed_data)

    # Plot data
    fig = plot_data(pt_name, pt_info[pt_name], df_w_preds)

    # Return the compiled LFP dataframe and figure object
    return df_w_preds, fig

def main():
    pt_name = 'TRBD001'
    df, fig = config_dash(pt_name)
    fig.show()

if __name__=='__main__':
    main()