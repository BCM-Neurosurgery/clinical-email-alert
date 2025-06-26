import numpy as np
import pandas as pd
import json
from generate_raw import generate_raw
from process_data import process_data
from model_data import model_data
from burdened_class_logreg import run_regression
from plotting_funcs import plot_data


# Main LFP configure function
def config_dash(pt_name: str, save_path: str = None):
    """
    Analyzes LFP data for a given patient, generates a figure, and optionally saves it.

    Args:
        pt_name (str): The name of the patient.
        save_path (str, optional): The full path (including filename and extension)
                                   to save the static image. Defaults to None.
    """
    # open trbd and ocd patient info jsons
    if pt_name[0] == "T":
        # It's good practice to use a more robust path handling mechanism
        # For example, using os.path.join
        json_path = "/home/auto/CODE/trbd/TRBD-null-pipeline/lfp_analysis/trbd_patient_info.json"
    elif pt_name[0] in ["A", "B"]:
        json_path = (
            "/home/auto/CODE/trbd/TRBD-null-pipeline/lfp_analysis/ocd_patient_info.json"
        )
    else:
        print("Patient not in the database!")
        return None, None

    try:
        with open(json_path, "r") as f:
            pt_info = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return None, None

    # Analyze LFP data for the patient
    raw_df, pt_changes_df = generate_raw(pt_name, pt_info[pt_name])
    processed_data = process_data(pt_name, raw_df, pt_info[pt_name])
    df_w_preds = model_data(processed_data)

    # Plot data
    fig = plot_data(pt_name, pt_info[pt_name], df_w_preds)

    # Save the figure if a save_path is provided
    if save_path:
        try:
            # The write_image method saves the figure to the specified path.
            # The format is inferred from the file extension (.png, .jpg, .svg, .pdf).
            # You can also control dimensions and scale.
            fig.write_image(save_path, width=1280, height=720)
            print(f"Figure successfully saved to: {save_path}")
        except Exception as e:
            print(f"An error occurred while saving the figure: {e}")

    # Return the compiled LFP dataframe and figure object
    return df_w_preds, fig


def main():
    pt_name = "TRBD001"
    save_path = "/home/auto/CODE/trbd/TRBD-null-pipeline/lfp_analysis/lfp_dashboard.png"
    df, fig = config_dash(pt_name, save_path=save_path)
    fig.show()


if __name__ == "__main__":
    main()
