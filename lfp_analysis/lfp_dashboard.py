import json
from lfp_analysis.generate_raw import generate_raw
from lfp_analysis.process_data import process_data
from lfp_analysis.model_data import model_data
from lfp_analysis.plotting_funcs import plot_data

json_path = "/home/auto/CODE/trbd/TRBD-null-pipeline/lfp_analysis/lfp_config.json"


# Main LFP configure function
def config_dash(pt_name: str, save_path: str = None):
    """
    Analyzes LFP data for a given patient, generates a figure, and optionally saves it.

    Args:
        pt_name (str): The name of the patient.
        save_path (str, optional): The full path (including filename and extension)
                                   to save the static image. Defaults to None.
    """
    try:
        with open(json_path, "r") as f:
            pt_info = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return None, None

    try:
        raw_df, pt_changes_df = generate_raw(pt_name, pt_info[pt_name])
    except TypeError as e:
        print(f"Error during data generation for patient {pt_name}: No Data - {e}.")
        return None, None

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
    pt_name = "TRBD002"
    save_path = f"/home/auto/CODE/trbd/TRBD-null-pipeline/lfp_analysis/lfp_dashboard_{pt_name}.png"
    df, fig = config_dash(pt_name, save_path=save_path)


if __name__ == "__main__":
    main()
