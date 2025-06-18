import os
import pandas as pd
from abc import ABC, abstractmethod
from importlib import import_module
from typing import List, Dict, Any
from trbdv0.utils import get_yesterdays_date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class SurveyProcessor(ABC):
    """Base processor for Qualtrics survey CSVs.

    Subclasses **must** set:
      survey_id (str): the Qualtrics survey identifier, e.g. "PHQ-8".
    columns_to_extract (List[str]): which CSV columns to pull.
    """

    #: e.g. "PHQ-8"; override in each subclass
    survey_id: str
    #: list of column names in the CSV this processor will extract
    columns_to_extract: List[str] = []

    def __init__(self, patient_id: str, survey_folder: str, patient_out_dir: str):
        """
        Args:
            patient_id (str): ID of the patient.
                e.g. "TRBD001"
            survey_folder (str): Path to folder that holds the surveys.
                e.g. "/mnt/datalake/data/TRBD-53761/TRBD001/qualtrics/ASRM/"
            patient_out_dir (str): Directory where outputs will be written.
        """
        self.patient_id = patient_id
        self.survey_folder = survey_folder
        self.patient_out_dir = patient_out_dir

    def get_available_years(self) -> List[str]:
        years = [
            name
            for name in os.listdir(self.survey_folder)
            if os.path.isdir(os.path.join(self.survey_folder, name)) and name.isdigit()
        ]
        return sorted(years)

    def get_survey_files_by_year(self) -> Dict[str, List[str]]:
        """
        Return a mapping from each available year to the list of
        Qualtrics CSV file paths found under that year's subfolder.

        Assumes under self.survey_folder you have year-named dirs
        (e.g. "2025") containing CSVs like "ASRM_2025-04-16_....csv".

        Returns:
            Dict[str, List[str]]: {
                "2025": ["/abs/path/.../2025/file1.csv", "..."],
                "2026": [...],
                ...
            }
        """
        files_by_year: Dict[str, List[str]] = {}
        for year in self.get_available_years():
            year_dir = os.path.join(self.survey_folder, year)
            csv_files = [
                os.path.join(year_dir, fn)
                for fn in os.listdir(year_dir)
                if fn.lower().endswith(".csv")
            ]
            files_by_year[year] = sorted(csv_files)
        return files_by_year

    def parse_csv(self, csv_path: str) -> Dict[str, Any]:
        """
        Load and parse a single survey CSV, extracting values from row 2
        for each column in `columns_to_extract`.

        Args:
            csv_path (str): Absolute path to the survey CSV file.

        Returns:
            Dict[str, Any]: Mapping from each column in `columns_to_extract`
                to its value (from row 2), type may vary.
        """
        df = pd.read_csv(csv_path)

        # Check for missing columns
        missing = set(self.columns_to_extract) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")

        # Ensure at least two rows
        if len(df) < 2:
            raise ValueError(f"Expected at least 2 rows in {csv_path}")

        # Extract values
        result: Dict[str, Any] = {}
        for col in self.columns_to_extract:
            raw = df.iloc[1][col]
            result[col] = raw
        return result

    def get_survey_filled_date(self, csv_path: str) -> str:
        """
        Extract the completion timestamp ('EndDate') from a survey CSV.

        Args:
            csv_path (str): Path to the survey CSV file.

        Returns:
            str: Timestamp string in 'YYYY-MM-DD HH:MM:SS' format.
        """
        df = pd.read_csv(csv_path)

        # Validate presence of EndDate column
        if "EndDate" not in df.columns:
            raise ValueError(f"'EndDate' column missing in {csv_path}")
        # Validate row count
        if len(df) < 2:
            raise ValueError(f"Expected at least 2 rows in {csv_path}")

        raw = df.iloc[1]["EndDate"]
        try:
            ts = pd.to_datetime(raw)
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            raise ValueError(f"Invalid EndDate value in {csv_path}: {raw!r}")

    def get_most_recent_survey_file(self) -> str:
        """
        Traverse all survey CSVs and return the path of the CSV with the latest 'EndDate'.

        Returns:
            str: Absolute path to the CSV file with the most recent completion timestamp.
        """
        # Gather all CSV files
        all_files = []
        for files in self.get_survey_files_by_year().values():
            all_files.extend(files)

        if not all_files:
            raise ValueError(f"No survey CSV files found in {self.survey_folder}")

        # Identify the most recent
        latest_file = None
        latest_ts = None
        for csv_path in all_files:
            date_str = self.get_survey_filled_date(csv_path)
            ts = pd.to_datetime(date_str)
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
                latest_file = csv_path

        if latest_file is None:
            raise ValueError("Unable to determine most recent survey file.")
        return latest_file

    @abstractmethod
    def get_warnings(self, csv_path: str) -> Dict[str, bool]:
        """
        Evaluate warning flags for a single survey CSV.

        Returns a mapping from warning name to True/False.
        """
        raise NotImplementedError("Subclasses must implement get_warnings")

    def get_latest_survey_results(self) -> Dict[str, Any]:
        """
        Get the most recent survey CSV, extract its values, and include yesterday's warnings.

        Returns a dict containing:
          - 'path': CSV file path
          - each column in `columns_to_extract`: extracted raw value
          - 'latest_warnings': latest warning flags
        """
        latest_path = self.get_most_recent_survey_file()
        values = self.parse_csv(latest_path)
        latest_flags = self.get_latest_warnings()
        return {"path": latest_path, **values, "latest_warnings": latest_flags}

    def get_latest_warnings(self) -> Dict[str, bool]:
        """
        Return warning flags for the most recent survey CSV.

        Returns empty dict if no survey files exist.
        """
        try:
            latest = self.get_most_recent_survey_file()
        except ValueError:
            return {}
        return self.get_warnings(latest)


class PHQ8Processor(SurveyProcessor):
    survey_id = "PHQ-8"
    columns_to_extract = ["SC0", "EndDate"]

    def get_warnings(self, csv_path: str) -> Dict[str, bool]:
        val = float(self.parse_csv(csv_path)["SC0"])
        return {"Depression": val > 10}


class ISSProcessor(SurveyProcessor):
    """Processor for the ISS (Inventory of Suicide Symptoms) survey."""

    survey_id = "ISS"
    columns_to_extract = ["SC1", "SC2", "SC3", "SC4", "EndDate"]

    def get_warnings(self, csv_path: str) -> Dict[str, bool]:
        vals = self.parse_csv(csv_path)
        activation = float(vals["SC1"])
        well_being = float(vals["SC2"])
        return {
            "(Hypo)Mania": activation >= 155 and well_being >= 125,
            "Mixed State": activation >= 155 and well_being < 125,
            "Euthymia": activation < 155 and well_being >= 125,
            "Depression": activation < 155 and well_being < 125,
        }

    def plot_historical_scores(
        self,
        output_filename: str = "ISS_historical_plot.png",
        activation_cutoff: float = 155.0,
        wellbeing_cutoff: float = 125.0,
    ) -> str:
        """
        Generates and saves a plot of historical ISS scores over time.

        The plot shows Activation vs. Well-Being scores, divided into four
        quadrants. The chronological order of surveys is indicated by a
        color gradient (older surveys are darker, newer are brighter) and
        connecting lines, showing the patient's trajectory.

        Args:
            output_filename (str): The name for the output plot file.
            activation_cutoff (float): The threshold for the Activation score.
            wellbeing_cutoff (float): The threshold for the Well-Being score.

        Returns:
            str: The absolute path to the saved plot image.
                 Returns an empty string if no data was found to plot.
        """
        all_files = [
            file
            for files_in_year in self.get_survey_files_by_year().values()
            for file in files_in_year
        ]

        if not all_files:
            print(f"No ISS survey files found for patient {self.patient_id}.")
            return ""

        # 1. Gather and parse all survey data
        survey_data = []
        for csv_path in all_files:
            try:
                scores = self.parse_csv(csv_path)
                date_str = self.get_survey_filled_date(csv_path)
                survey_data.append(
                    {
                        "date": pd.to_datetime(date_str),
                        "activation": float(scores["SC1"]),
                        "well_being": float(scores["SC2"]),
                    }
                )
            except (ValueError, KeyError) as e:
                print(f"Skipping file {os.path.basename(csv_path)} due to error: {e}")
                continue

        if not survey_data:
            print(
                f"No valid ISS score data could be extracted for patient {self.patient_id}."
            )
            return ""

        # Create and sort dataframe
        df = pd.DataFrame(survey_data).sort_values(by="date").reset_index()

        # 2. Set up the plot
        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

        # Determine plot bounds
        x_min, x_max = df.activation.min() - 10, df.activation.max() + 10
        y_min, y_max = df.well_being.min() - 10, df.well_being.max() + 10

        # 3. Draw and label the quadrants
        ax.axvline(activation_cutoff, color="grey", linestyle="--", lw=1.5)
        ax.axhline(wellbeing_cutoff, color="grey", linestyle="--", lw=1.5)

        # Quadrant labels
        text_props = dict(
            ha="center", va="center", fontsize=14, fontweight="bold", color="white"
        )
        bg_props = dict(boxstyle="round,pad=0.5", fc="gray", ec="none", alpha=0.6)
        ax.text(
            x_min + (activation_cutoff - x_min) / 2,
            y_min + (wellbeing_cutoff - y_min) / 2,
            "Depression",
            **text_props,
            bbox=bg_props,
        )
        ax.text(
            x_min + (activation_cutoff - x_min) / 2,
            wellbeing_cutoff + (y_max - wellbeing_cutoff) / 2,
            "Euthymia",
            **text_props,
            bbox=bg_props,
        )
        ax.text(
            activation_cutoff + (x_max - activation_cutoff) / 2,
            y_min + (wellbeing_cutoff - y_min) / 2,
            "Mixed State",
            **text_props,
            bbox=bg_props,
        )
        ax.text(
            activation_cutoff + (x_max - activation_cutoff) / 2,
            wellbeing_cutoff + (y_max - wellbeing_cutoff) / 2,
            "(Hypo)Mania",
            **text_props,
            bbox=bg_props,
        )

        # 4. Plot the data with chronological visualization
        # Use numeric dates for color mapping
        numeric_dates = mdates.date2num(df["date"])

        # Plot scatter points with a color gradient for time
        scatter = ax.scatter(
            df["activation"],
            df["well_being"],
            c=numeric_dates,
            cmap="viridis",  # 'viridis' is a good choice for visibility
            s=100,  # Size of the markers
            edgecolors="black",
            zorder=2,
        )

        # 5. Finalize plot aesthetics
        ax.set_title(
            f"ISS Score Trajectory for Patient: {self.patient_id}", fontsize=16, pad=20
        )
        ax.set_xlabel("Activation Score (SC1)", fontsize=12)
        ax.set_ylabel("Well-Being Score (SC2)", fontsize=12)
        ax.grid(True, which="both", linestyle=":", linewidth="0.5", color="gray")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Add a colorbar to show the date mapping
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Survey Date", fontsize=12)
        # Format colorbar ticks as dates
        tick_locs = cbar.get_ticks()
        cbar.ax.set_yticklabels(
            [mdates.num2date(loc).strftime("%Y-%m-%d") for loc in tick_locs]
        )

        fig.tight_layout()

        # 6. Save the figure
        # Ensure the output directory exists
        os.makedirs(self.patient_out_dir, exist_ok=True)
        output_path = os.path.join(self.patient_out_dir, output_filename)
        fig.savefig(output_path)
        plt.close(fig)  # Free up memory

        print(f"Plot saved successfully to {output_path}")
        return output_path


class ASRMProcessor(SurveyProcessor):
    """Processor for the ASRM (Altman Self-Rating Mania) survey."""

    survey_id = "ASRM"
    columns_to_extract = ["SC0", "EndDate"]

    def get_warnings(self, csv_path: str) -> Dict[str, bool]:
        val = float(self.parse_csv(csv_path)["SC0"])
        return {"(Hypo)mania": val > 6}


def init_processor(
    class_path: str, patient_id: str, survey_folder: str, patient_out_dir: str
):
    """Initializes and returns a survey processor instance.

    Args:
        class_path (str): Full import path to the processor class, e.g.
            "trbdv0.survey_processor.PHQ8Processor".
        patient_id (str): Identifier of the patient for whom to process surveys.
        survey_folder (str): Folder of quatrics surveys.
        patient_out_dir (str): Path to the directory where the processor should
            write its output for this patient.

    Returns:
        object: An instance of the specified processor class, initialized with
            (patient_id, survey_id, patient_out_dir).
    """
    module_name, cls_name = class_path.rsplit(".", 1)
    module = import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(patient_id, survey_folder, patient_out_dir)
