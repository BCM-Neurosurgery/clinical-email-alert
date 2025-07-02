import os
import pandas as pd
from abc import ABC, abstractmethod
from importlib import import_module
from typing import List, Dict, Any
from trbdv0.utils import get_yesterdays_date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
import matplotlib.cm as cm


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

    def plot_historical_scores(
        self,
        output_filename: str = "PHQ8_historical_plot.png",
        cutoff: float = 10.0,
    ) -> str:
        """
        Generates a time-series plot of PHQ-8 scores.

        The plot shows the score on the y-axis against the date on the
        x-axis. A horizontal line and shaded region indicate the clinical
        cutoff for likely major depression.

        Args:
            output_filename (str): The name for the output plot file.
            cutoff (float): The clinical cutoff score.

        Returns:
            str: The absolute path to the saved plot image, or an empty
                 string if no data was found.
        """
        all_files = [
            file
            for files_in_year in self.get_survey_files_by_year().values()
            for file in files_in_year
        ]
        if not all_files:
            print(f"No {self.survey_id} files found for patient {self.patient_id}.")
            return ""

        survey_data = []
        for csv_path in all_files:
            try:
                scores = self.parse_csv(csv_path)
                date_str = self.get_survey_filled_date(csv_path)
                survey_data.append(
                    {
                        "date": pd.to_datetime(date_str),
                        "score": float(scores["SC0"]),
                    }
                )
            except (ValueError, KeyError) as e:
                print(f"Skipping file {os.path.basename(csv_path)} due to error: {e}")
                continue

        if not survey_data:
            print(
                f"No valid {self.survey_id} data found for patient {self.patient_id}."
            )
            return ""

        df = pd.DataFrame(survey_data).sort_values(by="date")

        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

        # Plot the data points and connecting line
        ax.plot(
            df["date"],
            df["score"],
            color="royalblue",
            alpha=0.6,
            marker="o",
            linestyle="-",
            zorder=2,
            markeredgecolor="black",
        )

        # Add cutoff line
        ax.axhline(cutoff, color="darkred", linestyle="--", lw=1.5, zorder=1)

        # Set y-axis limits to provide space for the shaded region
        y_max = max(df.score.max(), cutoff) * 1.15 + 2
        ax.set_ylim(bottom=-1, top=y_max)

        # Shade the "at-risk" area using axhspan
        ax.axhspan(cutoff, y_max, facecolor="red", alpha=0.15, zorder=0)

        # Add text label to explain the shaded region
        ax.text(
            df["date"].min(),
            cutoff + (y_max - cutoff) * 0.5,
            "Depression Range",
            fontsize=14,
            color="darkred",
            alpha=0.8,
            ha="left",
            va="center",
            style="italic",
            bbox=dict(
                facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2"
            ),
        )

        # Finalize plot aesthetics
        ax.set_title(
            f"PHQ-8 Score History for Patient: {self.patient_id}", fontsize=16, pad=20
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("PHQ-8 Score", fontsize=12)
        ax.grid(
            True, which="major", axis="y", linestyle=":", linewidth="0.5", color="gray"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Format dates on x-axis for readability
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        fig.tight_layout()

        # Save the figure
        os.makedirs(self.patient_out_dir, exist_ok=True)
        output_path = os.path.join(self.patient_out_dir, output_filename)
        fig.savefig(output_path)
        plt.close(fig)

        print(f"Plot saved successfully to {output_path}")
        return output_path


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

    def get_most_recent_free_response(self) -> str:
        """
        Retrieves the free response text (from column "Q8") from the most recent ISS survey.

        Returns:
            str: The text content of the "Q8" column from the latest survey.
                 Returns an empty string if no survey is found, the column is missing,
                 or the response is empty.
        """
        try:
            latest_path = self.get_most_recent_survey_file()
        except ValueError:
            # This happens if no survey files are found at all.
            return ""

        try:
            df = pd.read_csv(latest_path)

            if "Q8" not in df.columns:
                print(f"Warning: 'Q8' column not found in {latest_path}")
                return ""

            if len(df) < 2:
                print(f"Warning: Not enough rows in {latest_path} to extract data.")
                return ""

            response = df.iloc[1]["Q8"]
            if pd.isna(response):
                return ""

            return str(response)

        except Exception as e:
            print(f"Error processing free response from {latest_path}: {e}")
            return ""

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
        color gradient. The plot axes are dynamically set to ensure all four
        quadrants are always visible, regardless of data distribution.

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

        df = pd.DataFrame(survey_data).sort_values(by="date").reset_index()

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

        # Enforce a minimum size for each quadrant to ensure labels fit.
        padding = 15  # Extra space around the edges of the plot
        min_quadrant_width = 80  # Minimum space on each side of the vertical cutoff
        min_quadrant_height = 60  # Minimum space on each side of the horizontal cutoff

        # --- Calculate X-axis limits ---
        # Find data range left of the cutoff
        data_on_left = df.activation[df.activation < activation_cutoff]
        data_width_left = (
            activation_cutoff - data_on_left.min() if not data_on_left.empty else 0
        )
        final_width_left = max(data_width_left, min_quadrant_width)
        x_min_bound = activation_cutoff - final_width_left

        # Find data range right of the cutoff
        data_on_right = df.activation[df.activation > activation_cutoff]
        data_width_right = (
            data_on_right.max() - activation_cutoff if not data_on_right.empty else 0
        )
        final_width_right = max(data_width_right, min_quadrant_width)
        x_max_bound = activation_cutoff + final_width_right

        ax.set_xlim(x_min_bound - padding, x_max_bound + padding)

        # --- Calculate Y-axis limits ---
        # Find data range below the cutoff
        data_below = df.well_being[df.well_being < wellbeing_cutoff]
        data_height_below = (
            wellbeing_cutoff - data_below.min() if not data_below.empty else 0
        )
        final_height_below = max(data_height_below, min_quadrant_height)
        y_min_bound = wellbeing_cutoff - final_height_below

        # Find data range above the cutoff
        data_above = df.well_being[df.well_being > wellbeing_cutoff]
        data_height_above = (
            data_above.max() - wellbeing_cutoff if not data_above.empty else 0
        )
        final_height_above = max(data_height_above, min_quadrant_height)
        y_max_bound = wellbeing_cutoff + final_height_above

        ax.set_ylim(y_min_bound - padding, y_max_bound + padding)

        # Get final plot limits for label placement
        final_xlim = ax.get_xlim()
        final_ylim = ax.get_ylim()

        # Draw quadrant lines
        ax.axvline(activation_cutoff, color="grey", linestyle="--", lw=1.5)
        ax.axhline(wellbeing_cutoff, color="grey", linestyle="--", lw=1.5)

        # Place quadrant labels dynamically based on final plot boundaries
        text_props = dict(
            ha="center", va="center", fontsize=14, fontweight="bold", color="white"
        )
        bg_props = dict(boxstyle="round,pad=0.5", fc="gray", ec="none", alpha=0.6)

        # Depression (bottom-left)
        ax.text(
            (final_xlim[0] + activation_cutoff) / 2,
            (final_ylim[0] + wellbeing_cutoff) / 2,
            "Depression",
            **text_props,
            bbox=bg_props,
        )
        # Euthymia (top-left)
        ax.text(
            (final_xlim[0] + activation_cutoff) / 2,
            (wellbeing_cutoff + final_ylim[1]) / 2,
            "Euthymia",
            **text_props,
            bbox=bg_props,
        )
        # Mixed State (bottom-right)
        ax.text(
            (activation_cutoff + final_xlim[1]) / 2,
            (final_ylim[0] + wellbeing_cutoff) / 2,
            "Mixed State",
            **text_props,
            bbox=bg_props,
        )
        # (Hypo)Mania (top-right)
        ax.text(
            (activation_cutoff + final_xlim[1]) / 2,
            (wellbeing_cutoff + final_ylim[1]) / 2,
            "(Hypo)Mania",
            **text_props,
            bbox=bg_props,
        )

        # 1. Separate historical data from the most recent score
        df_historical = df.iloc[:-1]
        df_most_recent = df.iloc[-1:]

        # 2. Set up colormap normalization based on the full date range of the original dataframe
        numeric_dates = mdates.date2num(df["date"])
        # Handle case with single data point where min and max would be the same
        norm = (
            Normalize(vmin=numeric_dates.min(), vmax=numeric_dates.max())
            if len(numeric_dates) > 1
            else Normalize(vmin=numeric_dates[0] - 1, vmax=numeric_dates[0] + 1)
        )
        cmap = "viridis"

        # 3. Plot historical scores (if any) with standard circle markers
        if not df_historical.empty:
            ax.scatter(
                df_historical["activation"],
                df_historical["well_being"],
                c=mdates.date2num(df_historical["date"]),
                cmap=cmap,
                norm=norm,  # Apply the common normalization
                s=120,
                edgecolors="black",
                zorder=2,
            )

        # 4. Plot the most recent score with a star marker
        # This is guaranteed to have one row if we passed the initial "if not survey_data" check
        ax.scatter(
            df_most_recent["activation"].values[0],
            df_most_recent["well_being"].values[0],
            c=[
                mdates.date2num(df_most_recent["date"]).item()
            ],  # Pass color value in a list
            cmap=cmap,
            norm=norm,  # Apply the same normalization
            marker="*",
            s=450,
            edgecolors="white",
            linewidth=1.5,
            zorder=3,  # Place it on top
            label="Most Recent Score",
        )

        # Finalize plot aesthetics
        ax.set_title(
            f"ISS Score History for Patient: {self.patient_id}", fontsize=16, pad=20
        )
        ax.set_xlabel("Activation Score (SC1)", fontsize=12)
        ax.set_ylabel("Well-Being Score (SC2)", fontsize=12)
        ax.grid(True, which="both", linestyle=":", linewidth="0.5", color="gray")

        # It will automatically use the labels, markers, and colors defined above.
        # 'loc' places the legend in the best position to avoid data.
        legend = ax.legend(loc="best", fontsize=12)
        legend.get_frame().set_alpha(0.8)

        # Create a ScalarMappable for the colorbar to ensure it reflects the full range of dates
        mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(mappable, ax=ax, pad=0.02)

        cbar.set_label("Survey Date", fontsize=12)
        tick_locs = cbar.get_ticks()
        cbar.ax.set_yticklabels(
            [mdates.num2date(loc).strftime("%Y-%m-%d") for loc in tick_locs]
        )

        fig.tight_layout()

        # Save the figure
        os.makedirs(self.patient_out_dir, exist_ok=True)
        output_path = os.path.join(self.patient_out_dir, output_filename)
        fig.savefig(output_path)
        plt.close(fig)

        print(f"Plot saved successfully to {output_path}")
        return output_path


class ASRMProcessor(SurveyProcessor):
    """Processor for the ASRM (Altman Self-Rating Mania) survey."""

    survey_id = "ASRM"
    columns_to_extract = ["SC0", "EndDate"]

    def get_warnings(self, csv_path: str) -> Dict[str, bool]:
        val = float(self.parse_csv(csv_path)["SC0"])
        return {"(Hypo)mania": val > 6}

    def plot_historical_scores(
        self,
        output_filename: str = "ASRM_historical_plot.png",
        cutoff: float = 6.0,
    ) -> str:
        """
        Generates a time-series plot of ASRM scores.

        The plot shows the score on the y-axis against the date on the
        x-axis. A horizontal line and shaded region indicate the clinical
        cutoff for a likely (hypo)manic state.

        Args:
            output_filename (str): The name for the output plot file.
            cutoff (float): The clinical cutoff score.

        Returns:
            str: The absolute path to the saved plot image, or an empty
                 string if no data was found.
        """
        all_files = [
            file
            for files_in_year in self.get_survey_files_by_year().values()
            for file in files_in_year
        ]
        if not all_files:
            print(f"No {self.survey_id} files found for patient {self.patient_id}.")
            return ""

        survey_data = []
        for csv_path in all_files:
            try:
                scores = self.parse_csv(csv_path)
                date_str = self.get_survey_filled_date(csv_path)
                survey_data.append(
                    {
                        "date": pd.to_datetime(date_str),
                        "score": float(scores["SC0"]),
                    }
                )
            except (ValueError, KeyError) as e:
                print(f"Skipping file {os.path.basename(csv_path)} due to error: {e}")
                continue

        if not survey_data:
            print(
                f"No valid {self.survey_id} data found for patient {self.patient_id}."
            )
            return ""

        df = pd.DataFrame(survey_data).sort_values(by="date")

        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

        # Plot the data points and connecting line
        ax.plot(
            df["date"],
            df["score"],
            color="darkorange",
            alpha=0.7,
            marker="o",
            linestyle="-",
            zorder=2,
            markeredgecolor="black",
        )

        # Add cutoff line
        ax.axhline(cutoff, color="purple", linestyle="--", lw=1.5, zorder=1)

        # Set y-axis limits
        y_max = max(df.score.max(), cutoff) * 1.15 + 2
        ax.set_ylim(bottom=-1, top=y_max)

        # Shade the "at-risk" area
        ax.axhspan(cutoff, y_max, facecolor="purple", alpha=0.1, zorder=0)

        # Add text label
        ax.text(
            df["date"].min(),
            cutoff + (y_max - cutoff) * 0.5,
            "(Hypo)mania Range",
            fontsize=14,
            color="purple",
            alpha=0.8,
            ha="left",
            va="center",
            style="italic",
            bbox=dict(
                facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2"
            ),
        )

        # Finalize plot aesthetics
        ax.set_title(
            f"ASRM Score History for Patient: {self.patient_id}", fontsize=16, pad=20
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("ASRM Score", fontsize=12)
        ax.grid(
            True, which="major", axis="y", linestyle=":", linewidth="0.5", color="gray"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        fig.tight_layout()

        # Save the figure
        os.makedirs(self.patient_out_dir, exist_ok=True)
        output_path = os.path.join(self.patient_out_dir, output_filename)
        fig.savefig(output_path)
        plt.close(fig)

        print(f"Plot saved successfully to {output_path}")
        return output_path


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
