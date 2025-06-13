import os
import pandas as pd
from abc import ABC, abstractmethod
from importlib import import_module
from typing import List, Dict, Any
from trbdv0.utils import get_yesterdays_date


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
        Get the most recent survey CSV and extract its numeric values.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'path': the file path of the most recent survey CSV
                - each column from `columns_to_extract`: its extracted float value
        """
        latest_path = self.get_most_recent_survey_file()
        values = self.parse_csv(latest_path)
        return {"path": latest_path, **values}

    def get_yesterday_warnings(self) -> Dict[str, bool]:
        """
        Check if the most recent survey was filled out yesterday,
        and if so, return its warning flags; otherwise, return empty dict.
        """
        try:
            latest = self.get_most_recent_survey_file()
        except ValueError:
            return {}

        # Compare date portion of EndDate to yesterday's date
        filled_datetime = self.get_survey_filled_date(latest)
        filled_date = filled_datetime.split()[0]
        if filled_date == get_yesterdays_date():
            return self.get_warnings(latest)
        return {}


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
