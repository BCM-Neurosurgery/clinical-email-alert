import os
from utils import (
    read_json,
    get_past_dates,
    get_todays_date,
    get_iter_dates,
)
import numpy as np


class Activity:
    def __init__(
        self,
        patient,
        config,
        patient_in_dir,
        patient_out_dir,
        logger,
        timezone="America/Chicago",
    ):
        """
        Initialize an Activity instance.
        :param patient: The patient associated with this activity.
        :param config: Configuration settings for the activity.
        :param patient_in_dir: Input directory for the patient.
        """
        self.patient = patient
        self.config = config
        self.timezone = timezone
        self.num_past_days = config["past_days"]
        self.today_date = get_todays_date()
        self.past_dates = get_past_dates(self.today_date, self.num_past_days)
        # often the daily_activity.json in the date folder records the daily activity
        # starting at 4am on that day, which would make the early part of the earliest
        # date missing when we join later, so we should go past that date further
        self.iter_past_dates = get_iter_dates(self.today_date, self.num_past_days + 2)
        # start date of the range, earliest
        self.start_date = self.past_dates[-1]
        # end date of the range, latest
        self.end_date = self.past_dates[0]
        # this contains a series of date folders
        # e.g. "2023-07-05", "2023-07-06", ...
        self.patient_in_dir = patient_in_dir
        # e.g. /home/auto/CODE/PerceptOCD/oura-null-pipeline/oura_out/DBSOCD002/2025-04-09
        self.patient_out_dir = patient_out_dir
        self.logger = logger
        self.ingest()

    def ingest(self):
        """Ingests activity data for a range of past dates.

        For each date in `self.past_dates`, this function attempts to read a corresponding
        `daily_activity.json` file from `self.patient_in_dir`. If the file does not exist,
        the date is skipped with a logged error. If the file exists but contains missing fields,
        those fields are filled with default values (e.g., NaN or None).

        Populates:
            self.activity_data (list of dict): Each dictionary contains activity-related information
            for a specific date entry with the following fields:
                - class_5_min (str or NaN): 5-minute resolution activity class string (e.g., sedentary, active).
                - non_wear_time (float or NaN): Estimated total non-wear time in seconds or minutes.
                - steps (int or NaN): Total number of steps for the day.
                - timestamp (str or NaN): Start time of the activity summary (typically midnight).
                - met (dict or None): Dictionary of metabolic equivalent task (MET) data with keys:
                    * interval (int): Sampling interval in seconds
                    * items (list of float): MET values per interval
                    * timestamp (str): Start timestamp of the MET time series
        """
        self.activity_data = []
        self.activity_phases = []
        self.met = []
        self.steps = []
        self.nonweartime = []

        for date in self.iter_past_dates:
            patient_date_json = os.path.join(
                self.patient_in_dir, date, "daily_activity.json"
            )

            if not os.path.exists(patient_date_json):
                self.logger.error(f"{date} daily_activity.json not found.")
                continue

            try:
                activity_data = read_json(patient_date_json)
            except Exception as e:
                self.logger.error(f"Failed to read JSON for {date}: {e}")
                continue

            for activity_entry in activity_data:
                entry = {
                    "class_5_min": activity_entry.get("class_5_min", np.nan),
                    "non_wear_time": activity_entry.get("non_wear_time", np.nan),
                    "steps": activity_entry.get("steps", np.nan),
                    "timestamp": activity_entry.get("timestamp", np.nan),
                    "met": activity_entry.get("met", None),
                }
                self.activity_data.append(entry)

                self.met.append(entry["met"])

                self.activity_phases.append(
                    {
                        "class_5_min": entry["class_5_min"],
                        "timestamp": entry["timestamp"],
                    }
                )

                self.steps.append(
                    {
                        "steps": entry["steps"],
                        "timestamp": entry["timestamp"],
                    }
                )

                self.nonweartime.append(
                    {
                        "non_wear_time": entry["non_wear_time"],
                        "timestamp": entry["timestamp"],
                    }
                )
