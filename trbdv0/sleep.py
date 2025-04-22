import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils import (
    get_todays_date,
    get_past_dates,
    read_json,
)
import pytz


class Sleep:
    def __init__(
        self, patient, config, patient_in_dir, patient_out_dir, logger
    ) -> None:
        """Init class

        Args:
            patient (str): e.g. Percept010, DBSOCD001
            config (dict): config dict
            patient_in_dir (str): input oura folder which contains date folders
            patient_out_dir (str): output folder for this patient
            data (list): list of dicts of sleep info
        """
        self.patient = patient
        self.config = config
        self.num_past_days = config["past_days"]
        self.study_name = config["study_name"]
        self.today_date = get_todays_date()
        self.past_dates = get_past_dates(self.today_date, self.num_past_days)
        # start date of the range, earliest
        self.start_date = self.past_dates[0]
        # end date of the range, latest
        self.end_date = self.past_dates[-1]
        # this contains a series of date folders
        # e.g. "2023-07-05", "2023-07-06", ...
        self.patient_in_dir = patient_in_dir
        # e.g. /home/auto/CODE/PerceptOCD/oura-null-pipeline/oura_out/DBSOCD002/2025-04-09
        self.patient_out_dir = patient_out_dir
        self.logger = logger
        # ingest sleep.json data into dataframe
        self.ingest()
        # get only bedtimes
        self.bedtimes_df = self.get_bedtimes_df()
        # transform the bedtimes and break it down to chunks
        self.splitted_bedtimes_df = self.split_overnight_sleep(self.bedtimes_df)
        # add missing dates with empty rows
        self.filled_splitted_bedtimes_df = self.fill_missing_dates(
            self.splitted_bedtimes_df
        )
        self.plot_bedtime_schedule(self.filled_splitted_bedtimes_df)
        print()

    def get_start_date(self):
        return self.start_date

    def get_end_date(self):
        return self.end_date

    def get_past_dates(self):
        return self.past_dates

    def get_patient(self):
        return self.patient

    def get_bedtimes(self):
        return self.bedtimes

    def get_sleep_data(self):
        return self.sleep_data

    def ingest(self):
        """Ingests sleep data for a range of past dates.

        For each date in `self.past_dates`, this function attempts to read a corresponding
        `sleep.json` file from `self.patient_in_dir`. If the file does not exist, the date is skipped.
        If the file exists but has missing fields, those fields are filled with default values (e.g., NaN).

        Populates:
            self.sleep_data (list of dict): Each dict contains:
                - sleep_phase_5_min (str or NaN)
                - bedtime_start (datetime string or NaN)
                - bedtime_end (datetime string or NaN)
                - total_sleep_duration (float or NaN)

            self.bedtimes (list of dict): Each dict contains:
                - bedtime_start (datetime string or NaN)
                - bedtime_end (datetime string or NaN)
        """
        self.sleep_data = []
        self.bedtimes = []

        for date in self.past_dates:
            patient_date_json = os.path.join(self.patient_in_dir, date, "sleep.json")

            if not os.path.exists(patient_date_json):
                self.logger.error(f"{date} sleep data not found.")
                continue

            try:
                sleep_data = read_json(patient_date_json)
            except Exception as e:
                self.logger.error(f"Failed to read JSON for {date}: {e}")
                continue

            for sleep_entry in sleep_data:
                entry = {
                    "sleep_phase_5_min": sleep_entry.get("sleep_phase_5_min", np.nan),
                    "bedtime_start": sleep_entry.get("bedtime_start", np.nan),
                    "bedtime_end": sleep_entry.get("bedtime_end", np.nan),
                    "total_sleep_duration": sleep_entry.get(
                        "total_sleep_duration", np.nan
                    ),
                }
                self.sleep_data.append(entry)

                # Collect bedtime-only entries
                self.bedtimes.append(
                    {
                        "bedtime_start": entry["bedtime_start"],
                        "bedtime_end": entry["bedtime_end"],
                    }
                )

    def get_bedtimes_df(self) -> pd.DataFrame:
        """Transforms sleep schedule data into a DataFrame indexed by sleep end date.

        Converts `bedtime_start` and `bedtime_end` strings to timezone-aware datetime objects
        in America/Chicago time, and constructs a DataFrame indexed by the date part of
        `bedtime_end` (i.e., the wake-up day).

        Handles missing or malformed datetime fields by setting them to NaT (Not a Time).

        Returns:
            pd.DataFrame: A DataFrame indexed by date (based on bedtime_end),
                        with columns:
                            - bedtime_start (datetime in Chicago timezone or NaT)
                            - bedtime_end (datetime in Chicago timezone or NaT)
        """
        if not self.sleep_data:
            return pd.DataFrame()

        df = pd.DataFrame(self.bedtimes)

        df["bedtime_start"] = pd.to_datetime(
            df["bedtime_start"], errors="coerce", utc=True
        )
        df["bedtime_end"] = pd.to_datetime(df["bedtime_end"], errors="coerce", utc=True)

        # Convert to Chicago timezone
        chicago_tz = pytz.timezone("America/Chicago")
        df["bedtime_start"] = df["bedtime_start"].dt.tz_convert(chicago_tz)
        df["bedtime_end"] = df["bedtime_end"].dt.tz_convert(chicago_tz)

        # Create 'day' column from bedtime_end (for indexing), keep NaT if bedtime_end is missing
        df["day"] = df["bedtime_start"].dt.date

        # Drop rows only if BOTH start and end are missing
        df = df.dropna(subset=["bedtime_start", "bedtime_end"], how="all")
        df.set_index("day", inplace=True)

        return df[["bedtime_start", "bedtime_end"]]

    def split_overnight_sleep(self, df: pd.DataFrame) -> pd.DataFrame:
        """Splits overnight sleep periods into two rows: before and after midnight.

        For each sleep period, if the sleep crosses midnight, it is split into two rows:
        one from `bedtime_start` to midnight, and one from midnight to `bedtime_end`.

        Args:
            df (pd.DataFrame): DataFrame indexed by day with columns:
                            - bedtime_start (datetime)
                            - bedtime_end (datetime)

        Returns:
            pd.DataFrame: A transformed DataFrame with possibly more rows,
                        each representing part of a sleep segment that
                        does not cross midnight.
        """
        if df.empty:
            return df.copy()

        split_rows = []
        for day, row in df.iterrows():
            start = row["bedtime_start"]
            end = row["bedtime_end"]

            if pd.isna(start) or pd.isna(end):
                continue

            if start.date() == end.date():
                split_rows.append({"bedtime_start": start, "bedtime_end": end})
            else:
                # Sleep crosses midnight, split into two
                midnight = start.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + pd.Timedelta(days=1)
                split_rows.append({"bedtime_start": start, "bedtime_end": midnight})
                split_rows.append({"bedtime_start": midnight, "bedtime_end": end})

        result_df = pd.DataFrame(split_rows)

        # Create index again from bedtime_end's date
        result_df["day"] = result_df["bedtime_start"].dt.date
        result_df.set_index("day", inplace=True)

        return result_df[["bedtime_start", "bedtime_end"]]

    def fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing dates in the sleep dataframe with NaT rows.

        Args:
            df (pd.DataFrame): Original sleep DataFrame with datetime.date index.

        Returns:
            pd.DataFrame: DataFrame with all expected dates included,
                        and NaT rows inserted for missing days.
        """
        full_range = [pd.to_datetime(d).date() for d in self.past_dates]

        existing_days = set(df.index)
        missing_days = set(full_range) - existing_days

        empty_rows = pd.DataFrame(
            {
                "bedtime_start": pd.NaT,
                "bedtime_end": pd.NaT,
            },
            index=sorted(missing_days),
        )

        # Combine and sort
        df_filled = pd.concat([df, empty_rows], axis=0).sort_index()

        return df_filled

    def plot_bedtime_schedule(self, df: pd.DataFrame, title="Sleep Schedule", ax=None):
        """Plots sleep segments for each date as horizontal bars.

        Args:
            df (pd.DataFrame): DataFrame with columns `bedtime_start` and `bedtime_end`,
                            and date as index (after split into sleep segments).
            title (str): Title of the plot.
            ax (matplotlib.axes.Axes, optional): An existing axis to plot into.
                                                If None, a new figure and axis are created.

        Returns:
            tuple: (fig, ax) where:
                - fig (matplotlib.figure.Figure): The figure object.
                - ax (matplotlib.axes.Axes): The axis containing the sleep schedule plot.
        """
        if df.empty:
            print("No sleep data to plot.")
            return None, None

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, max(5, len(df.index.unique()) * 0.4)))
        else:
            fig = ax.figure

        for i, (day, row) in enumerate(df.iterrows()):
            start = row["bedtime_start"]
            end = row["bedtime_end"]

            if pd.isna(start) or pd.isna(end):
                continue

            start_hour = start.hour + start.minute / 60
            end_hour = end.hour + end.minute / 60

            if end_hour < start_hour:
                end_hour += 24

            ax.barh(
                y=day,
                width=end_hour - start_hour,
                left=start_hour,
                height=0.6,
                color="skyblue",
                edgecolor="black",
            )

        ax.set_xlim(0, 24)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Date")
        ax.set_title(title)
        ax.set_xticks(range(0, 25, 2))
        ax.set_yticks(sorted(df.index.unique()))
        ax.set_yticklabels([str(d) for d in sorted(df.index.unique())])
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        fig.savefig("debug_sleep_plot.png")

        return fig, ax
