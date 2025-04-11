import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
from matplotlib.cm import get_cmap
import os
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from utils import (
    get_todays_date,
    get_yesterdays_date,
    PHASE_MAPPING,
    calculate_average_met,
    get_past_dates,
    read_json,
)
from datetime import datetime
import pytz


class SleepData:
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

    def get_summary_stats(self) -> dict:
        """Get summary stats of sleep data"""
        # Collect counters by day
        sleep_counts = {}
        for entry in self.data:
            day = entry["day"]

            if day not in sleep_counts:
                sleep_counts[day] = {
                    "phases": Counter(),
                    "non_wear_time": 0,
                    "steps": 0,
                    "average_met": 0,
                    "met_interval": 0,
                    "met_items": [],
                    "met_timestamp": "",
                }

            # Update the phase counter
            sleep_counts[day]["phases"] += Counter(entry.get("sleep_phase_5_min", ""))

            # Accumulate or overwrite the other fields
            sleep_counts[day]["non_wear_time"] += entry.get("non_wear_time", 0) / 3600
            sleep_counts[day]["class_5_mins"] = entry.get("class_5_min", "")
            sleep_counts[day]["steps"] += entry.get("steps", 0)

            met_data = entry.get("met", {})
            sleep_counts[day]["met_interval"] = met_data.get("interval", 0)
            sleep_counts[day]["met_items"] = met_data.get("items", [])
            sleep_counts[day]["met_timestamp"] = met_data.get("timestamp", "")
            if met_data:
                sleep_counts[day]["average_met"] = calculate_average_met(entry)

        # Convert 5-min increments in phases to hours
        for day, day_data in sleep_counts.items():
            for phase, count in day_data["phases"].items():
                # Each count is a number of 5-min increments
                day_data["phases"][phase] = (count * 5) / 60.0

        # Flatten data into a DataFrame
        # Define which phase keys we care about
        phase_keys = ["1", "2", "3", "4"]  # "1"=Deep, "2"=Light, "3"=REM, "4"=Awake
        other_keys = [
            "non_wear_time",
            "class_5_mins",
            "steps",
            "average_met",
            "met_interval",
            "met_items",
            "met_timestamp",
        ]

        flattened_data = {}
        for d, day_data in sleep_counts.items():
            flattened_row = {}
            for ph in phase_keys:
                flattened_row[ph] = day_data["phases"].get(ph, 0.0)

            for k in other_keys:
                flattened_row[k] = day_data.get(k, np.nan)

            flattened_data[d] = flattened_row

        df = pd.DataFrame.from_dict(flattened_data, orient="index").sort_index()

        # Rename columns using PHASE_MAPPING
        df.columns = [PHASE_MAPPING.get(col, f"Phase {col}") for col in df.columns]

        # Reorder columns and fill missing
        column_order = [
            "Deep Sleep",
            "Light Sleep",
            "REM Sleep",
            "Awake",
            "Non-Wear Time",
            "Activity Classification",
            "Step Count",
            "Average MET",
            "MET Interval",
            "MET Items",
            "MET Timestamp",
        ]
        df = df.reindex(columns=column_order).fillna(0)
        df["Total Sleep"] = df[["Deep Sleep", "Light Sleep", "REM Sleep"]].sum(axis=1)

        today_date = get_todays_date()
        yesterday_date = get_yesterdays_date()
        yesterday_sleep = (
            df.loc[yesterday_date, "Total Sleep"]
            if yesterday_date in df.index
            else np.nan
        )
        yesterday_steps = (
            df.loc[yesterday_date, "Step Count"]
            if yesterday_date in df.index
            else np.nan
        )
        yesterday_average_met = (
            df.loc[yesterday_date, "Average MET"]
            if yesterday_date in df.index
            else np.nan
        )
        original_df = df.copy()
        date_range = (df.index.min(), df.index.max()) if not df.empty else (None, None)
        df = df[df.any(axis=1)]

        return {
            "patient": self.patient,
            "date_range": date_range,
            "today": today_date,
            "yesterday": yesterday_date,
            "yesterday_sleep": np.nan if not yesterday_sleep else yesterday_sleep,
            "yesterday_steps": np.nan if not yesterday_steps else yesterday_steps,
            "average_sleep": df["Total Sleep"].mean() if not df.empty else np.nan,
            "average_steps": df["Step Count"].mean() if not df.empty else np.nan,
            "yesterday_average_met": (
                np.nan if not yesterday_average_met else yesterday_average_met
            ),
            "average_met": df["Average MET"].mean() if not df.empty else np.nan,
            "sleep_df": original_df,
        }

    def plot_combined_sleep_plots(self, out_dir: str):
        """
        Plot sleep distribution and sleep habit polar plot side by side.
        """
        df = self.summary_stats["sleep_df"].drop(
            columns=[
                "Step Count",
                "Total Sleep",
                "Average MET",
                "MET Interval",
                "MET Items",
                "MET Timestamp",
                "Activity Classification",
            ]
        )
        if df.empty:
            return
        # Create a figure with two subplots
        fig = plt.figure(figsize=(30, 12), constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=2)

        # Plot sleep distribution on the first subplot
        ax1 = fig.add_subplot(gs[0, 0])
        color_palette = list(get_cmap("Set2").colors[:4]) + ["lightgray"]
        ax1 = df.plot(
            kind="bar",
            stacked=True,
            color=color_palette,
            alpha=0.7,
            edgecolor="black",
            ax=ax1,
        )
        ax1.set_title("Distribution of Sleep Phases per Day")
        ax1.set_ylabel("Hours")
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df.index, rotation=45)

        # Adding labels to each bar
        for p in ax1.patches:
            width, height = p.get_width(), p.get_height()
            if (
                height > 0
            ):  # Only label the bar if the height is significant to avoid clutter
                x, y = p.get_xy()
                ax1.text(
                    x + width / 2,
                    y + height / 2,
                    f"{height:.1f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        average_sleep = self.summary_stats["average_sleep"]
        ax1.axhline(
            y=average_sleep,
            color="r",
            linestyle="--",
            label=f"Average Sleep ({average_sleep:.2f} hrs)",
        )

        ax1.legend(
            title="Sleep Phases and Averages",
            loc="upper left",  # Aligns inside but we move it outside with bbox_to_anchor
            bbox_to_anchor=(1, 1),  # Moves it just outside the right side
        )

        # Plot spiral chart
        dataframe = pd.DataFrame(self.data)
        dataframe["day"] = pd.to_datetime(dataframe["day"])
        dataframe["bedtime_start"] = pd.to_datetime(
            dataframe["bedtime_start"].str.slice(0, 19)
        )
        dataframe["bedtime_end"] = pd.to_datetime(
            dataframe["bedtime_end"].str.slice(0, 19)
        )
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"].str.slice(0, 19))

        # Find the minimum date from bedtime_start
        dataframe = dataframe.dropna()
        min_day = dataframe["bedtime_start"].dt.date.min()

        colors = list(mcolors.TABLEAU_COLORS.values())
        day_colors = {
            day: colors[i % len(colors)]
            for i, day in enumerate(dataframe["day"].unique())
        }

        ax2 = fig.add_subplot(gs[:, 1], projection="polar")
        ax2.set_theta_direction(-1)
        ax2.set_theta_offset(np.pi / 2.0)
        ax2.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
        ax2.set_xticklabels([f"{i}:00" for i in range(24)])
        ax2.set_yticklabels([])

        ax2.grid(False)
        ax2.xaxis.grid(True, color="black", linestyle="-", linewidth=0.5, alpha=0.8)

        def plot_period(
            start_time,
            end_time,
            start_offset,
            color,
            linestyle="-",
            lw=10,
            alpha=1,
            edgecolor=None,
        ):
            start_angle = (start_time.hour * 60 + start_time.minute) / 1440 * 2 * np.pi
            end_angle = (end_time.hour * 60 + end_time.minute) / 1440 * 2 * np.pi

            if end_angle < start_angle:
                end_angle += 2 * np.pi

            start_radius = start_offset + start_angle / (2 * np.pi)
            end_radius = start_offset + end_angle / (2 * np.pi)

            angles = np.linspace(
                start_angle + 2 * np.pi * start_offset,
                end_angle + 2 * np.pi * start_offset,
                100,
            )
            radii = np.linspace(start_radius, end_radius, 100)

            if edgecolor:
                ax2.plot(
                    angles,
                    radii,
                    color=edgecolor,
                    linewidth=lw,
                    linestyle=linestyle,
                    alpha=alpha,
                )
            ax2.plot(
                angles,
                radii,
                color=color,
                linewidth=lw,
                linestyle=linestyle,
                alpha=alpha,
            )

        dataframe = dataframe.sort_values(by="bedtime_start")
        awake_periods = []
        non_worn_periods = []

        for index in range(len(dataframe)):
            row = dataframe.iloc[index]
            bedtime_start = row["bedtime_start"]
            bedtime_end = row["bedtime_end"]
            day_color = day_colors[row["day"]]
            class_5_min = row["class_5_min"]
            timestamp = row["timestamp"]

            start_offset = (bedtime_start.date() - min_day).days

            # Plot sleep period
            plot_period(bedtime_start, bedtime_end, start_offset, day_color)

            # Collect awake periods if not the last row
            if index < len(dataframe) - 1:
                next_row = dataframe.iloc[index + 1]
                next_bedtime_start = next_row["bedtime_start"]
                awake_start = bedtime_end
                awake_end = next_bedtime_start
                awake_start_offset = (awake_start.date() - min_day).days

                if awake_start.date() == awake_end.date():
                    awake_periods.append((awake_start, awake_end, awake_start_offset))
                else:
                    # Awake period spans multiple days
                    midnight = awake_start.replace(hour=23, minute=59, second=59)
                    awake_periods.append((awake_start, midnight, awake_start_offset))
                    next_start_offset = (awake_end.date() - min_day).days
                    next_midnight = awake_end.replace(hour=0, minute=0, second=0)
                    awake_periods.append((next_midnight, awake_end, next_start_offset))

            # Collect non-worn periods as continuous segments
            start_time = timestamp
            current_date = start_time.date()
            non_worn_offset = (current_date - min_day).days
            non_worn_start = None
            for i, status in enumerate(class_5_min):
                end_time = start_time + pd.Timedelta(minutes=5)
                if end_time.date() != current_date:
                    non_worn_offset += 1
                    current_date = end_time.date()
                if status == "0":
                    if non_worn_start is None:
                        non_worn_start = start_time
                else:
                    if non_worn_start is not None:
                        non_worn_periods.append(
                            (non_worn_start, start_time, non_worn_offset)
                        )
                        non_worn_start = None
                start_time = end_time
            # Add the last segment if it ends in a non-worn period
            if non_worn_start is not None:
                non_worn_periods.append((non_worn_start, start_time, non_worn_offset))

        # Plot awake periods first
        for start_time, end_time, start_offset in awake_periods:
            plot_period(
                start_time,
                end_time,
                start_offset,
                "white",
                linestyle="-",
                edgecolor="black",
            )
        # Plot non-worn periods last to ensure they are on top
        for start_time, end_time, start_offset in non_worn_periods:
            plot_period(
                start_time, end_time, start_offset, "lightgray", edgecolor="black"
            )

        # Custom legend
        custom_lines = [
            Line2D(
                [0],
                [0],
                color="white",
                lw=4,
                linestyle="--",
            ),
            Line2D([0], [0], color="lightgray", lw=4),
        ] + [
            Line2D([0], [0], color=day_colors[day], lw=4, linestyle="-")
            for day in dataframe["day"].unique()
        ]
        custom_labels = ["Awake Period", "Non-worn Period"] + [
            f"{day.date()}" for day in dataframe["day"].unique()
        ]
        ax2.legend(custom_lines, custom_labels, loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        df = self.summary_stats["sleep_df"]
        # Parse timestamps while ignoring the timezone offsets
        df = df.dropna(subset=["MET Timestamp"])

        # Ensure valid timestamps
        def safe_parse_timestamp(x):
            try:
                return (
                    datetime.strptime(x[:19], "%Y-%m-%dT%H:%M:%S")
                    if pd.notna(x) and x
                    else None
                )
            except ValueError:
                return None

        df["datetime"] = df["MET Timestamp"].apply(safe_parse_timestamp)
        df = df.dropna(subset=["datetime"])

        if df.empty:
            return

        df["day"] = df["datetime"].dt.date
        ax = fig.add_subplot(gs[1, 0])
        unique_days = sorted(df["day"].unique())

        y_scale = 15

        y_ticks = []
        y_labels = []

        # Loop over each day
        for i, day in enumerate(unique_days):
            # If each day is guaranteed to appear exactly once in the df, just pick that row:
            row = df.loc[df["day"] == day].iloc[0]

            # day_offset is the hour/minute at which the first MET item starts
            start_ts = row["datetime"]
            offset_h = start_ts.hour + start_ts.minute / 60.0 + start_ts.second / 3600.0
            day_met = np.array(row["MET Items"], dtype=float)

            # For each MET item, figure out which minute it belongs to (0..N-1),
            # convert to hours by dividing by 60, then add offset_h
            hour_met = np.arange(len(day_met)) / 60.0 + offset_h

            # Convert Activity Classification (5-min intervals) to an array
            activity_class = np.array(list(row["Activity Classification"]), dtype=int)

            # Expand each classification value over 5 consecutive minutes
            expanded_activity_class = np.repeat(activity_class, 5)

            # Ensure the length matches `day_met`
            expected_length = len(day_met)
            if len(expanded_activity_class) < expected_length:
                expanded_activity_class = np.pad(
                    expanded_activity_class,
                    (0, expected_length - len(expanded_activity_class)),
                    constant_values=0,
                )
            elif len(expanded_activity_class) > expected_length:
                # Trim if somehow longer
                expanded_activity_class = expanded_activity_class[:expected_length]

            # Identify non-wear periods (where classification is 0)
            non_wear_mask = expanded_activity_class == 0

            # The y-value: i * y_scale is the vertical offset for day i
            # So we do (i*y_scale + day_met).
            # Plot MET data
            ax.plot(
                hour_met,
                i * y_scale + day_met,
                color="mediumblue",
                alpha=0.67,
                label="Oura MET Score" if i == 0 else "",
            )

            # Overlay non-wear regions
            ax.scatter(
                hour_met[non_wear_mask],
                (i * y_scale + day_met)[non_wear_mask],
                color="gray",
                alpha=0.5,
                label="Non-Wear" if i == 0 else "",
            )

            # Keep track of where to place the day label on the y-axis
            y_ticks.append(i * y_scale)
            y_labels.append(day)

        ax.set_xlim([4, 28])

        hours = list(range(4, 29))
        labels = []
        for h in hours:
            hour = h if h < 24 else h - 24
            labels.append(f"{hour}:00")

        ax.set_xticks(hours)
        ax.set_xticklabels(labels, rotation=45)
        # Put each day on its own tick
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_title("MET Scores per Day")
        ax.legend(
            title="MET Scores per Day",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                out_dir,
                f"{self.patient}.png",
            )
        )
