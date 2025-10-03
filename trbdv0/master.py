import pandas as pd
from trbdv0.sleep import Sleep
from trbdv0.activity import Activity
import pytz
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from trbdv0.utils import (
    get_yesterdays_date,
    get_todays_date,
    get_iter_dates,
    get_last_day,
)
from trbdv0.constants import *
from datetime import datetime
import numpy as np
import os


class Master:
    def __init__(self, sleep: Sleep, activity: Activity, timezone="America/Chicago"):
        self.sleep = sleep
        self.activity = activity
        self.timezone = timezone
        self.logger = self.sleep.logger
        self.study_name = self.sleep.study_name
        self.patient = self.sleep.get_patient()
        self.plot_save_path = os.path.join(
            self.sleep.patient_out_dir, f"{self.patient}_wearables.png"
        )
        self.sleep_hours_plot = os.path.join(
            self.sleep.patient_out_dir, f"{self.patient}_sleep_hours.png"
        )
        self.patient = self.sleep.get_patient()
        self.analysis_timegrid = self.build_time_grid(
            num_past_days=self.sleep.num_past_days, end_date="yesterday"
        )
        self.master_integrated_time = self.build_master_integrated_time(
            self.analysis_timegrid
        )
        self.plot_timegrid = self.build_time_grid(
            num_past_days=self.sleep.num_past_days + 1, end_date="today"
        )
        self.plot_integrated_time = self.build_master_integrated_time(
            self.plot_timegrid
        )

    def build_time_grid(
        self, num_past_days: int, offset: int = 12, end_date: str = "yesterday"
    ) -> pd.DataFrame:
        """Builds a 1-minute resolution time grid from start_date to end_date (inclusive),
        with timestamps localized to the given timezone (e.g., America/Chicago).

        Args:
            num_past_days (int):
                Number of past days to include in the grid, starting from the end_date.
            offset (int):
                Hour of day (0-23) at which each day's window begins (e.g., 12 for 12 PM).
            end_date (str):
                Which date to end the grid on. Must be one of:
                - `"yesterday"`: uses `get_yesterdays_date(self.timezone)`
                - `"today"`: uses `get_todays_date(self.timezone)`

        Raises:
            ValueError: If `end_date` is not `"yesterday"` or `"today"`.

        Returns:
            pd.DataFrame:
                Single-column DataFrame named `"timestamp"`, containing one-minute
                localized (to `self.timezone`) timestamps from the start to end.

        Definition updated on 2025-04-29, we define the date range to be up to yesterday 12pm at the
        time of running the program.
            - if the program is run on 2025-04-29, the date range will be
            - from 2025-04-14 12pm to 2025-04-28 12pm (America/Chicago)
            - aka we will not include the actual yesterday's data from 2025-04-28 12pm to 2025-04-29 12pm

        Definition updated on 2025-06-12, we gave the option to specify the end_date. If it's "yesterday", then
        the end date will be yesterday at 12pm, if it's "today", then the end date will be today at 12pm.
        """
        tz = pytz.timezone(self.timezone)

        # Resolve the reference end date
        if end_date.lower() == "yesterday":
            ref_date_str = get_yesterdays_date(self.timezone)
        elif end_date.lower() == "today":
            ref_date_str = get_todays_date(self.timezone)
        else:
            raise ValueError("end_date must be either 'yesterday' or 'today'")

        # we need to get self.num_past_days + 1 days so that
        # we actually have self.num_past_days intervals
        date_range = get_iter_dates(ref_date_str, num_past_days + 1)
        start_date, end_date = date_range[0], date_range[-1]

        start_ts = pd.to_datetime(start_date).replace(hour=offset, minute=0, second=0)
        end_ts = pd.to_datetime(end_date).replace(hour=offset, minute=0, second=0)

        # Localize start and end to the specified timezone
        start_ts = tz.localize(start_ts)
        end_ts = tz.localize(end_ts)

        # Generate the 1-minute timestamp range
        timeline = pd.date_range(
            start=start_ts, end=end_ts, freq="1min", inclusive="left"
        )

        return pd.DataFrame({"timestamp": timeline})

    def build_master_sleep_bedtimes(self) -> pd.DataFrame:
        """Builds a master sleep timeline with 1-minute resolution from sleep bedtimes.

        The function normalizes each bedtime start/end to the local timezone, compiles
        the intervals, and emits a single row per minute. When overlapping bedtime
        records occur (e.g., duplicate uploads), the earliest interval seen for a
        given minute wins so downstream merges do not double-count time in bed.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - timestamp: unique 1-minute timestamps between bedtime_start and bedtime_end
                - source: 'sleep'
        """
        tz = pytz.timezone(self.timezone)
        minute_rows = []
        seen_minutes = set()  # avoid duplicate timestamps from overlapping bedtimes

        bedtime_entries = []
        for entry in self.sleep.get_bedtimes():
            start = pd.to_datetime(
                entry.get("bedtime_start"), errors="coerce", utc=True
            )
            end = pd.to_datetime(entry.get("bedtime_end"), errors="coerce", utc=True)

            if pd.isna(start) or pd.isna(end):
                continue

            start = start.tz_convert(tz).floor("min")
            end = end.tz_convert(tz).ceil("min")

            if start >= end:
                continue

            bedtime_entries.append((start, end))

        if not bedtime_entries:
            return pd.DataFrame()

        bedtime_entries.sort(
            key=lambda item: (item[0], -(item[1] - item[0]).total_seconds())
        )

        for start, end in bedtime_entries:
            for ts in pd.date_range(
                start=start, end=end, freq="1min", inclusive="left"
            ):
                if ts in seen_minutes:
                    continue
                seen_minutes.add(ts)
                minute_rows.append({"timestamp": ts, "source": "sleep"})

        return pd.DataFrame(minute_rows)

    def build_master_sleep_phases(self) -> pd.DataFrame:
        """Builds a 1-minute resolution DataFrame of sleep phases.

        Sleep entries are normalized to the local timezone and expanded into 1-minute
        rows per phase bucket. When different records cover the same minute, the first
        entry processed retains ownership of that timestamp, ensuring a single
        sleep phase per minute in the master timeline.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - timestamp: unique 1-minute timestamps during sleep
                - phase: int code (1=deep, 2=light, 3=REM, 4=awake)
                - phase_label: human-readable sleep phase
                - source: 'sleep_phase'
        """
        tz = pytz.timezone(self.timezone)
        phase_map = {"1": "deep", "2": "light", "3": "REM", "4": "awake"}

        all_rows = []
        seen_minutes = set()  # keep only the first phase recorded for each minute

        phase_entries = []

        for entry in self.sleep.sleep_data:
            phase_string = entry.get("sleep_phase_5_min")
            start_str = entry.get("bedtime_start")

            if not phase_string or not start_str:
                continue

            try:
                start_time = (
                    pd.to_datetime(start_str, utc=True).tz_convert(tz).floor("min")
                )
            except Exception as e:
                self.logger.warning(f"Invalid bedtime_start: {start_str} — {e}")
                continue

            duration = len(phase_string) * 5
            phase_entries.append((start_time, duration, phase_string))

        if not phase_entries:
            return pd.DataFrame()

        phase_entries.sort(key=lambda item: (item[0], -item[1]))

        for start_time, duration_minutes, phase_string in phase_entries:
            for i, char in enumerate(phase_string):
                if char not in phase_map:
                    continue

                phase = int(char)
                label = phase_map[char]
                segment_start = start_time + timedelta(minutes=5 * i)

                for j in range(5):
                    ts = segment_start + timedelta(minutes=j)
                    if ts in seen_minutes:
                        continue
                    seen_minutes.add(ts)
                    all_rows.append(
                        {
                            "timestamp": ts,
                            "phase": phase,
                            "phase_label": label,
                            "source": "sleep_phase",
                        }
                    )

        return pd.DataFrame(all_rows)

    def build_master_activity_phase(self) -> pd.DataFrame:
        """Builds a 1-minute resolution DataFrame of activity phases from all activity data entries.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - timestamp: 1-min resolution timestamps during activity measurement
                - activity_class: int (0-5)
                - activity_label: string label for activity level
                - source: 'activity_class'
        """
        tz = pytz.timezone(self.timezone)
        class_map = {
            "0": "no_wear",
            "1": "rest",
            "2": "inactive",
            "3": "low_activity",
            "4": "medium_activity",
            "5": "high_activity",
        }

        all_rows = []

        for entry in self.activity.activity_phases:
            class_string = entry.get("class_5_min")
            start_str = entry.get("timestamp")

            if not class_string or not start_str:
                continue

            try:
                start_time = (
                    pd.to_datetime(start_str, utc=True).tz_convert(tz).floor("min")
                )
            except Exception as e:
                self.logger.warning(f"Invalid timestamp: {start_str} — {e}")
                continue

            for i, char in enumerate(class_string):
                if char not in class_map:
                    continue  # skip unknown class codes

                class_code = int(char)
                label = class_map[char]
                segment_start = start_time + timedelta(minutes=5 * i)

                for j in range(5):
                    ts = segment_start + timedelta(minutes=j)
                    all_rows.append(
                        {
                            "timestamp": ts,
                            "activity_class": class_code,
                            "activity_label": label,
                            "source": "activity_class",
                        }
                    )

        return pd.DataFrame(all_rows)

    def build_master_met(self) -> pd.DataFrame:
        """Builds a 1-minute resolution DataFrame of MET values from activity MET data.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - timestamp: Timestamp for each MET reading
                - met: MET value (float)
                - source: 'met'
        """
        tz = pytz.timezone(self.timezone)
        all_dfs = []

        for entry in self.activity.met:
            interval_sec = entry.get("interval", 60)
            met_values = entry.get("items")
            start_str = entry.get("timestamp")

            if not met_values or not start_str:
                continue

            try:
                start_time = (
                    pd.to_datetime(start_str, utc=True).tz_convert(tz).floor("min")
                )
            except Exception as e:
                self.logger.warning(f"Invalid MET timestamp: {start_str} — {e}")
                continue

            timestamps = [
                start_time + timedelta(seconds=interval_sec * i)
                for i in range(len(met_values))
            ]

            df = pd.DataFrame({"timestamp": timestamps, "met": met_values})
            df["source"] = "met"

            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def build_master_integrated_time(
        self, time_grid: pd.DataFrame, offset: int = 12
    ) -> pd.DataFrame:
        """Builds a unified 1-minute resolution timeline combining sleep, activity, and MET data.

        Args:
            time_grid (pd.DataFrame):
                DataFrame with a “timestamp” column of 1-minute, localized datetimes
                to use as the base timeline.
            offset (int):
                Number of hours to offset the day boundary (e.g., 12 means day runs
                from 12 PM on one date to 12 PM the next). Used to compute
                `shifted_day` and `shifted_hour`.

        Returns:
            pd.DataFrame: DataFrame indexed by timestamp with the following columns:
                - in_bed: True if within a sleep interval
                - sleep_phase: integer 1-4 if available
                - sleep_phase_label: str if available
                - activity_class: integer 0-5 if available
                - activity_label: str if available
                - met: float MET value if available
        """
        # Base timeline
        df = time_grid.copy()

        # Sleep bedtimes
        df_in_bed = self.build_master_sleep_bedtimes()
        if not df_in_bed.empty:
            df_in_bed["in_bed"] = True
            df = df.merge(
                df_in_bed[["timestamp", "in_bed"]], on="timestamp", how="left"
            )

        # Sleep phases
        df_sleep_phase = self.build_master_sleep_phases()
        if not df_sleep_phase.empty:
            df = df.merge(
                df_sleep_phase[["timestamp", "phase", "phase_label"]],
                on="timestamp",
                how="left",
            )

        # Activity class
        df_activity = self.build_master_activity_phase()
        if not df_activity.empty:
            df = df.merge(
                df_activity[["timestamp", "activity_class", "activity_label"]],
                on="timestamp",
                how="left",
            )

        # MET data
        df_met = self.build_master_met()
        if not df_met.empty:
            df = df.merge(df_met[["timestamp", "met"]], on="timestamp", how="left")

        # 8. Fill default values
        if "in_bed" in df.columns:
            df["in_bed"] = df["in_bed"].astype("boolean").fillna(False)

        # 9. Add day field
        df["day"] = df["timestamp"].dt.date

        # define shifted_day such that 2025-04-10 12pm to 2025-04-11 12pm is 2025-04-10
        # anything that falls before 12pm is considered the previous day
        df["shifted_day"] = (df["timestamp"] - pd.Timedelta(hours=offset)).dt.date
        # define shifted_hour such at 12pm is 0, 1pm is 1, 2pm is 2, etc.
        # with % because 3am is 15, 4am is 16, etc.
        df["shifted_hour"] = (
            df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0 - offset
        ) % 24

        if "phase" in df.columns:
            df.rename(columns={"phase": "sleep_phase"}, inplace=True)
        if "phase_label" in df.columns:
            df.rename(columns={"phase_label": "sleep_phase_label"}, inplace=True)

        return df

    def plot_integrated_sleep_activity_schedule(
        self, title="Sleep & Activity Schedule", ax=None
    ):
        """Plots a Gantt chart with in-bed intervals as background, overlaid with sleep phase, activity class 0 (no-wear),
        and low MET segments.

        Args:
            df (pd.DataFrame): Output from `build_master_integrated_time()`
            title (str): Plot title
        """
        # Define state color map
        state_colors = {
            "not_worn/battery_dead": "#aaaaaa",  # medium gray with hatch
            "deep_sleep": "#0b3d91",  # dark blue
            "light_sleep": "#3c82e0",  # medium blue
            "REM_sleep": "#9ec5f2",  # light blue
            "awake": "#e6e6e6",  # light grey
            "unidentified": "#ffe17b",  # soft yellow for unknown in-bed state
        }

        df = self.plot_integrated_time
        if df.empty:
            print("No data to plot.")
            return
        # if there's no in_bed column
        # that means there's no sleep data
        if "in_bed" not in df.columns:
            print("No sleep data to plot.")
            return

        df = df.copy()
        days = sorted(df["shifted_day"].unique())
        day_to_y = {day: i for i, day in enumerate(days)}

        height = max(6, len(days) * 0.4)
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, height))
        else:
            fig = ax.figure

        for day in days:
            day_df = df[(df["shifted_day"] == day) & (df["in_bed"] == True)].copy()
            is_yesterday = day == days[-1]

            if day_df.empty:
                continue

            day_df["state"] = day_df.apply(self.assign_state, axis=1)

            # Step 3: Plot each state as a block
            for state, state_df in day_df.groupby("state"):
                segments = self.get_segments(state_df)
                for start, end in segments:
                    ax.barh(
                        y=day_to_y[day],
                        width=end - start,
                        left=start,
                        height=0.6,
                        color=state_colors.get(state),
                        alpha=(
                            0.4 if is_yesterday else 1.0
                        ),  # signal that yesterday's sleep is not included in table
                        edgecolor=(
                            "#888888" if state == "not_worn/battery_dead" else None
                        ),
                        hatch="///" if state == "not_worn/battery_dead" else None,
                        linewidth=0.2,
                        zorder=2,
                    )

        # Annotate yesterday to indicate exclusion from analysis
        yesterday = days[-1]
        y = day_to_y[yesterday]
        ax.axhspan(
            y - 0.3,
            y + 0.3,
            facecolor="#cccccc",
            alpha=0.15,
            zorder=1,
        )
        ax.text(
            12,
            y,
            "Excluded from analysis",
            va="center",
            ha="center",
            fontsize=10,
            color="red",
            zorder=3,
        )

        # Final formatting
        ax.set_xlim(0, 24)
        xticks = range(0, 25, 2)
        xticklabels = [
            f"{(12 + h) % 12 or 12} {'AM' if (12 + h) % 24 < 12 else 'PM'}"
            for h in xticks
        ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(f"Hour of Day (starting at 12PM — {self.timezone})")
        ax.tick_params("x", labelbottom=True)
        ax.set_yticks(range(len(days)))
        ax.set_yticklabels([str(day) for day in days])
        # reserve exactly one “bar‐height” above and below
        ax.set_ylim(-0.5, len(days) - 0.5)
        ax.set_title(f"{title} — Patient {self.patient}")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        # Legend
        legend_handles = [
            mpatches.Patch(
                facecolor=color,
                label=state.replace("_", " ").capitalize(),
                hatch="///" if state == "not_worn/battery_dead" else None,
                edgecolor="#888888" if state == "not_worn/battery_dead" else None,
                linewidth=0.2,
            )
            for state, color in state_colors.items()
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),  # move legend outside the right edge
            borderaxespad=0.0,
            frameon=True,
        )

        fig.tight_layout()

        return fig, ax

    def assign_state(self, row):
        if (
            row.get("activity_class") == 0
            or pd.isna(row.get("met"))
            or row.get("met") <= 0.1
        ):
            return "not_worn/battery_dead"
        if row.get("sleep_phase_label") == "deep":
            return "deep_sleep"
        if row.get("sleep_phase_label") == "light":
            return "light_sleep"
        if row.get("sleep_phase_label") == "REM":
            return "REM_sleep"
        if row.get("sleep_phase_label") == "awake":
            return "awake"
        return "unidentified"

    def get_segments(self, df: pd.DataFrame, offset: int = 12) -> list:
        """
        Returns (start_hour, end_hour) segments where rows are contiguous by 1-minute timestamps,
        with optional hour offset for custom day alignment (e.g., 12pm-to-12pm = offset 12).

        Args:
            df (pd.DataFrame): DataFrame with 1-minute timestamps.
            offset (int): Number of hours to offset the hour-of-day (default 12 for 12pm–12pm).

        Returns:
            list of (start_hour, end_hour) tuples in offset-adjusted hour-of-day space.
        """
        segments = []
        current_start = None
        previous_hour = None

        for i, timestamp in enumerate(df["timestamp"]):
            # Apply offset to wrap day around (e.g., 12pm → 0)
            hour = (timestamp.hour + timestamp.minute / 60.0 - offset) % 24

            if current_start is None:
                current_start = hour
            elif (timestamp - df["timestamp"].iloc[i - 1]).seconds > 60:
                segments.append((current_start, previous_hour + 1 / 60))
                current_start = hour

            previous_hour = hour

        if current_start is not None:
            segments.append((current_start, previous_hour + 1 / 60))

        return segments

    def plot_met_buckets_gantt(self, title="MET Binned Gantt Chart", ax=None):
        """
        Plots MET Gantt chart using 7 discrete color buckets. One row per day.
        """
        df = self.plot_integrated_time
        if df.empty or "met" not in df.columns:
            print("No MET data to plot.")
            return

        df = df.copy()
        df = df.sort_values(["shifted_day", "shifted_hour"])

        non_worn_label = "non_worn/battery_dead"

        # Define color buckets
        # Step 1: Define value-based bucket labels
        def bucketize_met(val):
            if pd.isna(val) or val <= 0.1:
                return non_worn_label
            elif val < 0.5:
                return "0.1 - 0.5"
            elif val < 1.0:
                return "0.5 - 1.0"
            elif val < 2.0:
                return "1.0 - 2.0"
            elif val < 3.0:
                return "2.0 - 3.0"
            elif val < 4.5:
                return "3.0 - 4.5"
            else:
                return "> 4.5"

        df["met_bucket"] = df["met"].apply(bucketize_met)

        # Color map
        bucket_colors = {
            non_worn_label: "#aaaaaa",  # gray, non-worn
            "0.1 - 0.5": "#deebf7",  # very light blue
            "0.5 - 1.0": "#9ecae1",  # light blue
            "1.0 - 2.0": "#6baed6",  # medium blue
            "2.0 - 3.0": "#4292c6",  # standard blue
            "3.0 - 4.5": "#2171b5",  # deep blue
            "> 4.5": "#084594",  # darkest blue
        }

        days = sorted(df["shifted_day"].unique())
        day_to_y = {day: i for i, day in enumerate(days)}

        height = max(6, len(days) * 0.4)
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, height))
        else:
            fig = ax.figure

        for day in days:
            day_df = df[df["shifted_day"] == day]
            for bucket, bucket_df in day_df.groupby("met_bucket"):
                segments = self.get_segments(bucket_df)
                for start, end in segments:
                    ax.barh(
                        y=day_to_y[day],
                        left=start,
                        width=end - start,
                        height=0.6,
                        color=bucket_colors[bucket],
                        hatch="///",
                        edgecolor=("#888888" if bucket == non_worn_label else "none"),
                        linewidth=0.2,
                        zorder=2,
                    )

        # Annotate yesterday to indicate exclusion from analysis
        yesterday = days[-1]
        y = day_to_y[yesterday]
        ax.axhspan(
            y - 0.3,
            y + 0.3,
            facecolor="#cccccc",
            alpha=0.15,
            zorder=1,
        )
        ax.text(
            12,
            y,
            "Excluded from analysis",
            va="center",
            ha="center",
            fontsize=10,
            color="red",
            zorder=3,
        )

        # Formatting
        ax.set_xlim(0, 24)
        xticks = range(0, 25, 2)
        xticklabels = [
            f"{(12 + h) % 12 or 12} {'AM' if (12 + h) % 24 < 12 else 'PM'}"
            for h in xticks
        ]

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(f"Hour of Day (starting at 12PM — {self.timezone})")
        ax.set_yticks(range(len(days)))
        ax.set_yticklabels([str(day) for day in days])
        ax.set_title(f"{title} — Patient {self.patient}")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        # Legend
        legend_handles = [
            mpatches.Patch(
                facecolor=color,
                hatch="///" if bucket == non_worn_label else None,
                edgecolor="#888888" if bucket == non_worn_label else "none",
                label=bucket.replace("_", " ").capitalize(),
            )
            for bucket, color in bucket_colors.items()
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),  # move legend outside the right edge
            borderaxespad=0.0,
            frameon=True,
        )

        fig.tight_layout()
        return fig, ax

    def plot_combined_sleep_and_met(self, title="Sleep + MET Summary"):
        """
        Plots a combined figure of sleep/activity and MET Gantt charts stacked vertically,
        sharing the same x-axis (Hour of Day).

        Args:
            title (str): Combined plot title

        Returns:
            (fig, (ax1, ax2)): The matplotlib figure and axes tuple
        """
        df = self.plot_integrated_time

        if df.empty or "in_bed" not in df.columns:
            self.logger.error(
                "plot_combined_sleep_and_met error: No sleep data available."
            )
            return

        days = sorted(df["day"].unique())
        height = max(6, len(days) * 0.4)

        # Create a new figure with 2 rows and shared x-axis
        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            figsize=(14, height * 2),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1]},
        )

        # Plot both charts into the provided axes
        self.plot_integrated_sleep_activity_schedule(ax=ax1)
        self.plot_met_buckets_gantt(ax=ax2)

        # Set subplot titles if needed
        ax1.set_title("Sleep & Activity Schedule")
        ax2.set_title("MET Intensity")

        # Set shared figure title
        fig.suptitle(f"{title} — Patient {self.patient}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fig.savefig(self.plot_save_path, dpi=150)

        return fig, (ax1, ax2)

    def compute_average_sleep_hours(self) -> pd.DataFrame:
        """
        Computes average daily sleep duration from master_integrated_time.

        Automatically assigns 'state' based on sleep_phase_label and activity/met status.

        Returns:
            float: Average sleep duration in hours per day.
        """
        df = self.master_integrated_time.copy()
        if df.empty or "in_bed" not in df.columns:
            self.logger.error(
                "compute_average_sleep_hours error: No sleep data available."
            )
            return np.nan

        df["state"] = df.apply(self.assign_state, axis=1)

        sleep_states = {"deep_sleep", "light_sleep", "REM_sleep"}
        sleep_df = df[df["state"].isin(sleep_states)]

        daily_sleep = (
            sleep_df.groupby("shifted_day").size().rename("sleep_minutes").reset_index()
        )

        daily_sleep["sleep_hours"] = daily_sleep["sleep_minutes"] / 60.0
        avg_sleep_hours = daily_sleep["sleep_hours"].mean()

        return avg_sleep_hours

    def compute_lastday_sleep_hours(self) -> float:
        """
        Computes total sleep duration (in hours) for last day in self.timezone.
        Last day could be yesterday or day before yesterday

        Sleep includes deep, light, and REM phases. Excludes non-wear, awake, unidentified.

        Returns:
            float: Total sleep hours for last day.
        """
        df = self.master_integrated_time
        if df.empty or "in_bed" not in df.columns:
            return np.nan

        lastday = datetime.strptime(get_last_day(self.timezone), "%Y-%m-%d").date()

        df = df.copy()
        df["state"] = df.apply(self.assign_state, axis=1)

        sleep_states = {"deep_sleep", "light_sleep", "REM_sleep"}
        sleep_df = df[(df["shifted_day"] == lastday) & (df["state"].isin(sleep_states))]

        if sleep_df.empty:
            return np.nan

        sleep_minutes = len(sleep_df)
        sleep_hours = sleep_minutes / 60.0

        return sleep_hours

    def compute_sleep_hours(self) -> pd.DataFrame:
        """
        Computes total sleep duration (in hours) for each unique day in the data.

        This function processes the `self.plot_integrated_time` DataFrame. It groups
        the data by `shifted_day`, calculates the total minutes spent in sleep states
        (deep, light, REM), and converts this duration to hours.

        Sleep includes 'deep_sleep', 'light_sleep', and 'REM_sleep' phases.

        Returns:
            pd.DataFrame: A DataFrame with three columns: 'Date', 'TimeRange', and
                          'SleepHours'. Each row represents a unique day, its
                          corresponding total sleep hours, and the 12pm-12pm time
                          range. Returns an empty DataFrame if no sleep data is found.
        """
        df = self.plot_integrated_time

        if df.empty or "in_bed" not in df.columns or "shifted_day" not in df.columns:
            print(
                "Input DataFrame is empty or missing required columns ('in_bed', 'shifted_day')."
            )
            return pd.DataFrame({"Date": [], "TimeRange": [], "SleepHours": []})

        df = df.copy()
        # Assign a state (e.g., 'deep_sleep', 'awake') to each minute/row
        df["state"] = df.apply(self.assign_state, axis=1)
        sleep_states = {"deep_sleep", "light_sleep", "REM_sleep"}
        sleep_df = df[df["state"].isin(sleep_states)]

        if sleep_df.empty:
            return pd.DataFrame({"Date": [], "TimeRange": [], "SleepHours": []})

        # Group by day, count the number of minutes (rows), and store it
        daily_sleep_minutes = sleep_df.groupby("shifted_day").size()
        daily_sleep_hours = daily_sleep_minutes / 60.0
        result_df = daily_sleep_hours.reset_index(name="SleepHours")
        result_df.rename(columns={"shifted_day": "Date"}, inplace=True)
        result_df["TimeRange"] = result_df["Date"].apply(
            lambda d: f"{d.strftime('%Y-%m-%d')} 12:00 PM - {(d + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} 12:00 PM"
        )
        result_df = result_df[["Date", "TimeRange", "SleepHours"]]
        return result_df

    def plot_daily_sleep_hours(self):
        """
        Generates and saves a plot of daily sleep hours with an average line.
        This method computes the sleep data internally before plotting.
        """
        # Directly call the compute method to get the data
        sleep_data_df = self.compute_sleep_hours()

        if sleep_data_df.empty:
            print("[Warning] Sleep data is empty. Cannot generate plot.")
            return

        # Ensure 'Date' is in datetime format for proper plotting
        df = sleep_data_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        # Calculate yesterday's date
        yesterday = (
            datetime.now(tz=pytz.timezone(self.timezone)) - timedelta(days=1)
        ).date()

        # Filter out yesterday's data for average calculation
        df_for_avg = df[df["Date"].dt.date != yesterday]
        avg_sleep = df_for_avg["SleepHours"].mean()

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot daily sleep hours as a line with points
        # Separate plotting for yesterday's data if it exists
        yesterday_data = df[df["Date"].dt.date == yesterday]
        other_data = df[df["Date"].dt.date != yesterday]

        if not other_data.empty:
            ax.plot(
                other_data["Date"],
                other_data["SleepHours"],
                marker="o",
                linestyle="-",
                label="Daily Sleep",
                color="blue",  # Default color for other data
            )

        if not yesterday_data.empty:
            ax.plot(
                yesterday_data["Date"],
                yesterday_data["SleepHours"],
                marker="o",
                linestyle="--",  # Use a different linestyle for distinction
                label="Excluded from analysis",  # Label for the legend
                color="lightgray",  # Lighter color
                markersize=8,
                markeredgecolor="darkgray",
            )
            # Annotate yesterday's point with both value and exclusion text
            for index, row in yesterday_data.iterrows():
                ax.text(
                    row["Date"],
                    row["SleepHours"]
                    + 0.5,  # Adjust vertical position for the main annotation
                    f"{row['SleepHours']:.1f} \n(Excluded from analysis)",  # Combine text
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="darkgray",
                    fontweight="bold",
                )

        ax.axhline(
            y=avg_sleep,
            color="r",
            linestyle="--",
            label=f"Average (Excl. Yesterday): {avg_sleep:.1f} hrs",
        )

        # Annotate each data point (excluding yesterday's, as it's handled separately)
        for index, row in other_data.iterrows():
            ax.text(
                row["Date"],
                row["SleepHours"] - 0.3,
                f"{row['SleepHours']:.1f}",
                ha="center",
                va="top",
                fontsize=9,
                color="navy",
            )

        # --- Formatting the plot ---
        ax.set_title("Daily Sleep Duration", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sleep (Hours)", fontsize=12)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        fig.autofmt_xdate(rotation=45)

        ax.set_ylim(
            bottom=max(0, df["SleepHours"].min() - 1), top=df["SleepHours"].max() + 1
        )
        ax.set_xlim(
            [
                self.plot_integrated_time["shifted_day"].min(),
                self.plot_integrated_time["shifted_day"].max(),
            ]
        )

        plt.tight_layout()
        plt.savefig(self.sleep_hours_plot)
        plt.close(fig)
        print(f"Sleep plot generated at: {self.sleep_hours_plot}")

    def compute_average_steps(self, offset: int = 12) -> int:
        """
        Computes the average daily step count from self.activity.steps.

        Returns:
            int: Average steps per day (rounded to nearest integer), or np.nan if no data.
        """
        if not self.activity.steps:
            return np.nan

        df = pd.DataFrame(self.activity.steps)

        # Validate 'steps' and 'timestamp' keys exist
        if "steps" not in df.columns or "timestamp" not in df.columns:
            return np.nan

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(self.timezone)
        df["shifted_day"] = (df["timestamp"] - pd.Timedelta(hours=offset)).dt.date

        # Group by day and sum steps per day
        daily_steps = df.groupby("shifted_day")["steps"].sum()
        avg_steps = daily_steps.mean()

        return int(round(avg_steps))

    def compute_lastday_steps(self, offset: int = 12) -> int:
        """
        Returns the total step count for last day based on self.timezone.

        Returns:
            int: Total number of steps last day, or np.nan if no data.
        """
        if not self.activity.steps:
            return np.nan

        lastday = datetime.strptime(get_last_day(self.timezone), "%Y-%m-%d").date()

        # Convert to DataFrame
        df = pd.DataFrame(self.activity.steps)

        if "steps" not in df.columns or "timestamp" not in df.columns:
            return np.nan

        # Parse timestamps and extract day
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(self.timezone)
        df["day"] = df["timestamp"].dt.date
        df["shifted_day"] = (df["timestamp"] - pd.Timedelta(hours=offset)).dt.date

        df_lastday = df[df["shifted_day"] == lastday]

        if df_lastday.empty:
            return np.nan

        return int(df_lastday["steps"].sum())

    def compute_average_met(self) -> float:
        """
        Computes the average MET from self.master_integrated_time, ignoring NaN values.

        Returns:
            float: Average MET (rounded to 2 decimals), or np.nan if no valid data.
        """
        df = self.master_integrated_time

        if df.empty or "met" not in df.columns:
            return np.nan

        valid_met = df["met"].dropna()

        if valid_met.empty:
            return np.nan

        return valid_met.mean()

    def compute_lastday_met(self) -> float:
        """
        Computes the average MET for last day based on self.timezone.

        Ignores NaN MET values. Includes all other MET values, even ≤ 0.1.

        Returns:
            float: Average MET for yesterday (rounded to 2 decimals), or np.nan if no data.
        """
        df = self.master_integrated_time

        if df.empty or "met" not in df.columns:
            self.logger.error("compute_yesterdays_met error: No MET data available.")
            return np.nan

        lastday = datetime.strptime(get_last_day(self.timezone), "%Y-%m-%d").date()

        df = df.copy()

        df_lastday = df[df["shifted_day"] == lastday]

        if df_lastday.empty:
            return np.nan

        valid_met = df_lastday["met"].dropna()

        if valid_met.empty:
            return np.nan

        return valid_met.mean()

    def get_nan_sleep_dates(self) -> list:
        """
        Loop through each shifted_day and return those with NaN total sleep hours.

        Returns:
            list of strings: e.g. ["2025-04-23", "2025-04-24"]
        """
        df = self.master_integrated_time.copy()
        if df.empty or "in_bed" not in df.columns:
            self.logger.error("get_nan_sleep_dates error: No sleep data available.")
            return []

        sleep_states = {"deep_sleep", "light_sleep", "REM_sleep"}

        nan_days = []
        for day in df["shifted_day"].unique():
            day_df = df[df["shifted_day"] == day].copy()
            day_df["state"] = day_df.apply(self.assign_state, axis=1)
            sleep_minutes = day_df[day_df["state"].isin(sleep_states)]
            if sleep_minutes.empty:
                nan_days.append(str(day))

        return sorted(nan_days)

    def get_total_days(self) -> int:
        df = self.master_integrated_time.copy()
        return len(df["shifted_day"].unique())

    def get_last_non_wear_time(self, offset: int = 12) -> int:
        """
        Returns the total non-wear time for last day in seconds.
        Last day could be yesterday or day before yesterday.

        Returns:
            int: Total non-wear time in seconds, or np.nan if no data.
        """
        if not self.activity.nonweartime:
            return np.nan

        lastday = datetime.strptime(get_last_day(self.timezone), "%Y-%m-%d").date()

        df = pd.DataFrame(self.activity.nonweartime)

        if "non_wear_time" not in df.columns or "timestamp" not in df.columns:
            return np.nan

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(self.timezone)
        df["day"] = df["timestamp"].dt.date
        df["shifted_day"] = (df["timestamp"] - pd.Timedelta(hours=offset)).dt.date

        df_lastday = df[df["shifted_day"] == lastday]

        if df_lastday.empty:
            return np.nan

        return int(df_lastday["non_wear_time"].sum())

    def get_summary_stats(self) -> dict:
        """
        Returns a dictionary of daily summary statistics

        Returns:
            dict: Summary stats with float or np.nan values
        """
        return {
            PATIENT: self.patient,
            STUDY_NAME: self.study_name,
            TODAYS_DATE: get_todays_date(),
            LASTDAY_DATE: get_last_day(),
            AVERAGE_SLEEP_HOURS: self.compute_average_sleep_hours(),
            LASTDAY_SLEEP_HOURS: self.compute_lastday_sleep_hours(),
            LASTDAY_NON_WEAR_TIME_S: self.get_last_non_wear_time(),
            AVERAGE_STEPS: self.compute_average_steps(),
            LASTDAY_STEPS: self.compute_lastday_steps(),
            AVERAGE_MET: self.compute_average_met(),
            LASTDAY_MET: self.compute_lastday_met(),
            MISSING_SLEEP_DATES: self.get_nan_sleep_dates(),
            NUMBER_OF_NANSLEEP_DAYS: len(self.get_nan_sleep_dates()),
            NUMBER_OF_DAYS: self.get_total_days(),
        }

    def generate_warning_flags(self, summary: dict) -> dict:
        """
        Generate a simple dictionary of triggered warning flags from a patient's summary.

        Args:
            summary (dict): Output from get_summary_stats()

        Returns:
            dict: Dictionary of warning flags with boolean values
        """
        l_sleep = summary.get(LASTDAY_SLEEP_HOURS)
        avg_sleep = summary.get(AVERAGE_SLEEP_HOURS)
        l_steps = summary.get(LASTDAY_STEPS)
        avg_steps = summary.get(AVERAGE_STEPS)
        l_met = summary.get(LASTDAY_MET)
        avg_met = summary.get(AVERAGE_MET)
        nan_sleep_days = summary.get(NUMBER_OF_NANSLEEP_DAYS)
        l_non_wear_time = summary.get(LASTDAY_NON_WEAR_TIME_S)

        return {
            LASTDAY_SLEEP_NAN: pd.isna(l_sleep),
            # LASTDAY_STEPS_NAN: pd.isna(l_steps),
            LASTDAY_MET_NAN: pd.isna(l_met),
            AVERAGE_SLEEP_NAN: pd.isna(avg_sleep),
            # AVERAGE_STEPS_NAN: pd.isna(avg_steps),
            AVERAGE_MET_NAN: pd.isna(avg_met),
            # HAS_NAN_SLEEP_DAYS: nan_sleep_days > 0,
            LASTDAY_SLEEP_LESS_THAN_6: l_sleep < 6,
            LASTDAY_NON_WEAR_TIME_OVER_8: l_non_wear_time > 8 * 3600,
            SLEEP_VARIATION: (
                not pd.isna(l_sleep)
                and not pd.isna(avg_sleep)
                and avg_sleep > 0
                and (l_sleep < 0.75 * avg_sleep or l_sleep > 1.25 * avg_sleep)
            ),
            # STEPS_VARIATION: (
            #     not pd.isna(l_steps)
            #     and not pd.isna(avg_steps)
            #     and avg_steps > 0
            #     and (l_steps < 0.75 * avg_steps or l_steps > 1.25 * avg_steps)
            # ),
            MET_VARIATION: (
                not pd.isna(l_met)
                and not pd.isna(avg_met)
                and avg_met > 0
                and (l_met < 0.75 * avg_met or l_met > 1.25 * avg_met)
            ),
        }
