import pandas as pd
from sleep import Sleep
from activity import Activity
import pytz
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import get_yesterdays_date, get_todays_date
from datetime import datetime
import numpy as np
import os


class Master:
    def __init__(self, sleep: Sleep, activity: Activity, timezone="America/Chicago"):
        self.sleep = sleep
        self.activity = activity
        self.timezone = timezone
        self.logger = self.sleep.logger
        self.patient = self.sleep.get_patient()
        self.plot_save_path = os.path.join(
            self.sleep.patient_out_dir, f"{self.patient}.png"
        )
        # e.g. 2025-04-01
        self.start_date = self.sleep.get_start_date()
        self.end_date = self.sleep.get_end_date()
        self.patient = self.sleep.get_patient()
        self.timegrid = self.build_time_grid()
        self.master_integrated_time = self.build_master_integrated_time()
        # self.plot_met_buckets_gantt()
        self.plot_integrated_sleep_activity_schedule()

    def build_time_grid(self) -> pd.DataFrame:
        """Builds a 1-minute resolution time grid from start_date to end_date (inclusive),
        with timestamps localized to the given timezone (e.g., America/Chicago)."""

        tz = pytz.timezone(self.timezone)

        start_ts = pd.to_datetime(self.start_date).replace(hour=0, minute=0, second=0)
        end_ts = pd.to_datetime(self.end_date).replace(
            hour=0, minute=0, second=0
        ) + pd.Timedelta(days=1)

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

        For each sleep segment, generates 1-minute timestamps from bedtime_start to bedtime_end,
        normalized to America/Chicago timezone, and associates each segment with the day
        defined as bedtime_start.date().

        Returns:
            pd.DataFrame: DataFrame with columns:
                - timestamp: 1-minute timestamps from bedtime_start to bedtime_end
                - source: 'sleep'
        """
        tz = pytz.timezone(self.timezone)
        all_rows = []

        for entry in self.sleep.get_bedtimes():
            start = pd.to_datetime(
                entry.get("bedtime_start"), errors="coerce", utc=True
            )
            end = pd.to_datetime(entry.get("bedtime_end"), errors="coerce", utc=True)

            # Skip if invalid
            if pd.isna(start) or pd.isna(end):
                continue

            # Floor to mins so we can merge with
            # timeline later with no errors
            start = start.tz_convert(tz).floor("min")
            end = end.tz_convert(tz).ceil("min")

            # Generate 1-minute timestamps
            timestamps = pd.date_range(
                start=start, end=end, freq="1min", inclusive="left"
            )

            all_rows.extend([{"timestamp": ts, "source": "sleep"} for ts in timestamps])

        return pd.DataFrame(all_rows)

    def build_master_sleep_phases(self) -> pd.DataFrame:
        """Builds a 1-minute resolution DataFrame of sleep phases from all sleep data entries.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - timestamp: 1-min resolution timestamps during sleep
                - phase: int code (1=deep, 2=light, 3=REM, 4=awake)
                - phase_label: human-readable sleep phase
                - source: 'sleep_phase'
        """
        tz = pytz.timezone(self.timezone)
        phase_map = {"1": "deep", "2": "light", "3": "REM", "4": "awake"}

        all_rows = []

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

            for i, char in enumerate(phase_string):
                if char not in phase_map:
                    continue  # skip unknown characters

                phase = int(char)
                label = phase_map[char]
                segment_start = start_time + timedelta(minutes=5 * i)

                # Create 5 one-minute timestamps per phase block
                for j in range(5):
                    ts = segment_start + timedelta(minutes=j)
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

    def build_master_integrated_time(self) -> pd.DataFrame:
        """Builds a unified 1-minute resolution timeline combining sleep, activity, and MET data.

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
        df = self.build_time_grid().copy()

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

        df = self.master_integrated_time
        if df.empty:
            print("No data to plot.")
            return
        # if there's no in_bed column
        # that means there's no sleep data
        if "in_bed" not in df.columns:
            print("No sleep data to plot.")
            return

        df = df.copy()
        # hour as float, since we are plotting on hours
        df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
        days = sorted(df["day"].unique())
        day_to_y = {day: i for i, day in enumerate(days)}

        height = max(6, len(days) * 0.4)
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, height))
        else:
            fig = ax.figure

        for day in days:
            day_df = df[(df["day"] == day) & (df["in_bed"] == True)].copy()

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
                        edgecolor=(
                            "#888888" if state == "not_worn/battery_dead" else None
                        ),
                        hatch="///" if state == "not_worn/battery_dead" else None,
                        linewidth=0.2,
                        zorder=2,
                    )

        # Final formatting
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel(f"Hour of Day ({self.timezone})")
        ax.set_yticks(range(len(days)))
        ax.set_yticklabels([str(day) for day in days])
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

    def get_segments(self, df: pd.DataFrame) -> list:
        """Returns (start_hour, end_hour) segments where rows are contiguous by 1-minute timestamps."""
        segments = []
        current_start = None
        previous_hour = None

        for i, timestamp in enumerate(df["timestamp"]):
            hour = timestamp.hour + timestamp.minute / 60.0

            if current_start is None:
                current_start = hour
            elif (timestamp - df["timestamp"].iloc[i - 1]).seconds > 60:
                # There's a gap in continuity
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
        df = self.master_integrated_time
        if df.empty or "met" not in df.columns:
            print("No MET data to plot.")
            return

        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
        df = df.sort_values(["day", "hour"])

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

        days = sorted(df["day"].unique())
        day_to_y = {day: i for i, day in enumerate(days)}

        height = max(6, len(days) * 0.4)
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, height))
        else:
            fig = ax.figure

        for day in days:
            day_df = df[df["day"] == day]
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

        # Formatting
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel(f"Hour of Day ({self.timezone})")
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
        df = self.master_integrated_time

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
        df = self.master_integrated_time
        if df.empty or "in_bed" not in df.columns:
            self.logger.error(
                "compute_average_sleep_hours error: No sleep data available."
            )
            return np.nan

        df = df.copy()
        df["state"] = df.apply(self.assign_state, axis=1)

        sleep_states = {"deep_sleep", "light_sleep", "REM_sleep"}
        sleep_df = df[df["state"].isin(sleep_states)]

        daily_sleep = (
            sleep_df.groupby("day").size().rename("sleep_minutes").reset_index()
        )

        daily_sleep["sleep_hours"] = daily_sleep["sleep_minutes"] / 60.0
        avg_sleep_hours = daily_sleep["sleep_hours"].mean()

        return avg_sleep_hours

    def compute_yesterday_sleep_hours(self) -> float:
        """
        Computes total sleep duration (in hours) for 'yesterday' in self.timezone.

        Sleep includes deep, light, and REM phases. Excludes non-wear, awake, unidentified.

        Returns:
            float: Total sleep hours for yesterday.
        """
        df = self.master_integrated_time
        if df.empty or "in_bed" not in df.columns:
            return np.nan

        yesterday = datetime.strptime(
            get_yesterdays_date(self.timezone), "%Y-%m-%d"
        ).date()

        df = df.copy()
        df["state"] = df.apply(self.assign_state, axis=1)

        sleep_states = {"deep_sleep", "light_sleep", "REM_sleep"}
        sleep_df = df[(df["day"] == yesterday) & (df["state"].isin(sleep_states))]

        sleep_minutes = len(sleep_df)
        sleep_hours = sleep_minutes / 60.0

        return sleep_hours

    def compute_average_steps(self) -> int:
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

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["day"] = df["timestamp"].dt.date

        # Group by day and sum steps per day
        daily_steps = df.groupby("day")["steps"].sum()
        avg_steps = daily_steps.mean()

        return int(round(avg_steps))

    def compute_yesterdays_steps(self) -> int:
        """
        Returns the total step count for yesterday based on self.timezone.

        Returns:
            int: Total number of steps yesterday, or np.nan if no data.
        """
        if not self.activity.steps:
            return np.nan

        yesterday = datetime.strptime(
            get_yesterdays_date(self.timezone), "%Y-%m-%d"
        ).date()

        # Convert to DataFrame
        df = pd.DataFrame(self.activity.steps)

        if "steps" not in df.columns or "timestamp" not in df.columns:
            return np.nan

        # Parse timestamps and extract day
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["day"] = df["timestamp"].dt.date

        # Filter for yesterday's entries
        df_yesterday = df[df["day"] == yesterday]

        if df_yesterday.empty:
            return np.nan

        # Sum steps for yesterday
        return int(df_yesterday["steps"].sum())

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

    def compute_yesterdays_met(self) -> float:
        """
        Computes the average MET for yesterday based on self.timezone.

        Ignores NaN MET values. Includes all other MET values, even ≤ 0.1.

        Returns:
            float: Average MET for yesterday (rounded to 2 decimals), or np.nan if no data.
        """
        df = self.master_integrated_time

        if df.empty or "met" not in df.columns:
            self.logger.error("compute_yesterdays_met error: No MET data available.")
            return np.nan

        yesterday = datetime.strptime(
            get_yesterdays_date(self.timezone), "%Y-%m-%d"
        ).date()

        df = df.copy()
        df["day"] = pd.to_datetime(df["day"]).dt.date

        df_yesterday = df[df["day"] == yesterday]

        if df_yesterday.empty:
            return np.nan

        valid_met = df_yesterday["met"].dropna()

        if valid_met.empty:
            return np.nan

        return valid_met.mean()

    def get_missing_sleep_dates(self) -> list:
        """Return a list of missing dates in the dates list

        Returns:
            list: ["2023-06-23", "2023-06-24"]
        """
        res = []
        for date in self.sleep.get_past_dates():
            patient_date_json = os.path.join(
                self.sleep.patient_in_dir, date, "sleep.json"
            )
            if not os.path.exists(patient_date_json):
                res.append(date)
        return res

    def get_summary_stats(self) -> dict:
        """
        Returns a dictionary of daily summary statistics including:
            - average_sleep_hours
            - yesterday_sleep_hours
            - average_steps
            - yesterday_steps
            - average_met
            - yesterday_met
            - missing_sleep_dates

        Returns:
            dict: Summary stats with float or np.nan values
        """
        return {
            "patient": self.patient,
            "todays_date": get_todays_date(),
            "yesterdays_date": get_yesterdays_date(),
            "average_sleep_hours": self.compute_average_sleep_hours(),
            "yesterday_sleep_hours": self.compute_yesterday_sleep_hours(),
            "average_steps": self.compute_average_steps(),
            "yesterday_steps": self.compute_yesterdays_steps(),
            "average_met": self.compute_average_met(),
            "yesterday_met": self.compute_yesterdays_met(),
            "missing_sleep_dates": self.get_missing_sleep_dates(),
            "number_of_noncompliance_days": len(self.get_missing_sleep_dates()),
            "number_of_days": len(self.sleep.get_past_dates()),
        }

    def generate_warning_flags(self, summary: dict) -> dict:
        """
        Generate a simple dictionary of triggered warning flags from a patient's summary.

        Args:
            summary (dict): Output from get_summary_stats()

        Returns:
            dict: {
                "missing_data": bool,
                "sleep_variation": bool,
                "steps_variation": bool,
                "met_variation": bool
            }
        """
        y_sleep = summary.get("yesterday_sleep_hours")
        avg_sleep = summary.get("average_sleep_hours")
        y_steps = summary.get("yesterday_steps")
        avg_steps = summary.get("average_steps")
        y_met = summary.get("yesterday_met")
        avg_met = summary.get("average_met")
        non_compliance_days = summary.get("number_of_noncompliance_days")

        return {
            "yesterday_sleep_nan": pd.isna(y_sleep),
            "yesterday_steps_nan": pd.isna(y_steps),
            "yesterday_met_nan": pd.isna(y_met),
            "average_sleep_nan": pd.isna(avg_sleep),
            "average_steps_nan": pd.isna(avg_steps),
            "average_met_nan": pd.isna(avg_met),
            "has_noncompliance_days": non_compliance_days > 0,
            "sleep_variation": (
                not pd.isna(y_sleep)
                and not pd.isna(avg_sleep)
                and avg_sleep > 0
                and (y_sleep < 0.75 * avg_sleep or y_sleep > 1.25 * avg_sleep)
            ),
            "steps_variation": (
                not pd.isna(y_steps)
                and not pd.isna(avg_steps)
                and avg_steps > 0
                and (y_steps < 0.75 * avg_steps or y_steps > 1.25 * avg_steps)
            ),
            "met_variation": (
                not pd.isna(y_met)
                and not pd.isna(avg_met)
                and avg_met > 0
                and (y_met < 0.75 * avg_met or y_met > 1.25 * avg_met)
            ),
        }
