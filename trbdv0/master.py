import pandas as pd
from sleep import Sleep
from activity import Activity
import pytz
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm


class Master:
    def __init__(self, sleep: Sleep, activity: Activity, timezone="America/Chicago"):
        self.sleep = sleep
        self.activity = activity
        self.timezone = timezone
        self.logger = self.sleep.logger
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
        timeline = self.build_time_grid()

        # Sleep bedtimes
        df_in_bed = self.build_master_sleep_bedtimes()
        df_in_bed["in_bed"] = True

        # Sleep phases
        df_sleep_phase = self.build_master_sleep_phases()

        # Activity class
        df_activity = self.build_master_activity_phase()

        # MET data
        df_met = self.build_master_met()

        # Merge one by one
        df = timeline.merge(
            df_in_bed[["timestamp", "in_bed"]], on="timestamp", how="left"
        )
        df = df.merge(
            df_sleep_phase[["timestamp", "phase", "phase_label"]],
            on="timestamp",
            how="left",
        )
        df = df.merge(
            df_activity[["timestamp", "activity_class", "activity_label"]],
            on="timestamp",
            how="left",
        )
        df = df.merge(df_met[["timestamp", "met"]], on="timestamp", how="left")

        # 8. Fill default values
        df["in_bed"] = df["in_bed"].astype("boolean").fillna(False)

        # 9. Add day field
        df["day"] = df["timestamp"].dt.date

        # 10. Optional: rename for clarity
        df.rename(
            columns={"phase": "sleep_phase", "phase_label": "sleep_phase_label"},
            inplace=True,
        )

        return df

    def plot_integrated_sleep_activity_schedule(
        self, title="Sleep & Activity Schedule"
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

        def assign_state(row):
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

        df = self.master_integrated_time
        if df.empty:
            print("No data to plot.")
            return

        df = df.copy()
        # hour as float, since we are plotting on hours
        df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
        days = sorted(df["day"].unique())
        day_to_y = {day: i for i, day in enumerate(days)}

        fig, ax = plt.subplots(figsize=(14, max(6, len(days) * 0.4)))

        for day in days:
            day_df = df[(df["day"] == day) & (df["in_bed"] == True)].copy()

            if day_df.empty:
                continue

            day_df["state"] = day_df.apply(assign_state, axis=1)

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
        fig.savefig("debug_sleep_activity_plot.png", dpi=150)

        return fig, ax

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

    def plot_met_buckets_gantt(self, title="MET Binned Gantt Chart"):
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

        # Define color buckets
        # Step 1: Define value-based bucket labels
        def bucketize_met(val):
            if pd.isna(val) or val <= 0.1:
                return "non_wear/battery_dead"
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
            "non_wear/battery_dead": "#aaaaaa",  # gray, non-wear
            "0.1 - 0.5": "#deebf7",  # very light blue
            "0.5 - 1.0": "#9ecae1",  # light blue
            "1.0 - 2.0": "#6baed6",  # medium blue
            "2.0 - 3.0": "#4292c6",  # standard blue
            "3.0 - 4.5": "#2171b5",  # deep blue
            "> 4.5": "#084594",  # darkest blue
        }

        days = sorted(df["day"].unique())
        day_to_y = {day: i for i, day in enumerate(days)}

        fig, ax = plt.subplots(figsize=(14, max(6, len(days) * 0.4)))

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
                        edgecolor=(
                            "#888888" if bucket == "non_wear/battery_dead" else "none"
                        ),
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
                hatch="///" if bucket == "non_wear/battery_dead" else None,
                edgecolor="#888888" if bucket == "non_wear/battery_dead" else "none",
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
        fig.savefig(f"debug_met_bucket_gantt_{self.patient}.png", dpi=150)
        return fig, ax
