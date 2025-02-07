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
)

# TODO: how to define a day?


class SleepData:
    def __init__(self, patient, data: list) -> None:
        """Init class

        Args:
            patient (str): e.g. Percept010
            data (list): list of dicts of sleep info
        """
        # self.data looks like [{}, {}, ..., {}]
        self.data = data
        # self.sleep_data_one_day -> SleepDataOneDay object
        self.sleep_data_one_day = {}
        self.num_past_days = len(self.get_available_dates())
        self.patient = patient
        self.summary_stats = self.get_summary_stats()

    def get_num_past_days(self):
        return self.num_past_days

    def get_available_dates(self) -> set:
        """Return all recorded dates

        Returns:
            set: set("2023-07-15", "2023-07-16")
        """
        res = set()
        for sleep_chunk in self.data:
            res.add(sleep_chunk["day"])
        return res

    def get_sleep_data_on_date(self, date: str) -> list:
        """Return a list of sleep data on that day

        Args:
            day (str): "2023-09-14"

        Returns:
            list: [{}, {}, ...]
        """
        res = []
        for sleep_chunk in self.data:
            if sleep_chunk["day"] == date:
                res.append(sleep_chunk)
        return res

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
            sleep_counts[day]["steps"] += entry.get("steps", 0)
            sleep_counts[day]["average_met"] = calculate_average_met(entry)

            met_data = entry.get("met", {})
            sleep_counts[day]["met_interval"] = met_data.get("interval", 0)
            sleep_counts[day]["met_items"] = met_data.get("items", [])
            sleep_counts[day]["met_timestamp"] = met_data.get("timestamp", "")

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

        return {
            "patient": self.patient,
            "date_range": (
                (df.index.min(), df.index.max()) if not df.empty else (None, None)
            ),
            "today": today_date,
            "yesterday": yesterday_date,
            "yesterday_sleep": yesterday_sleep,
            "yesterday_steps": yesterday_steps,
            "average_sleep": df["Total Sleep"].mean() if not df.empty else np.nan,
            "average_steps": df["Step Count"].mean() if not df.empty else np.nan,
            "yesterday_average_met": yesterday_average_met,
            "average_met": df["Average MET"].mean() if not df.empty else np.nan,
            "sleep_df": df,
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
            ]
        )
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
        ax1.set_xlabel("Day")
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

        ax1.legend(title="Sleep Phases and Averages")

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

        # 1) Convert MET Timestamp to a proper datetime and add "day" and "hour" columns.
        df = self.summary_stats["sleep_df"]
        df["datetime"] = pd.to_datetime(df["MET Timestamp"])
        df["day"] = df["datetime"].dt.date

        ax = fig.add_subplot(gs[1, 0])
        unique_days = sorted(df["day"].unique())

        y_scale = 15

        y_ticks = []
        y_labels = []

        # Loop over each day
        for i, day in enumerate(unique_days):
            # If each day is guaranteed to appear exactly once in the df, just pick that row:
            row = df.loc[df["day"] == day].iloc[0]  # get the single row for this day

            # day_offset is the hour/minute at which the first MET item starts
            start_ts = row["datetime"]
            offset_h = start_ts.hour + start_ts.minute / 60.0 + start_ts.second / 3600.0

            # Convert the list of MET Items into a numpy array
            day_met = np.array(row["MET Items"], dtype=float)

            # For each MET item, figure out which minute it belongs to (0..N-1),
            # convert to hours by dividing by 60, then add offset_h
            hour_met = np.arange(len(day_met)) / 60.0 + offset_h

            # The y-value: i * y_scale is the vertical offset for day i
            # So we do (i*y_scale + day_met).
            if i == 0:
                # Provide label for the legend
                ax.plot(
                    hour_met,
                    i * y_scale + day_met,
                    color="mediumblue",
                    alpha=0.67,
                    label="Oura MET Score",
                )
            else:
                ax.plot(hour_met, i * y_scale + day_met, color="mediumblue", alpha=0.67)

            # Keep track of where to place the day label on the y-axis
            y_ticks.append(i * y_scale)
            y_labels.append(day)

        ax.set_xlim([4, 28])

        hours = list(range(4, 29))
        labels = []
        for h in hours:
            hour = h if h <= 12 else h - 12 if h < 24 else h - 24
            ampm = "am" if h < 12 or h >= 24 else "pm"
            labels.append(f"{hour}{ampm}")

        ax.set_xticks(hours)
        ax.set_xticklabels(labels, rotation=45)
        # Put each day on its own tick
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                out_dir,
                f"{self.patient}.png",
            )
        )
